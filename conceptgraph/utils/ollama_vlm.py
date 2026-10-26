"""Local Ollama transport with upstream response handling and explicit strict diagnostics."""
import base64
import hashlib
import json
import logging
import re
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid
from collections import Counter

import httpx
import jsonschema
from openai import APIError, OpenAI

from conceptgraph.utils.local_runtime import is_loopback


class VLMError(RuntimeError):
    pass


def parse_json_response(text):
    """Allow one whole JSON code fence; never repair or extract malformed content."""
    match = re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\n?```\s*', text, flags=re.DOTALL)
    normalization = 'single_json_fence' if match else None
    if match:
        text = match.group(1)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise VLMError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_keys), normalization


def validate_local_endpoint(endpoint):
    url = urlsplit(endpoint)
    if (url.scheme not in {"http", "https"} or not url.hostname or not is_loopback(url.hostname)
            or url.username or url.password or url.query or url.fragment or url.path.rstrip('/') != '/v1'):
        raise ValueError("Local-only VLM endpoint must be a loopback http(s) URL ending in /v1")
    _ = url.port  # Reject malformed ports at startup.
    return endpoint.rstrip('/')


def label_map(labels):
    result = {}
    for label in labels:
        identifier, name = label.split(":", 1)
        identifier, name = identifier.strip(), name.strip()
        if not identifier.isdecimal() or not name or identifier in result:
            raise VLMError(f"Invalid or duplicate detection label: {label!r}")
        result[identifier] = name
    return result


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def task_schema(task, labels=None):
    ids = list((labels or {}).keys())
    identifier = {"type": "string", "enum": ids}
    if task == "relations":
        item = object_schema({"source": identifier, "relation": {"type": "string", "enum": ["on top of", "under"]},
                              "target": identifier})
        return object_schema({"relations": {"type": "array", "items": item}})
    if task == "captions":
        item = {"oneOf": [object_schema({"id": {"type": "string", "const": identifier},
                              "name": {"type": "string", "const": name},
                              "caption": {"type": "string", "minLength": 1}})
                           for identifier, name in labels.items()]}
        return object_schema({"captions": {"type": "array", "items": item,
                                           "minItems": len(ids), "maxItems": len(ids)}})
    if task == "consolidate":
        return object_schema({"consolidated_caption": {"type": "string", "minLength": 1}})
    raise VLMError(f"Unknown task {task!r}")


def validate_output(task, payload, labels=None):
    jsonschema.validate(payload, task_schema(task, labels))
    if task == "relations":
        seen = set()
        for edge in payload['relations']:
            key = (edge['source'], edge['relation'], edge['target'])
            if edge['source'] == edge['target'] or key in seen:
                raise VLMError(f"Self-loop or duplicate relation: {key}")
            seen.add(key)
        return [tuple((edge['source'], edge['relation'], edge['target'])) for edge in payload['relations']]
    if task == "captions":
        by_id = {cap['id']: cap for cap in payload['captions']}
        if len(by_id) != len(payload['captions']) or set(by_id) != set(labels):
            raise VLMError("Caption IDs must cover the annotated detections exactly once")
        for identifier, cap in by_id.items():
            if cap['name'] != labels[identifier] or not cap['caption'].strip():
                raise VLMError(f"Caption label mismatch/empty text for ID {identifier}")
        return [by_id[identifier] for identifier in labels]
    if not payload['consolidated_caption'].strip():
        raise VLMError("Empty consolidated caption")
    return payload['consolidated_caption']


def drop_unmapped_relations(payload, labels):
    """Drop only unknown detection endpoints; malformed responses still raise."""
    shape = task_schema('relations', labels)
    for field in ('source', 'target'):
        shape['properties']['relations']['items']['properties'][field] = {'type': 'string'}
    jsonschema.validate(payload, shape)
    accepted, rejected = [], []
    for index, edge in enumerate(payload['relations']):
        unknown = {field: edge[field] for field in ('source', 'target') if edge[field] not in labels}
        if unknown:
            rejected.append({'index': index, 'relation': edge,
                             'reason': 'unknown_detection_id', 'unknown_endpoints': unknown})
        else:
            accepted.append(edge)
    return {'relations': accepted}, rejected


def validation_feedback(task, payload, labels, error):
    """Give the model the actual missing IDs, not a truncated dump of long captions."""
    details = {'task': task, 'violation': getattr(error, 'validator', type(error).__name__)}
    if task == 'captions' and isinstance(payload, dict) and isinstance(payload.get('captions'), list):
        rows = [row for row in payload['captions'] if isinstance(row, dict)]
        ids = [row.get('id') for row in rows]
        details['missing_ids_and_names'] = {key: name for key, name in labels.items() if key not in ids}
        details['unknown_ids'] = [key for key in ids if key not in labels]
        details['duplicate_ids'] = list(dict.fromkeys(key for key in ids if ids.count(key) > 1))
        details['wrong_names'] = {row['id']: {'expected': labels[row['id']], 'received': row.get('name')}
            for row in rows if row.get('id') in labels and row.get('name') != labels[row['id']]}
        details['wrong_fields_for_ids'] = [row.get('id') for row in rows
                                           if set(row) != {'id', 'name', 'caption'}]
        details['required_fields'] = ['id', 'name', 'caption']
        details['required_count'] = len(labels)
        details['received_count'] = len(payload['captions'])
    elif task == 'relations' and isinstance(payload, dict) and isinstance(payload.get('relations'), list):
        details['message'] = getattr(error, 'message', str(error))[:500]
        details['allowed_ids_and_names'] = labels
        details['invalid_endpoints'] = [
            {'row': index, 'field': field, 'value': row.get(field)}
            for index, row in enumerate(payload['relations']) if isinstance(row, dict)
            for field in ('source', 'target')
            if not isinstance(row.get(field), str) or row.get(field) not in labels]
        details['wrong_fields_for_rows'] = [index for index, row in enumerate(payload['relations'])
            if not isinstance(row, dict) or set(row) != {'source', 'relation', 'target'}]
        details['self_loop_rows'] = [index for index, row in enumerate(payload['relations'])
            if isinstance(row, dict) and row.get('source') == row.get('target')]
        details['instruction'] = ('Re-examine the image and generate relations afresh using only allowed IDs. '
            'Do not renumber objects or copy the previous relations. Omit uncertain relationships.')
    elif task == 'consolidate':
        details['required_output'] = 'Only consolidated_caption, a nonempty description of the observed object'
    else:
        details['message'] = (error.msg if isinstance(error, json.JSONDecodeError)
                              else getattr(error, 'message', str(error)))[:500]
    return json.dumps(details)


class LocalOllamaClient:
    def __init__(self, *, base_url="http://127.0.0.1:11434/v1", model="qwen3.5:9b",
                 log_dir="outputs/local-only/vlm", temperature=0.0, timeout=300,
                 max_tokens=4096, reuse_cache=True, transport=None, validation_retries=2,
                 invalid_relation_policy='error', response_policy='upstream'):
        self.base_url = validate_local_endpoint(base_url)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.reuse_cache = reuse_cache
        if response_policy not in ('upstream', 'strict'):
            raise ValueError('response_policy must be upstream or strict')
        self.response_policy = response_policy
        self.statistics = Counter()
        if not 0 <= validation_retries <= 4:
            raise ValueError('validation_retries must be between 0 and 4')
        self.validation_retries = validation_retries
        if invalid_relation_policy not in ('error', 'log_and_drop'):
            raise ValueError('invalid_relation_policy must be error or log_and_drop')
        self.invalid_relation_policy = invalid_relation_policy
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout, transport=transport)
        self.origin = self.base_url.rsplit('/v1', 1)[0]
        # No real API key is read. SDK retries and proxy/redirect fallbacks are disabled.
        self.sdk = OpenAI(base_url=self.base_url, api_key="ollama-local-only", http_client=self.http,
                          max_retries=0, timeout=timeout)
        try:
            status = self._get('/api/status')
            if status.get('cloud', {}).get('disabled') is not True:
                raise VLMError("Ollama cloud is enabled/unverified; require server-side OLLAMA_NO_CLOUD=1")
            tags = self._get('/api/tags')
            models = [entry for entry in tags['models'] if entry['name'] == model]
            if len(models) != 1 or models[0].get('remote_host') or models[0].get('remote_model'):
                raise VLMError(f"Model {model!r} is not a verified, already-installed local model")
            self.digest = models[0]['digest']
            (self.log_dir / 'server.json').write_text(json.dumps({"base_url": self.base_url,
                "model": model, "digest": self.digest, "status": status, "model_details": models[0]}, indent=2))
        except Exception as exc:
            self._error('startup', exc)
            self.http.close()
            raise VLMError(f"Local Ollama startup failed at {self.base_url}: {exc}") from exc

    def _get(self, path):
        response = self.http.get(self.origin + path)
        response.raise_for_status()
        return response.json()

    def _error(self, task, exc):
        if task in ('relations', 'captions', 'consolidate'):
            self.statistics['response_errors'] += 1
        with (self.log_dir / 'errors.jsonl').open('a') as stream:
            stream.write(json.dumps({"task": task, "error": str(exc), "type": type(exc).__name__,
                                     "model": self.model, "base_url": self.base_url}) + '\n')

    def close(self):
        self.sdk.close()

    def quality_summary(self):
        result = {key: self.statistics[key] for key in (
            'requests', 'cache_hits', 'response_errors', 'discarded_relations',
            'discarded_captions', 'missing_captions', 'caption_name_differences', 'empty_consolidations')}
        result['response_policy'] = self.response_policy
        result['has_issues'] = any(result[key] for key in (
            'response_errors', 'discarded_relations', 'discarded_captions',
            'missing_captions', 'empty_consolidations'))
        return result

    def _response_event(self, record, *, cache_hit=False):
        issues = record.get('issues', {})
        counts = {'discarded_relations': len(issues.get('rejected_relations', [])),
                  'discarded_captions': len(issues.get('rejected_captions', [])),
                  'missing_captions': len(issues.get('missing_caption_ids', [])),
                  'caption_name_differences': len(issues.get('caption_name_differences', [])),
                  'empty_consolidations': int(record.get('task') == 'consolidate'
                      and record.get('status') != 'failed'
                      and not record.get('output', {}).get('consolidated_caption', '').strip())}
        self.statistics.update(counts)
        self.statistics['cache_hits'] += int(cache_hit)
        context = {key: record[key] for key in ('task', 'model', 'digest', 'image_path', 'image_sha256', 'response_key')
                   if key in record}
        event = dict(context, status=record['status'], cache_hit=cache_hit, counts=counts)
        if 'error' in record:
            event['error'] = record['error']
        with (self.log_dir / 'response_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        for field, filename in (('rejected_relations', 'rejected_relations.jsonl'),
                                ('rejected_captions', 'rejected_captions.jsonl')):
            if issues.get(field) and not cache_hit:
                with (self.log_dir / filename).open('a') as stream:
                    for issue in issues[field]:
                        stream.write(json.dumps(dict(context, **issue)) + '\n')
        if issues.get('missing_caption_ids') and not cache_hit:
            with (self.log_dir / 'missing_captions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(context, ids=issues['missing_caption_ids'])) + '\n')

    def _request_upstream(self, task, system_prompt, user_prompt, *, image_path, labels):
        from conceptgraph.utils.vlm_response_policy import (
            ResponseError, decode_response, filter_payload, result_value, transport_schema)
        schema = transport_schema(task)
        encoding = {'relations': 'Encode each triple as an object with source, relation, target fields '
                    'inside a top-level relations array.',
                    'captions': 'Wrap the list of dictionaries in {"captions": [...]}.',
                    'consolidate': 'Return the consolidated_caption JSON object described above.'}[task]
        prompt = system_prompt + ('\nTransport format override (output encoding only): ' + encoding +
            ' Return only a JSON object, without markdown. The task instructions above are unchanged.')
        record = {'task': task, 'model': self.model, 'digest': self.digest, 'base_url': self.base_url,
                  'temperature': self.temperature, 'max_tokens': self.max_tokens, 'think': False,
                  'response_policy': 'upstream', 'system_prompt': prompt, 'user_prompt': user_prompt,
                  'labels': labels, 'schema': schema}
        messages = [{'role': 'system', 'content': prompt}]
        saved_messages = list(messages)
        if image_path:
            path = Path(image_path)
            data = path.read_bytes()
            if not data.startswith(b'\xff\xd8'):
                raise VLMError('Annotated input must be JPEG for data:image/jpeg encoding')
            record['image_path'] = str(path)
            record['image_sha256'] = hashlib.sha256(data).hexdigest()
            messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')}}]})
            saved_messages.append({'role': 'user', 'content': [{'type': 'image_file',
                'sha256': record['image_sha256'], 'mime_type': 'image/jpeg'}]})
        messages.append({'role': 'user', 'content': user_prompt})
        saved_messages.append(messages[-1])
        record['messages'] = saved_messages  # Exact layout; image bytes referenced by verified hash.
        key = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'image_path'},
                                       sort_keys=True).encode()).hexdigest()
        record['response_key'] = key
        cache_path = self.log_dir / f'{task}_{key}.json'
        if self.reuse_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            cached['response_key'] = key
            accepted, _ = filter_payload(task, cached['output'], labels)
            self._response_event(cached, cache_hit=True)
            return result_value(task, accepted)
        started = time.monotonic()
        try:
            self.statistics['requests'] += 1
            response = self.sdk.chat.completions.create(model=self.model, temperature=self.temperature,
                max_tokens=self.max_tokens, messages=messages,
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': task, 'strict': True, 'schema': schema}},
                extra_body={'think': False, 'reasoning_effort': 'none'})
            record['response'] = response.model_dump(mode='json')
            if not response.choices or response.choices[0].finish_reason != 'stop':
                raise ResponseError('Missing or incomplete model response')
            text = response.choices[0].message.content
            if not isinstance(text, str):
                raise ResponseError('Missing text in model response')
            payload, normalization, parse_issues = decode_response(task, text)
            record['raw_output'] = payload
            accepted, issues = filter_payload(task, payload, labels)
            issues['rejected_captions'].extend(parse_issues)
            record.update(output=accepted, issues=issues, normalization=normalization,
                          elapsed_seconds=time.monotonic() - started)
            record['status'] = ('partial' if any(issues[k] for k in (
                'rejected_relations', 'rejected_captions', 'missing_caption_ids')) else 'ok')
            self._response_event(record)
            cache_path.write_text(json.dumps(record, indent=2))
            return result_value(task, accepted)
        except (ResponseError, APIError, httpx.HTTPError) as exc:
            record.update(status='failed', error=str(exc), error_type=type(exc).__name__,
                          elapsed_seconds=time.monotonic() - started)
            (self.log_dir / f'failed_{task}_{uuid.uuid4().hex}.json').write_text(json.dumps(record, indent=2))
            self._error(task, exc)
            # A prohibited destination/redirect remains fatal; never hide a guard violation.
            cause, seen = exc, set()
            prohibited = getattr(exc, 'status_code', 0) in range(300, 400)
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                prohibited |= isinstance(cause, PermissionError)
                cause = cause.__cause__ or cause.__context__
            if prohibited:
                raise VLMError('Local-only transport policy rejected the request') from exc
            self._response_event(record)
            logging.warning('Local VLM %s failed; recorded and continuing this item: %s', task, str(exc)[:240])
            return '' if task == 'consolidate' else []

    def request(self, task, system_prompt, user_prompt, *, image_path=None, labels=None,
                _attempt=0, _feedback=None):
        """Dispatch without mixing upstream behavior with strict diagnostics."""
        if self.response_policy == 'upstream':
            return self._request_upstream(task, system_prompt, user_prompt,
                                          image_path=image_path, labels=labels)
        return self._request_strict(task, system_prompt, user_prompt, image_path=image_path,
                                    labels=labels, _attempt=_attempt, _feedback=_feedback)

    def _request_strict(self, task, system_prompt, user_prompt, *, image_path, labels,
                        _attempt, _feedback):
        """Keep legacy validation and regeneration available for explicit diagnostics."""
        structured_prompt = (system_prompt + "\nFor this local adapter return ONLY the JSON object "
            "defined by this schema. No markdown fences or other text. "
            "Preserve the provided IDs and exact label names. Use only visible evidence.\n"
            + json.dumps(task_schema(task, labels)))
        if task == 'consolidate':
            # Qwen 35B can echo an inline JSON schema instead of instance data.
            # The API still receives the schema; the text describes the output value.
            structured_prompt = (system_prompt + '\nReturn only an object with the property '
                '"consolidated_caption" and fill its string value with the actual object description. '
                'No markdown or schema metadata.')
        record = {"task": task, "model": self.model, "digest": self.digest, "base_url": self.base_url,
                  "temperature": self.temperature, "max_tokens": self.max_tokens, "think": False,
                  "system_prompt": structured_prompt, "user_prompt": user_prompt,
                  "labels": labels, "schema": task_schema(task, labels)}
        content = [{"type": "text", "text": user_prompt}]
        if image_path:
            path = Path(image_path)
            data = path.read_bytes()
            if not data.startswith(b'\xff\xd8'):
                raise VLMError("Annotated input must be JPEG for data:image/jpeg encoding")
            record['image_path'] = str(path)
            record['image_sha256'] = hashlib.sha256(data).hexdigest()
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(data).decode('ascii')}})
        key_record = {key: value for key, value in record.items() if key != 'image_path'}
        cache_key = hashlib.sha256(json.dumps(key_record, sort_keys=True).encode()).hexdigest()
        cache_path = self.log_dir / f"{task}_{cache_key}.json"
        record['attempt'] = _attempt + 1
        messages = [{"role": "system", "content": structured_prompt}, {"role": "user", "content": content}]
        if _feedback is not None:
            record['validation_feedback'] = _feedback
            # Re-present the original evidence with concrete corrections. Replaying a
            # hallucinated assistant answer anchored some vision replies to that answer
            # despite validation feedback.
            record['retry_prompt_strategy'] = 'fresh_evidence_with_validation_feedback'
            content.insert(0, {'type': 'text', 'text':
                'The previous answer failed validation: ' + _feedback['error'] +
                '. Re-examine the SAME image/evidence and generate a fresh JSON response. '
                'Use exactly the provided IDs and names; no duplicate IDs or self-loops. '
                'Return instance data, never schema metadata. Keep each caption concise. '
                'Return only the required JSON object; do not omit required objects.'})
        started = time.monotonic()
        payload = None
        try:
            if self.reuse_cache and cache_path.exists() and _attempt == 0:
                cached = json.loads(cache_path.read_text())
                cached_payload = (cached.get('raw_output', cached['output'])
                    if task == 'relations' and self.invalid_relation_policy == 'error' else cached['output'])
                result = validate_output(task, cached_payload, labels)
                with (self.log_dir / 'cache_hits.jsonl').open('a') as stream:
                    stream.write(json.dumps({"task": task, "cache_file": cache_path.name,
                        "model": self.model, "rejected_relations": len(cached.get('rejected_relations', []))}) + '\n')
                return result
            response = self.sdk.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": {
                    "name": task, "strict": True, "schema": record['schema']}},
                extra_body={"think": False, "reasoning_effort": "none"})
            record['response'] = response.model_dump(mode='json')
            record['system_prompt'] = structured_prompt
            if response.choices[0].finish_reason != 'stop':
                raise VLMError(f"Incomplete response: {response.choices[0].finish_reason}")
            payload, record['normalization'] = parse_json_response(response.choices[0].message.content)
            accepted_payload = payload
            if task == 'relations' and self.invalid_relation_policy == 'log_and_drop':
                accepted_payload, rejected = drop_unmapped_relations(payload, labels)
                record['invalid_relation_policy'] = self.invalid_relation_policy
                record['rejected_relations'] = rejected
                if rejected:
                    record['raw_output'] = payload
                    context = {key: record[key] for key in ('task', 'model', 'digest', 'attempt',
                               'image_path', 'image_sha256') if key in record}
                    with (self.log_dir / 'rejected_relations.jsonl').open('a') as stream:
                        for rejection in rejected:
                            stream.write(json.dumps(dict(context, **rejection, response_key=cache_key)) + '\n')
                    logging.warning('Dropped %d relations with unknown detection IDs; retained %d. See %s',
                        len(rejected), len(accepted_payload['relations']), self.log_dir / 'rejected_relations.jsonl')
            result = validate_output(task, accepted_payload, labels)
            record['output'] = accepted_payload
            record['elapsed_seconds'] = time.monotonic() - started
            cache_path.write_text(json.dumps(record, indent=2))
            return result
        except Exception as exc:
            record['error'] = str(exc)
            record['elapsed_seconds'] = time.monotonic() - started
            (self.log_dir / f"failed_{task}_{uuid.uuid4().hex}.json").write_text(json.dumps(record, indent=2))
            self._error(task, exc)
            if isinstance(exc, (json.JSONDecodeError, jsonschema.ValidationError, VLMError)) and _attempt < self.validation_retries:
                error = validation_feedback(task, payload, labels, exc)
                previous = record.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content') or ''
                logging.warning('Local VLM %s validation failed on attempt %d; retrying locally: %s',
                                task, _attempt + 1, error[:180])
                return self.request(task, system_prompt, user_prompt, image_path=image_path, labels=labels,
                    _attempt=_attempt + 1, _feedback={'error': error, 'previous_response': previous})
            raise VLMError(f"Local Ollama {task} failed; inspect {self.log_dir}: {exc}") from exc

    def relations(self, image_path, labels):
        names = label_map(labels)
        if self.response_policy == 'upstream':
            from conceptgraph.utils.vlm import system_prompt_only_top
            return self.request('relations', system_prompt_only_top,
                f'Here is the list of labels for the annotations of the objects in the image: {labels}. '
                'Please describe the spatial relationships between the objects in the image.',
                image_path=image_path, labels=names)
        prompt = ("Identify physical support relationships between the objects in an annotated image. "
            "Each colored contour has a numeric ID. Use only provided IDs. "
            "For source physically resting on target, use 'on top of'; for source physically below target, "
            "use 'under'. Omit uncertain relationships and self-loops. Return an object with a 'relations' "
            "array of objects, each with keys 'source', 'relation', 'target'. The array may be empty.")
        return self.request('relations', prompt,
            f"Annotated labels: {json.dumps(names)}. Report physically supported on top of/under relations; "
            "omit uncertain relations. Return a JSON object with a relations array.", image_path=image_path, labels=names)

    def captions(self, image_path, labels):
        names = label_map(labels)
        if self.response_policy == 'upstream':
            from conceptgraph.utils.vlm import system_prompt_captions
            return self.request('captions', system_prompt_captions,
                f'Here is the list of labels for the annotations of the objects in the image: {labels}. '
                'Please accurately caption the objects in the image.', image_path=image_path, labels=names)
        prompt = ("Caption objects in the annotated image. Each colored contour has a numeric ID. "
            "Caption every provided ID exactly once with a concise description grounded in visible appearance. "
            "The detector's names may be inaccurate; preserve the exact provided name in the 'name' field "
            "and describe visible appearance in 'caption'. Resolve ambiguous numeric text using the provided "
            "ID-to-name list; never substitute another ID's object. Return an object with a 'captions' array "
            "of objects with keys 'id', 'name', 'caption'.")
        return self.request('captions', prompt,
            f"Annotated labels: {json.dumps(names)}. Caption each ID exactly once. "
            "Return a JSON object with a captions array.", image_path=image_path, labels=names)

    def consolidate(self, captions):
        if self.response_policy == 'upstream':
            from conceptgraph.utils.vlm import system_prompt_consolidate_captions
            text = '\n'.join(cap['caption'] for cap in captions if cap['caption'] is not None)
            return self.request('consolidate', system_prompt_consolidate_captions,
                f'Here are several captions for the same object:\n{text}\n\n'
                'Please consolidate these into a single, clear caption that accurately describes the object.')
        evidence = [cap['caption'] for cap in captions if cap['caption']]
        if not evidence:
            with (self.log_dir / 'skipped.jsonl').open('a') as stream:
                stream.write(json.dumps({"task": "consolidate", "reason": "no observed captions"}) + '\n')
            return None
        prompt = ("Consolidate observed captions for the SAME physical object into a single concise, "
                  "accurate description. Keep shared visual details, remove noise and outliers. "
                  "Return JSON instance data with one string field 'consolidated_caption'. "
                  "Its value must describe the object, never describe or repeat the JSON schema.")
        return self.request('consolidate', prompt,
            "Observed captions: " + json.dumps(evidence) +
            "\nFill consolidated_caption with a factual sentence describing this object.")
