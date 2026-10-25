"""Strict local Ollama adapter for the three existing ali-dev VLM tasks."""
import base64
import hashlib
import json
import re
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

import httpx
import jsonschema
from openai import OpenAI

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


class LocalOllamaClient:
    def __init__(self, *, base_url="http://127.0.0.1:11434/v1", model="qwen3.5:9b",
                 log_dir="outputs/local-only/vlm", temperature=0.0, timeout=300,
                 max_tokens=4096, reuse_cache=True, transport=None):
        self.base_url = validate_local_endpoint(base_url)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.reuse_cache = reuse_cache
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
        with (self.log_dir / 'errors.jsonl').open('a') as stream:
            stream.write(json.dumps({"task": task, "error": str(exc), "type": type(exc).__name__,
                                     "model": self.model, "base_url": self.base_url}) + '\n')

    def close(self):
        self.sdk.close()

    def request(self, task, system_prompt, user_prompt, *, image_path=None, labels=None):
        structured_prompt = (system_prompt + "\nFor this local adapter return ONLY the JSON object "
            "defined by this schema. No markdown fences or other text. "
            "Preserve the provided IDs and exact label names. Use only visible evidence.\n"
            + json.dumps(task_schema(task, labels)))
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
        started = time.monotonic()
        try:
            if self.reuse_cache and cache_path.exists():
                cached = json.loads(cache_path.read_text())
                result = validate_output(task, cached['output'], labels)
                with (self.log_dir / 'cache_hits.jsonl').open('a') as stream:
                    stream.write(json.dumps({"task": task, "cache_file": cache_path.name,
                                             "model": self.model}) + '\n')
                return result
            response = self.sdk.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": structured_prompt}, {"role": "user", "content": content}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": task, "strict": True, "schema": record['schema']}},
                extra_body={"think": False, "reasoning_effort": "none"})
            record['response'] = response.model_dump(mode='json')
            record['system_prompt'] = structured_prompt
            if response.choices[0].finish_reason != 'stop':
                raise VLMError(f"Incomplete response: {response.choices[0].finish_reason}")
            payload, record['normalization'] = parse_json_response(response.choices[0].message.content)
            result = validate_output(task, payload, labels)
            record['output'] = payload
            record['elapsed_seconds'] = time.monotonic() - started
            cache_path.write_text(json.dumps(record, indent=2))
            return result
        except Exception as exc:
            record['error'] = str(exc)
            record['elapsed_seconds'] = time.monotonic() - started
            (self.log_dir / f"failed_{task}_{uuid.uuid4().hex}.json").write_text(json.dumps(record, indent=2))
            self._error(task, exc)
            raise VLMError(f"Local Ollama {task} failed; inspect {self.log_dir}: {exc}") from exc

    def relations(self, image_path, labels):
        names = label_map(labels)
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
        from conceptgraph.utils.vlm import system_prompt_consolidate_captions
        evidence = [cap['caption'] for cap in captions if cap['caption']]
        if not evidence:
            with (self.log_dir / 'skipped.jsonl').open('a') as stream:
                stream.write(json.dumps({"task": "consolidate", "reason": "no observed captions"}) + '\n')
            return None
        return self.request('consolidate', system_prompt_consolidate_captions,
                            "Captions for the SAME mapped object: " + json.dumps(evidence))
