"""Local JSON transport with ali-dev's per-response/per-object failure boundaries."""
import ast
import json
import re


class ResponseError(ValueError):
    pass


def transport_schema(task):
    identifier = {'type': ['string', 'integer']}
    if task == 'relations':
        field = 'relations'
        properties = {'source': identifier, 'relation': {'type': 'string'}, 'target': identifier}
    elif task == 'captions':
        field = 'captions'
        properties = {'id': identifier, 'name': {'type': 'string'},
                      'caption': {'type': ['string', 'null']}}
    elif task == 'consolidate':
        return {'type': 'object', 'properties': {'consolidated_caption': {'type': 'string'}},
                'required': ['consolidated_caption'], 'additionalProperties': False}
    else:
        raise ValueError(f'Unknown VLM task: {task}')
    return {'type': 'object', 'properties': {field: {'type': 'array', 'items': {
        'type': 'object', 'properties': properties, 'required': list(properties),
        'additionalProperties': False}}}, 'required': [field], 'additionalProperties': False}


def decode_response(task, text):
    fence = re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\n?```\s*', text, re.DOTALL)
    normalization = 'single_json_fence' if fence else None
    if fence:
        text = fence.group(1)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ResponseError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_keys), normalization, []
    except json.JSONDecodeError:
        if task == 'consolidate':
            raise ResponseError('Caption consolidation is not valid JSON')
    # Accept the original Python-list encoding as well as the JSON transport.
    match = re.search(r'\[.*?\]', text.replace('\n', ' '))
    if not match:
        raise ResponseError('No parsable response list found')
    try:
        return ast.literal_eval(match.group(0)), 'legacy_python_list', []
    except (ValueError, SyntaxError) as exc:
        if task != 'captions':
            raise ResponseError('Cannot parse relation list') from exc
        rows, rejected = [], []
        for fragment in re.findall(r'{.*?}', match.group(0)):
            try:
                row = ast.literal_eval(fragment)
                if not isinstance(row, dict):
                    raise ValueError('Caption item is not a dictionary')
                rows.append(row)
            except (ValueError, SyntaxError) as item_error:
                rejected.append({'reason': 'unparseable_caption_item', 'raw': fragment,
                                 'error': str(item_error)})
        if not rows:
            raise ResponseError('Cannot parse any caption items') from exc
        return rows, 'legacy_partial_caption_list', rejected


def filter_payload(task, payload, labels):
    issues = {'rejected_relations': [], 'rejected_captions': [],
              'missing_caption_ids': [], 'caption_name_differences': []}
    if task == 'consolidate':
        if not isinstance(payload, dict) or not isinstance(payload.get('consolidated_caption'), str):
            raise ResponseError('Missing/string-invalid consolidated_caption')
        return {'consolidated_caption': payload['consolidated_caption']}, issues
    field = 'relations' if task == 'relations' else 'captions'
    rows = payload.get(field) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ResponseError(f'{field} must be a list')

    def identifier(value):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ResponseError('Detection ID must be a string or integer')
        value = str(value)
        if value not in labels:
            raise ResponseError(f'Unknown detection ID: {value}')
        return value

    accepted = []
    for index, row in enumerate(rows):
        try:
            if task == 'relations':
                if isinstance(row, (list, tuple)) and len(row) == 3:
                    row = dict(zip(('source', 'relation', 'target'), row))
                if not isinstance(row, dict):
                    raise ResponseError('Relation item must contain source/relation/target')
                source, target = identifier(row.get('source')), identifier(row.get('target'))
                relation = row.get('relation')
                if relation not in ('on top of', 'under'):
                    raise ResponseError(f'Unsupported relation: {relation!r}')
                if source == target:
                    raise ResponseError('Self-loop relation')
                # Original process_edges does not reject repeated valid observations.
                accepted.append({'source': source, 'relation': relation, 'target': target})
            else:
                if not isinstance(row, dict):
                    raise ResponseError('Caption item must be a dictionary')
                key = identifier(row.get('id'))
                if 'caption' not in row or (row['caption'] is not None and not isinstance(row['caption'], str)):
                    raise ResponseError('Caption must be a string or null')
                name = row.get('name', labels[key])
                if not isinstance(name, str):
                    raise ResponseError('Caption name must be a string')
                if name != labels[key]:
                    issues['caption_name_differences'].append({'id': key, 'detector_name': labels[key],
                                                             'caption_name': name})
                accepted.append({'id': key, 'name': name, 'caption': row['caption']})
        except ResponseError as exc:
            issues['rejected_' + field].append({'index': index, 'item': row, 'reason': str(exc)})
    if task == 'captions':
        present = {row['id'] for row in accepted}
        issues['missing_caption_ids'] = [key for key in labels if key not in present]
    return {field: accepted}, issues


def result_value(task, payload):
    if task == 'relations':
        return [(row['source'], row['relation'], row['target']) for row in payload['relations']]
    if task == 'captions':
        return payload['captions']
    return payload['consolidated_caption']
