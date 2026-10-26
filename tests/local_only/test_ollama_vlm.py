import json
from functools import partial

import httpx
import jsonschema
import pytest
from PIL import Image

from conceptgraph.utils.ollama_vlm import LocalOllamaClient, VLMError, label_map, parse_json_response, validate_output
from conceptgraph.utils.ollama_vlm import validation_feedback

# Preserve the historical strict-mode contract as an explicit diagnostic mode.
StrictClient = partial(LocalOllamaClient, response_policy="strict")


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "input.jpg"
    Image.new('RGB', (16, 16), 'white').save(path)
    return path


def transport_handler(requests, content, *, cloud_disabled=True, failure=None):
    def handle(request):
        requests.append(request)
        assert request.url.host == '127.0.0.1'
        if request.url.path == '/api/status':
            return httpx.Response(200, json={"cloud": {"disabled": cloud_disabled}})
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b", "digest": "test-local-digest"}]})
        if failure:
            return failure(request)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "qwen3.5:9b", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}]})
    return httpx.MockTransport(handle)


@pytest.mark.parametrize('url', ['https://api.openai.com/v1', 'https://ollama.com/v1',
    'http://192.168.1.10:11434/v1', 'http://127.0.0.1.evil/v1',
    'http://user:secret@127.0.0.1/v1', 'http://127.0.0.1/v1?redirect=1'])
def test_external_or_ambiguous_endpoint_rejected_without_request(url, tmp_path):
    requests = []
    with pytest.raises(ValueError):
        StrictClient(base_url=url, log_dir=tmp_path,
            transport=transport_handler(requests, '{}'))
    assert requests == []


def test_vision_schema_dummy_key_and_cache(tmp_path, image_path, monkeypatch):
    requests = []
    monkeypatch.setenv('OPENAI_API_KEY', 'SECRET-MUST-NEVER-BE-READ')
    monkeypatch.setenv('HTTPS_PROXY', 'http://203.0.113.1:9')
    body = json.dumps({"relations": [{"source": "7", "relation": "on top of", "target": "2"}]})
    client = StrictClient(log_dir=tmp_path, transport=transport_handler(requests, body))
    result = client.relations(image_path, ['7: cup', '2: table'])
    assert result == [('7', 'on top of', '2')]
    assert client.relations(image_path, ['7: cup', '2: table']) == result
    assert len(requests) == 3  # status, tags, one inference; second inference is cached
    request = requests[-1]
    assert request.headers['authorization'] == 'Bearer ollama-local-only'
    payload = json.loads(request.content)
    assert payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert payload['temperature'] == 0 and payload['think'] is False
    assert payload['response_format']['type'] == 'json_schema'
    for path in tmp_path.glob('*.json'):
        assert 'SECRET-MUST-NEVER-BE-READ' not in path.read_text()
        assert 'data:image' not in path.read_text()
    client.close()


@pytest.mark.parametrize('body', ["{broken", '{"relations":[{"source":"99","relation":"under","target":"2"}]}',
    '{"relations":[{"source":"7","relation":"under","target":"7"}]}',
    '{"relations":[{"source":"7","relation":"near","target":"2"}]}'])
def test_bad_response_is_logged_and_raised(body, tmp_path, image_path):
    client = StrictClient(log_dir=tmp_path, transport=transport_handler([], body))
    with pytest.raises(VLMError):
        client.relations(image_path, ['7: cup', '2: table'])
    assert list(tmp_path.glob('failed_relations_*.json'))
    assert (tmp_path / 'errors.jsonl').exists()
    client.close()


def test_cloud_enabled_service_refused(tmp_path):
    requests = []
    with pytest.raises(VLMError, match='cloud is enabled'):
        StrictClient(log_dir=tmp_path, transport=transport_handler(requests, '{}', cloud_disabled=False))
    assert len(requests) == 1


@pytest.mark.parametrize('kind', ['stopped', 'redirect'])
def test_request_failure_never_falls_back(tmp_path, image_path, kind):
    requests = []
    def failure(request):
        if kind == 'stopped':
            raise httpx.ConnectError('Ollama stopped', request=request)
        return httpx.Response(307, headers={'Location': 'https://api.openai.com/v1/chat/completions'})
    client = StrictClient(log_dir=tmp_path, transport=transport_handler(requests, '{}', failure=failure))
    with pytest.raises(VLMError):
        client.relations(image_path, ['7: cup', '2: table'])
    assert len(requests) == 3 and all(req.url.host == '127.0.0.1' for req in requests)
    client.close()


def test_stopped_at_startup_fails_once(tmp_path):
    requests = []
    def failure(request):
        requests.append(request)
        raise httpx.ConnectError('Connection refused', request=request)
    with pytest.raises(VLMError):
        StrictClient(log_dir=tmp_path, transport=httpx.MockTransport(failure))
    assert len(requests) == 1
    assert (tmp_path / 'errors.jsonl').exists()


def test_captions_are_reordered_and_invalid_names_are_rejected():
    labels = label_map(['7: coffee cup', '2: table'])
    captions = {'captions': [{'id': '2', 'name': 'table', 'caption': 'Wooden table.'},
                             {'id': '7', 'name': 'coffee cup', 'caption': 'White cup.'}]}
    assert [cap['id'] for cap in validate_output('captions', captions, labels)] == ['7', '2']
    captions['captions'][1]['name'] = 'hallucinated object'
    with pytest.raises((VLMError, jsonschema.ValidationError)):
        validate_output('captions', captions, labels)


def test_empty_consolidation_skips_request(tmp_path):
    requests = []
    client = StrictClient(log_dir=tmp_path, transport=transport_handler(requests, '{}'))
    assert client.consolidate([{'caption': None}]) is None
    assert len(requests) == 2
    assert (tmp_path / 'skipped.jsonl').exists()
    client.close()


def test_only_a_single_complete_json_fence_is_normalized():
    assert parse_json_response('```json\n{"relations": []}\n```') == ({'relations': []}, 'single_json_fence')
    for invalid in ('explanation ```json\n{}\n```', '```json\n{broken}\n```',
                    '{"relations": [], "relations": [1]}'):
        with pytest.raises((ValueError, VLMError)):
            parse_json_response(invalid)


def test_consolidation_schema_echo_is_logged_and_rejected(tmp_path):
    schema_echo = json.dumps({'type': 'object', 'properties': {'consolidated_caption': {'type': 'string'}}})
    client = StrictClient(log_dir=tmp_path, transport=transport_handler([], schema_echo))
    with pytest.raises(VLMError):
        client.consolidate([{'caption': 'A beige stool.'}])
    assert list(tmp_path.glob('failed_consolidate_*.json'))
    client.close()


def test_validation_retry_is_local_logged_and_reusable(tmp_path, image_path):
    requests = []
    count = 0
    def reply(request):
        nonlocal count
        count += 1
        content = '{broken' if count == 1 else json.dumps({'relations': [
            {'source': '7', 'relation': 'on top of', 'target': '2'}]})
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'qwen3.5:9b', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': content}}]})
    client = StrictClient(log_dir=tmp_path, validation_retries=2,
        transport=transport_handler(requests, '{}', failure=reply))
    expected = [('7', 'on top of', '2')]
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert count == 2 and len(requests) == 4
    assert len(list(tmp_path.glob('failed_relations_*.json'))) == 1
    saved = json.loads(next(tmp_path.glob('relations_*.json')).read_text())
    assert saved['attempt'] == 2 and saved['validation_feedback']['previous_response'] == '{broken'
    messages = json.loads(requests[-1].content)['messages']
    assert [message['role'] for message in messages] == ['system', 'user']
    assert messages[-1]['content'][-1]['image_url'] == json.loads(
        requests[-2].content)['messages'][-1]['content'][-1]['image_url']
    assert 'generate a fresh JSON response' in messages[-1]['content'][0]['text']
    client.close()


def test_missing_caption_feedback_names_the_omitted_object():
    labels = {'0': 'coffee kettle', '28': 'ottoman'}
    payload = {'captions': [{'id': '0', 'name': 'coffee kettle', 'caption': 'visible object ' * 500}]}
    with pytest.raises(jsonschema.ValidationError) as result:
        validate_output('captions', payload, labels)
    feedback = json.loads(validation_feedback('captions', payload, labels, result.value))
    assert feedback['missing_ids_and_names'] == {'28': 'ottoman'}
    assert feedback['required_count'] == 2 and feedback['received_count'] == 1
    assert len(json.dumps(feedback)) < 1000


def test_relation_feedback_lists_all_invalid_endpoints():
    labels = {'0': 'pillow', '7': 'couch', '17': 'lamp'}
    payload = {'relations': [
        {'source': '8', 'relation': 'on top of', 'target': '7'},
        {'source': '16', 'relation': 'on top of', 'target': '13'}]}
    with pytest.raises(jsonschema.ValidationError) as result:
        validate_output('relations', payload, labels)
    feedback = json.loads(validation_feedback('relations', payload, labels, result.value))
    assert feedback['allowed_ids_and_names'] == labels
    assert feedback['invalid_endpoints'] == [
        {'row': 0, 'field': 'source', 'value': '8'},
        {'row': 1, 'field': 'source', 'value': '16'},
        {'row': 1, 'field': 'target', 'value': '13'}]
    assert 'Do not renumber' in feedback['instruction']


def test_relation_feedback_handles_invalid_endpoint_types():
    payload = {'relations': [{'source': ['7'], 'relation': 'under', 'target': '2'}]}
    feedback = json.loads(validation_feedback('relations', payload, {'7': 'cup', '2': 'table'},
                                             VLMError('invalid endpoint')))
    assert feedback['invalid_endpoints'] == [{'row': 0, 'field': 'source', 'value': ['7']}]


def test_unknown_relation_is_logged_and_dropped_without_losing_valid_edges(tmp_path, image_path):
    requests = []
    payload = {'relations': [
        {'source': '7', 'relation': 'on top of', 'target': '2'},
        {'source': '99', 'relation': 'under', 'target': '2'}]}
    client = StrictClient(log_dir=tmp_path, invalid_relation_policy='log_and_drop',
        transport=transport_handler(requests, json.dumps(payload)))
    expected = [('7', 'on top of', '2')]
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert len(requests) == 3  # No inference retry; second call reuses the filtered cache.
    saved = json.loads(next(tmp_path.glob('relations_*.json')).read_text())
    assert saved['raw_output'] == payload
    assert saved['output'] == {'relations': payload['relations'][:1]}
    rejected = [json.loads(line) for line in (tmp_path/'rejected_relations.jsonl').read_text().splitlines()]
    assert len(rejected) == 1 and rejected[0]['reason'] == 'unknown_detection_id'
    assert rejected[0]['unknown_endpoints'] == {'source': '99'}
    assert rejected[0]['image_path'] == str(image_path)
    assert rejected[0]['relation'] == payload['relations'][1]
    assert not list(tmp_path.glob('failed_relations_*.json'))
    client.close()
    strict = StrictClient(log_dir=tmp_path, validation_retries=0,
        transport=transport_handler([], json.dumps(payload)))
    with pytest.raises(VLMError):
        strict.relations(image_path, ['7: cup', '2: table'])
    strict.close()


def test_all_unknown_relations_remain_explicit_rejections(tmp_path, image_path):
    requests = []
    payload = {'relations': [{'source': '99', 'relation': 'under', 'target': '98'}]}
    client = StrictClient(log_dir=tmp_path, invalid_relation_policy='log_and_drop',
        transport=transport_handler(requests, json.dumps(payload)))
    assert client.relations(image_path, ['7: cup', '2: table']) == []
    saved = json.loads(next(tmp_path.glob('relations_*.json')).read_text())
    assert saved['raw_output'] == payload and len(saved['rejected_relations']) == 1
    assert saved['rejected_relations'][0]['unknown_endpoints'] == {'source': '99', 'target': '98'}
    assert len(requests) == 3 and (tmp_path/'rejected_relations.jsonl').exists()
    client.close()


@pytest.mark.parametrize('body', ['{broken', '{"relations":"not an array"}',
    '{"relations":[{"source":"7","relation":"near","target":"2"}]}'])
def test_drop_unknown_ids_does_not_hide_other_response_errors(tmp_path, image_path, body):
    client = StrictClient(log_dir=tmp_path, invalid_relation_policy='log_and_drop',
        validation_retries=0, transport=transport_handler([], body))
    with pytest.raises(VLMError):
        client.relations(image_path, ['7: cup', '2: table'])
    assert list(tmp_path.glob('failed_relations_*.json'))
    assert not list(tmp_path.glob('relations_*.json'))
    client.close()


def test_unknown_id_policy_does_not_suppress_caption_errors(tmp_path, image_path):
    payload = {'captions': [{'id': '7', 'name': 'cup', 'caption': 'A white cup.'}]}
    client = StrictClient(log_dir=tmp_path, invalid_relation_policy='log_and_drop',
        validation_retries=0, transport=transport_handler([], json.dumps(payload)))
    with pytest.raises(VLMError):
        client.captions(image_path, ['7: cup', '2: table'])
    assert list(tmp_path.glob('failed_captions_*.json'))
    client.close()
