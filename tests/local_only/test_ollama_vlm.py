import json

import httpx
import jsonschema
import pytest
from PIL import Image

from conceptgraph.utils.ollama_vlm import LocalOllamaClient, VLMError, label_map, parse_json_response, validate_output


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
        LocalOllamaClient(base_url=url, log_dir=tmp_path,
            transport=transport_handler(requests, '{}'))
    assert requests == []


def test_vision_schema_dummy_key_and_cache(tmp_path, image_path, monkeypatch):
    requests = []
    monkeypatch.setenv('OPENAI_API_KEY', 'SECRET-MUST-NEVER-BE-READ')
    monkeypatch.setenv('HTTPS_PROXY', 'http://203.0.113.1:9')
    body = json.dumps({"relations": [{"source": "7", "relation": "on top of", "target": "2"}]})
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, body))
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
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler([], body))
    with pytest.raises(VLMError):
        client.relations(image_path, ['7: cup', '2: table'])
    assert list(tmp_path.glob('failed_relations_*.json'))
    assert (tmp_path / 'errors.jsonl').exists()
    client.close()


def test_cloud_enabled_service_refused(tmp_path):
    requests = []
    with pytest.raises(VLMError, match='cloud is enabled'):
        LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, '{}', cloud_disabled=False))
    assert len(requests) == 1


@pytest.mark.parametrize('kind', ['stopped', 'redirect'])
def test_request_failure_never_falls_back(tmp_path, image_path, kind):
    requests = []
    def failure(request):
        if kind == 'stopped':
            raise httpx.ConnectError('Ollama stopped', request=request)
        return httpx.Response(307, headers={'Location': 'https://api.openai.com/v1/chat/completions'})
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, '{}', failure=failure))
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
        LocalOllamaClient(log_dir=tmp_path, transport=httpx.MockTransport(failure))
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
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, '{}'))
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
