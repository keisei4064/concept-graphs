import json

import httpx
import pytest
from PIL import Image

from conceptgraph.utils.ollama_vlm import LocalOllamaClient, VLMError
from conceptgraph.utils.vlm_response_policy import decode_response, filter_payload
from test_ollama_vlm import transport_handler


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / 'image.jpg'
    Image.new('RGB', (16, 16), 'white').save(path)
    return path


def test_partial_relations_continue_without_retry_and_keep_original_duplicate_behavior(tmp_path, image_path):
    requests = []
    good = {'source': '7', 'relation': 'on top of', 'target': '2'}
    rows = [good, {'source': '99', 'relation': 'under', 'target': '2'},
            {'source': '7', 'relation': 'under', 'target': '7'},
            {'source': '7', 'relation': 'near', 'target': '2'}, None, good]
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, json.dumps({'relations': rows})))
    expected = [('7', 'on top of', '2'), ('7', 'on top of', '2')]
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert client.relations(image_path, ['7: cup', '2: table']) == expected
    assert len(requests) == 3
    assert client.quality_summary()['cache_hits'] == 1
    assert client.quality_summary()['discarded_relations'] == 8
    assert len((tmp_path/'rejected_relations.jsonl').read_text().splitlines()) == 4
    record = json.loads(next(tmp_path.glob('relations_*.json')).read_text())
    assert record['status'] == 'partial' and record['raw_output']['relations'] == rows
    client.close()


def test_caption_name_can_correct_detector_and_missing_caption_uses_existing_null_padding(tmp_path, image_path):
    from conceptgraph.slam.utils import filter_captions
    payload = {'captions': [{'id': '7', 'name': 'bowl', 'caption': 'A ceramic bowl.'},
                             {'id': '99', 'name': 'other', 'caption': 'Not an annotated object.'}]}
    requests = []
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, json.dumps(payload)))
    result = client.captions(image_path, ['7: cup', '2: table'])
    padded = filter_captions(result, ['cup 7', 'table 2'])
    assert padded == [payload['captions'][0], {'id': '2', 'name': 'table', 'caption': None}]
    summary = client.quality_summary()
    assert summary['missing_captions'] == 1 and summary['caption_name_differences'] == 1
    assert summary['discarded_captions'] == 1 and len(requests) == 3
    record = json.loads(next(tmp_path.glob('captions_*.json')).read_text())
    assert record['issues']['missing_caption_ids'] == ['2']
    client.close()


@pytest.mark.parametrize('task,body', [('relations', '{broken'), ('captions', '{"captions": 42}'),
                                      ('consolidate', '{"type":"object"}')])
def test_bad_whole_response_is_recorded_as_failure_and_not_cached_as_success(tmp_path, image_path, task, body):
    requests = []
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, body))
    if task == 'consolidate':
        result = client.consolidate([{'caption': 'A white cup.'}])
    else:
        result = getattr(client, task)(image_path, ['7: cup', '2: table'])
    assert result == ('' if task == 'consolidate' else [])
    assert len(requests) == 3
    assert list(tmp_path.glob(f'failed_{task}_*.json'))
    assert not list(tmp_path.glob(f'{task}_*.json'))
    assert client.quality_summary()['response_errors'] == 1
    event = json.loads((tmp_path/'response_events.jsonl').read_text().splitlines()[-1])
    assert event['status'] == 'failed'
    client.close()


def test_inference_connection_failure_is_local_and_recorded_without_stopping_mapping(tmp_path, image_path):
    requests = []
    def fail(request):
        raise httpx.ConnectError('Stopped after startup', request=request)
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, '{}', failure=fail))
    assert client.relations(image_path, ['7: cup', '2: table']) == []
    assert len(requests) == 3 and all(r.url.host == '127.0.0.1' for r in requests)
    assert client.quality_summary()['response_errors'] == 1
    client.close()


@pytest.mark.parametrize('failure_kind', ['redirect', 'guard'])
def test_transport_security_failures_remain_fatal(tmp_path, image_path, failure_kind):
    requests = []
    def fail(request):
        if failure_kind == 'guard':
            raise PermissionError('Local-only network guard rejected external communication')
        return httpx.Response(307, headers={'Location': 'https://api.openai.com/v1/chat/completions'})
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, '{}', failure=fail))
    with pytest.raises(VLMError, match='transport policy'):
        client.relations(image_path, ['7: cup', '2: table'])
    assert len(requests) == 3 and all(r.url.host == '127.0.0.1' for r in requests)
    client.close()


def test_original_task_prompts_and_message_layout_are_preserved(tmp_path, image_path):
    from conceptgraph.utils.vlm import system_prompt_only_top, system_prompt_captions
    requests = []
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, '{"relations":[]}'))
    client.relations(image_path, ['7: cup', '2: table'])
    sent = json.loads(requests[-1].content)
    assert sent['messages'][0]['content'].startswith(system_prompt_only_top)
    assert [m['role'] for m in sent['messages']] == ['system', 'user', 'user']
    assert sent['messages'][1]['content'][0]['type'] == 'image_url'
    client.close()
    requests = []
    client = LocalOllamaClient(log_dir=tmp_path, transport=transport_handler(requests, '{"captions":[]}'))
    client.captions(image_path, ['7: cup', '2: table'])
    sent = json.loads(requests[-1].content)
    assert sent['messages'][0]['content'].startswith(system_prompt_captions)
    schema = sent['response_format']['json_schema']['schema']['properties']['captions']
    assert 'minItems' not in schema and 'const' not in schema['items']['properties']['name']
    for path in tmp_path.glob('*.json'):
        assert 'data:image/jpeg;base64,' not in path.read_text()
    client.close()


def test_empty_evidence_consolidation_matches_original_call_and_keeps_empty_result_visible(tmp_path):
    from conceptgraph.utils.vlm import system_prompt_consolidate_captions
    requests = []
    client = LocalOllamaClient(log_dir=tmp_path,
        transport=transport_handler(requests, '{"consolidated_caption":""}'))
    assert client.consolidate([{'caption': None}]) == ''
    assert len(requests) == 3
    sent = json.loads(requests[-1].content)
    assert sent['messages'][0]['content'].startswith(system_prompt_consolidate_captions)
    assert client.quality_summary()['empty_consolidations'] == 1
    assert client.quality_summary()['has_issues']
    client.close()


def test_legacy_lists_and_partial_captions_still_work():
    payload, _, _ = decode_response('relations', 'Relations: [("7", "on top of", "2")]')
    accepted, issues = filter_payload('relations', payload, {'7': 'cup', '2': 'table'})
    assert accepted['relations'][0]['source'] == '7' and not issues['rejected_relations']
    payload, _, failures = decode_response('captions',
        '[{"id":"7", "name":"bowl", "caption":"A bowl."}, {broken}]')
    accepted, _ = filter_payload('captions', payload, {'7': 'cup'})
    assert accepted['captions'][0]['name'] == 'bowl' and len(failures) == 1


def test_duplicate_captions_keep_original_last_wins_behavior():
    from conceptgraph.slam.utils import filter_captions
    rows = [{'id': '7', 'name': 'cup', 'caption': 'First.'},
            {'id': '7', 'name': 'bowl', 'caption': 'Second.'}]
    accepted, _ = filter_payload('captions', {'captions': rows}, {'7': 'cup'})
    assert filter_captions(accepted['captions'], ['cup 7']) == rows[1:]


def test_reference_and_explicit_full_profiles():
    from hydra import compose, initialize_config_module
    with initialize_config_module(config_module='conceptgraph.hydra_configs', version_base=None):
        reference = compose(config_name='rerun_local_scene_graph')
        full = compose(config_name='rerun_local_room0_full')
        reference35b = compose(config_name='rerun_local_scene_graph_reference')
    assert reference.stride == 10 and reference.end == -1
    assert full.stride == 1 and full.end == 2000
    assert reference.rerun_edge_min_detections == full.rerun_edge_min_detections == 2
    assert reference.vlm.response_policy == full.vlm.response_policy == 'upstream'
    assert reference.vlm.validation_retries == full.vlm.validation_retries == 0
    assert reference35b.stride == 10 and reference35b.end == -1
    assert reference35b.vlm.model == full.vlm.model == 'qwen3.5:35b'
    assert reference35b.detections_exp_suffix == full.detections_exp_suffix
    assert not reference.save_objects_all_frames and not reference35b.periodically_save_pcd
