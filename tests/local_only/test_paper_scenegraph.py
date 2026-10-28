import json

import httpx
import jsonschema
import numpy as np
from PIL import Image
import pytest

from conceptgraph.scenegraph.local_paper_scenegraph import (
    PaperClient,
    crop_observation,
    minimum_forest,
    normalized_relation,
    select_observations,
    validated_output,
    validate_graph,
)
from conceptgraph.utils.ollama_vlm import VLMError


def test_observation_selection_preserves_source_order_and_no_backfill():
    assert select_observations([0.1, 0.9, 0.3, 0.8], 3) == [1, 3, 2]
    assert select_observations([0.99], 10) == []


def test_crop_rounding_padding_and_correspondence():
    image = Image.fromarray(
        np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    )
    mask = np.ones((100, 120), bool)
    crop, cm, box = crop_observation(image, mask, [5.6, 15.4, 80.3, 95.5])
    assert box == [0, 5, 90, 100]
    assert crop.size == (90, 95) and cm.shape == (95, 90)
    np.testing.assert_array_equal(np.asarray(crop), np.asarray(image)[5:100, :90])
    with pytest.raises(ValueError):
        crop_observation(image, mask[:90], [0, 0, 10, 10])


def test_mst_uses_upper_triangle_direct_costs_and_original_ids():
    matrix = np.array(
        [[0, 0.8, 0.2, 0], [0.1, 0, 0.3, 0], [0.9, 0.7, 0, 0], [0, 0, 0, 0]]
    )
    forest = minimum_forest(matrix, [2, 7, 19, 41])
    assert {(r["map_a"], r["map_b"]) for r in forest["pairs"]} == {(2, 19), (7, 19)}
    assert forest["components"] == [[2, 7, 19], [41]]
    assert forest["candidate_graph_edges"] == 3
    assert len(minimum_forest(np.empty((0, 0)), [])["pairs"]) == 0


@pytest.mark.parametrize(
    "relation,expected",
    [
        ("a on b", (2, 7, "on")),
        ("b on a", (7, 2, "on")),
        ("a in b", (2, 7, "in")),
        ("b in a", (7, 2, "in")),
        ("none of these", None),
    ],
)
def test_relation_direction(relation, expected):
    result = normalized_relation({"map_a": 2, "map_b": 7, "pair_id": "2_7"}, relation)
    assert (
        None
        if result is None
        else (result["source"], result["target"], result["relation"])
    ) == expected


def test_missing_summary_field_and_unknown_relation_are_rejected():
    with pytest.raises(jsonschema.ValidationError):
        validated_output("summary", '{"description":"chair"}')
    with pytest.raises(jsonschema.ValidationError):
        validated_output("relation", '{"reason":"near","object_relation":"next to"}')


def fake_transport(responses):
    def handle(request):
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"cloud": {"disabled": True}})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3.5:35b", "digest": "test"}]}
            )
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:35b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": response},
                    }
                ],
            },
        )

    return httpx.MockTransport(handle)


def test_paper_cache_reuse_and_failure_do_not_shift_pair_ids(tmp_path):
    responses = [
        '{"reason":"rests on surface","object_relation":"a on b"}',
        "bad json",
        '{"reason":"inside","object_relation":"b in a"}',
    ]
    client = PaperClient(tmp_path, transport=fake_transport(responses))
    try:
        first = client.request(
            "relation", [{"role": "user", "content": "pair 2_7"}], {"pair_id": "2_7"}
        )
        failed = client.request(
            "relation", [{"role": "user", "content": "pair 7_19"}], {"pair_id": "7_19"}
        )
        third = client.request(
            "relation",
            [{"role": "user", "content": "pair 19_41"}],
            {"pair_id": "19_41"},
        )
        cached = client.request(
            "relation", [{"role": "user", "content": "pair 2_7"}], {"pair_id": "2_7"}
        )
        assert (
            first["status"] == "ok"
            and failed["status"] == "failed"
            and third["status"] == "ok"
        )
        assert third["context"]["pair_id"] == "19_41"
        assert cached["cache_hit"] and cached["output"] == first["output"]
        assert len(list(tmp_path.glob("relation_*.json"))) == 2
        assert len(list(tmp_path.glob("failed_relation_*.json"))) == 1
        assert not responses
    finally:
        client.close()


def test_guard_failure_cannot_be_hidden_as_model_failure(tmp_path):
    client = PaperClient(
        tmp_path, transport=fake_transport([PermissionError("blocked destination")])
    )
    try:
        with pytest.raises(VLMError):
            client.request("caption", [{"role": "user", "content": "describe"}], {})
    finally:
        client.close()


def test_noncontiguous_node_ids_survive_saved_graph_reload(tmp_path):
    graph = {
        "nodes": [{"map_index": 2, "uuid": "a"}, {"map_index": 19, "uuid": "b"}],
        "edges": [{"source": 19, "target": 2, "relation": "on", "pair_id": "2_19"}],
    }
    forest = {
        "components": [[2, 19]],
        "pairs": [{"map_a": 2, "map_b": 19, "pair_id": "2_19"}],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    validate_graph(json.loads(path.read_text()), forest)
    graph["edges"][0]["target"] = 0
    with pytest.raises(AssertionError):
        validate_graph(graph, forest)


def test_bbox_extents_follow_public_map_reload_instead_of_saved_edge_order(tmp_path):
    import open3d as o3d
    from conceptgraph.scenegraph.local_paper_scenegraph import load_nodes

    box = o3d.geometry.OrientedBoundingBox(
        np.zeros(3), np.eye(3), np.array([0.2, 0.7, 0.4])
    )
    np.savez_compressed(
        tmp_path / "geometry.npz", bbox_np_0=np.asarray(box.get_box_points())
    )
    (tmp_path / "source_objects.json").write_text(
        json.dumps(
            [{"map_index": 0, "bbox_extent": [0.2, 0.7, 0.4], "bbox_center": [0, 0, 0]}]
        )
    )
    node = load_nodes(tmp_path)[0]
    assert node["saved_bbox_extent"] == [0.2, 0.7, 0.4]
    np.testing.assert_allclose(node["bbox_extent"], [0.7, 0.4, 0.2])


def test_missing_model_text_is_a_validation_failure():
    with pytest.raises(ValueError):
        validated_output("summary", None)


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("timeout"), httpx.ConnectError("offline")]
)
def test_paper_transport_failure_is_recorded_without_success_cache(tmp_path, failure):
    client = PaperClient(tmp_path, transport=fake_transport([failure]))
    try:
        result = client.request(
            "relation", [{"role": "user", "content": "pair 2_7"}], {"pair_id": "2_7"}
        )
        assert result["status"] == "failed" and result["context"]["pair_id"] == "2_7"
        assert not list(tmp_path.glob("relation_*.json"))
        assert len(list(tmp_path.glob("failed_relation_*.json"))) == 1
    finally:
        client.close()
