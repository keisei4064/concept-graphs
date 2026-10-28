"""Candidate inspection must show persisted decisions, not recompute a new tree."""

import json

import numpy as np

from conceptgraph.scenegraph.paper_scenegraph_candidates import candidate_data


def test_candidate_display_uses_saved_tree_and_upper_triangle(tmp_path):
    # Deliberately persist a non-minimal tree: visualization must not replace it.
    pairs = [
        {"map_a": 2, "map_b": 7, "pair_id": "2_7"},
        {"map_a": 2, "map_b": 19, "pair_id": "2_19"},
    ]
    graph = {
        "nodes": [{"map_index": i} for i in (2, 7, 19)],
        "edges": [{"source": 7, "target": 2, "relation": "on", "pair_id": "2_7"}],
    }
    (tmp_path / "scene_graph_with_evidence.json").write_text(json.dumps(graph))
    (tmp_path / "candidate_pairs.json").write_text(
        json.dumps({"pairs": pairs, "threshold": 0.01, "candidate_graph_edges": 3})
    )
    matrix = np.array([[0, 0.8, 0.2], [0.1, 0, 0.3], [0.9, 0.7, 0]])
    np.savez_compressed(
        tmp_path / "overlap_matrix.npz", overlap=matrix, map_indices=[2, 7, 19]
    )
    responses = [
        {"pair": pairs[0], "status": "ok", "output": {"object_relation": "b on a"}},
        {
            "pair": pairs[1],
            "status": "ok",
            "output": {"object_relation": "none of these"},
        },
    ]
    (tmp_path / "relations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in responses)
    )
    loaded, edges, requests = candidate_data(tmp_path)
    assert loaded == graph and requests == responses
    assert {(e["a"], e["b"]) for e in edges if e["mst_selected"]} == {(2, 7), (2, 19)}
    assert next(e for e in edges if e["a"] == 2 and e["b"] == 7)["overlap"] == 0.8
    assert next(e for e in edges if e["pair_id"] == "2_7")["final_edge"]["source"] == 7
    assert next(e for e in edges if e["pair_id"] == "2_19")["final_edge"] is None
