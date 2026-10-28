"""Independent checks of persisted graph optimality, provenance and model contracts."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

from conceptgraph.scenegraph.local_paper_scenegraph import (
    file_hash,
    read_rows,
    validated_output,
    validate_graph,
    write_json,
)


def verify(root, run_name):
    root = Path(root).resolve()
    folder = root / run_name
    manifest = json.loads((root / "manifest.json").read_text())
    graph = json.loads((folder / "scene_graph.json").read_text())
    forest = json.loads((folder / "candidate_pairs.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    validate_graph(graph, forest)
    assert file_hash(manifest["input"]["map"]) == manifest["input"]["map_sha256"]
    assert file_hash(root / "manifest.json") == settings["manifest_sha256"]
    for image, sha in manifest["source_image_sha256"].items():
        assert file_hash(image) == sha, image
    # Reconstruct all crops from original JPEG pixels, independently of preparation.
    views = [
        r for r in read_rows(root / "observations.jsonl") if r["status"] == "ready"
    ]
    for row in views:
        path = Path(row["crop_path"])
        assert file_hash(path) == row["crop_sha256"]
        assert file_hash(path.with_suffix(".mask.npz")) == row["crop_mask_sha256"]
        with Image.open(row["source_image"]) as original, Image.open(path) as crop:
            expected = np.asarray(original.convert("RGB"))
            x1, y1, x2, y2 = row["crop_xyxy"]
            np.testing.assert_array_equal(np.asarray(crop), expected[y1:y2, x1:x2])
    mapping = json.loads((folder / "id_mapping.json").read_text())
    by_index = {n["map_index"]: n for n in mapping}
    saved = np.load(folder / "overlap_matrix.npz")
    ids = saved["map_indices"].tolist()
    values = saved["overlap"]
    assert ids == [n["map_index"] for n in graph["nodes"]]
    assert [by_index[i]["graph_id"] for i in ids] == list(range(len(ids)))
    matrix = nx.Graph()
    matrix.add_nodes_from(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if values[i, j] > 0.01:
                matrix.add_edge(ids[i], ids[j], weight=float(values[i, j]))
    candidate = nx.Graph()
    candidate.add_nodes_from(ids)
    for pair in forest["pairs"]:
        assert matrix.has_edge(pair["map_a"], pair["map_b"])
        assert np.isclose(
            pair["weight"], matrix[pair["map_a"]][pair["map_b"]]["weight"]
        )
        candidate.add_edge(pair["map_a"], pair["map_b"], weight=pair["weight"])
    assert len(candidate) == 0 or nx.is_forest(candidate)
    components = list(nx.connected_components(matrix))
    assert {frozenset(c) for c in components} == {
        frozenset(c) for c in forest["components"]
    }
    # Different MST implementation: allow tied solutions but require identical minimum cost.
    oracle = nx.minimum_spanning_tree(matrix, weight="weight", algorithm="kruskal")
    assert np.isclose(candidate.size(weight="weight"), oracle.size(weight="weight"))
    assert candidate.number_of_edges() == len(ids) - len(components)
    assert candidate.number_of_edges() == len(forest["pairs"])
    events = read_rows(folder / "response_events.jsonl")
    checked = 0
    for event in events:
        if event["status"] != "ok":
            continue
        record = json.loads(
            (
                root / "vlm_cache" / f"{event['task']}_{event['response_key']}.json"
            ).read_text()
        )
        output = validated_output(
            event["task"], record["response"]["choices"][0]["message"]["content"]
        )
        assert output == event["output"]
        assert record["contract"]["model"] == settings["model"]
        if event["task"] == "caption":
            assert (
                file_hash(event["context"]["crop_path"])
                == record["contract"]["image_sha256"]
            )
        checked += 1
    summaries = {
        r["map_index"]: r for r in read_rows(folder / "object_summaries.jsonl")
    }
    relations = read_rows(folder / "relations.jsonl")
    assert len(relations) == len(forest["pairs"])
    assert {r["pair"]["pair_id"] for r in relations} == {
        r["pair_id"] for r in forest["pairs"]
    }
    for row in relations:
        assert (
            row["input"]["object1"]["object_tag"]
            == summaries[row["pair"]["map_a"]]["output"]["object_tag"]
        )
        assert (
            row["input"]["object2"]["object_tag"]
            == summaries[row["pair"]["map_b"]]["output"]["object_tag"]
        )
        for key, index in [
            ("object1", row["pair"]["map_a"]),
            ("object2", row["pair"]["map_b"]),
        ]:
            np.testing.assert_array_equal(
                row["input"][key]["bbox_extent"],
                np.round(by_index[index]["bbox_extent"], 1),
            )
            np.testing.assert_array_equal(
                row["input"][key]["bbox_center"],
                np.round(by_index[index]["bbox_center"], 1),
            )
    # Convenient export: retain the raw graph and add explicit links to its evidence.
    enriched = json.loads(json.dumps(graph))
    for node in enriched["nodes"]:
        summary = summaries[node["map_index"]]
        node.update(summary["output"], summary_response_key=summary["response_key"])
        node["observations"] = [r for r in views if r["map_index"] == node["map_index"]]
    for edge in enriched["edges"]:
        row = next(r for r in relations if r["pair"]["pair_id"] == edge["pair_id"])
        edge["response_key"] = row["response_key"]
    write_json(folder / "scene_graph_with_evidence.json", enriched)
    report = {
        "status": "PASS",
        "nodes": len(ids),
        "edges": len(graph["edges"]),
        "candidate_pairs": len(forest["pairs"]),
        "minimum_forest_weight": candidate.size(weight="weight"),
        "independent_mst_oracle": "networkx Kruskal",
        "verified_response_events": checked,
        "source_images_verified": len(manifest["source_image_sha256"]),
        "exact_crops_verified": len(views),
        "source_map_hash_unchanged": True,
    }
    write_json(folder / "validation.json", report)
    execution = {
        "input_manifest_sha256": file_hash(root / "manifest.json"),
        "settings": settings,
        "preparation_sources": manifest["sources"],
        "inference_sources": {
            str(p): file_hash(p)
            for p in [
                Path("conceptgraph/scenegraph/local_paper_scenegraph.py"),
                Path("conceptgraph/utils/ollama_vlm.py"),
            ]
        },
        "verification_sources": {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path("conceptgraph/scenegraph/paper_scenegraph_view.py"),
                Path("scripts/local_only/run_paper_scenegraph.py"),
                Path("scripts/local_only/view_paper_scenegraph.sh"),
            ]
        },
        "model": json.loads((folder / "report.json").read_text())["model"],
        "digest": json.loads((folder / "report.json").read_text())["digest"],
    }
    write_json(folder / "manifest.json", execution)
    return report
