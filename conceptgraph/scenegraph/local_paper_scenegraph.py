"""Post-process a frozen object map following the public main scenegraph path.

Mapping, detector outputs and realtime VLM tasks are intentionally not modified.
"""

import ast
import base64
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time
import uuid

import httpx
import jsonschema
import numpy as np
from openai import APIError
from PIL import Image, ImageDraw, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from conceptgraph.scenegraph.GPTPrompt import GPTPrompt
from conceptgraph.utils.ollama_vlm import (
    LocalOllamaClient,
    parse_json_response,
    VLMError,
)

RELATIONS = ["a on b", "b on a", "a in b", "b in a", "none of these"]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def reference_prompts(path):
    """Read prompt literals without importing the legacy cloud initialization."""
    tree = ast.parse(Path(path).read_text())
    result = {}
    for name, variable, key in [
        ("extract_node_captions", "query", "caption"),
        ("build_scenegraph", "DEFAULT_PROMPT", "relation"),
    ]:
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assignment = next(
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)
        )
        result[key] = ast.literal_eval(assignment.value)
    result["summary"] = GPTPrompt().get_json()
    return result


def select_observations(confidence, maximum=10):
    if len(confidence) < 2:
        return []
    return np.argsort(np.asarray(confidence))[::-1][:maximum].tolist()


def crop_observation(image, mask, bbox, padding=10):
    """Public main's rounded, padded crop, with explicit correspondence checks."""
    if image.size[::-1] != mask.shape:
        raise ValueError(f"Image/mask mismatch: {image.size[::-1]} vs {mask.shape}")
    x1, y1, x2, y2 = bbox
    coordinates = [
        round(max(0, x1 - padding)),
        round(max(0, y1 - padding)),
        round(min(image.width, x2 + padding)),
        round(min(image.height, y2 + padding)),
    ]
    x1, y1, x2, y2 = coordinates
    pixels = np.asarray(image)[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    if pixels.size == 0 or pixels.shape[:2] != cropped_mask.shape:
        raise ValueError("Empty or inconsistent crop")
    return Image.fromarray(pixels), cropped_mask, coordinates


def prepare(map_path, root, reference, maximum=10):
    """Materialize immutable inputs; masks are released after geometry is saved."""
    map_path, root = Path(map_path).resolve(), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "map": str(map_path),
        "map_sha256": file_hash(map_path),
        "reference": str(Path(reference).resolve()),
        "reference_sha256": file_hash(reference),
        "max_observations": maximum,
        "padding": 10,
        "masking": "none",
        "min_crop_area": 4900,
        "min_observations": 2,
        "point_distance_m": 0.025,
        "overlap_threshold": 0.01,
        "overlap_metric": "public_main_point_overlap",
        "mst_weights": "direct_upper_triangle",
        "caption_encoding": "lossless PNG instead of in-memory PIL; model replaced by Qwen",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["input"] != identity:
            raise ValueError(
                "Existing output belongs to different inputs/settings; use a new output directory"
            )
        if manifest.get("prepared"):
            for row in read_rows(root / "observations.jsonl"):
                if (
                    row["status"] == "ready"
                    and file_hash(row["crop_path"]) != row["crop_sha256"]
                ):
                    raise ValueError("Saved crop changed")
            return manifest
    write_json(manifest_path, {"input": identity, "prepared": False})
    with gzip.open(map_path, "rb") as stream:
        artifact = pickle.load(stream)  # Trusted local map, never external pickle.
    objects = artifact["objects"]
    nodes, observations, arrays, sources = [], [], {}, {}
    crop_dir = root / "crops"
    crop_dir.mkdir(exist_ok=True)
    for i, obj in enumerate(objects):
        lengths = [
            len(obj[k]) for k in ("color_path", "mask", "xyxy", "conf", "image_idx")
        ]
        if len(set(lengths)) != 1 or lengths[0] != obj["num_detections"]:
            raise ValueError(
                f"Observation histories disagree for map index {i}: {lengths}"
            )
        bbox = np.asarray(obj["bbox_np"])
        node = {
            "map_index": i,
            "map_id": int(obj["curr_obj_num"]),
            "uuid": str(obj["id"]),
            "class_name": obj["class_name"],
            "num_observations": lengths[0],
            "bbox_center": bbox.mean(axis=0).tolist(),
            "bbox_extent": np.linalg.norm(bbox[[1, 2, 3]] - bbox[0], axis=1).tolist(),
            "inst_color": np.asarray(obj["inst_color"]).tolist(),
        }
        nodes.append(node)
        for name in ("pcd_np", "pcd_color_np", "bbox_np", "clip_ft"):
            value = obj[name]
            arrays[f"{name}_{i}"] = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )
        selected = select_observations(obj["conf"], maximum)
        if not selected:
            observations.append(
                {
                    "map_index": i,
                    "status": "excluded",
                    "reason": "fewer_than_two_observations",
                }
            )
        for rank, j in enumerate(selected):
            path = Path(obj["color_path"][j]).resolve()
            row = {
                "map_index": i,
                "map_id": node["map_id"],
                "uuid": node["uuid"],
                "observation_index": j,
                "confidence_rank": rank,
                "source_image": str(path),
                "source_frame": path.name,
                "processing_frame": int(obj["image_idx"][j]),
                "confidence": float(obj["conf"][j]),
                "bbox_xyxy": np.asarray(obj["xyxy"][j]).tolist(),
            }
            # Correspondence failures stop preparation; no silent replacement view.
            with Image.open(path) as image:
                image = image.convert("RGB")
                crop, mask, coords = crop_observation(
                    image, np.asarray(obj["mask"][j]), row["bbox_xyxy"]
                )
            if str(path) not in sources:
                sources[str(path)] = file_hash(path)
            row.update(
                source_sha256=sources[str(path)],
                crop_xyxy=coords,
                crop_size=list(crop.size),
            )
            if crop.width * crop.height < 4900:
                row.update(status="excluded", reason="crop_area_below_4900")
            else:
                target = crop_dir / f"object_{i:03d}_view_{j:03d}.png"
                crop.save(target)
                np.savez_compressed(target.with_suffix(".mask.npz"), mask=mask)
                row.update(
                    status="ready",
                    crop_path=str(target),
                    crop_sha256=file_hash(target),
                    crop_mask_sha256=file_hash(target.with_suffix(".mask.npz")),
                )
            observations.append(row)
    old_edges = artifact.get("edges") or {"edges": []}
    write_json(root / "source_edges.json", old_edges["edges"])
    np.savez_compressed(root / "geometry.npz", **arrays)
    write_json(root / "source_objects.json", nodes)
    write_rows(root / "observations.jsonl", observations)
    write_json(root / "observations.json", observations)
    # The original masks can be several GB: do not deep-copy or carry them into inference.
    del obj, objects, artifact, arrays
    gc.collect()
    counts = Counter(row.get("reason", row["status"]) for row in observations)
    manifest = {
        "input": identity,
        "prepared": True,
        "objects": len(nodes),
        "observations": dict(counts),
        "source_image_sha256": sources,
        "prompts": reference_prompts(reference),
        "sources": {
            str(p): file_hash(p)
            for p in [
                Path(__file__),
                Path(reference),
                Path("conceptgraph/scenegraph/GPTPrompt.py"),
                Path("conceptgraph/slam/utils.py"),
            ]
        },
    }
    write_json(manifest_path, manifest)
    contact_sheet(observations, root / "contact_sheet.jpg")
    return manifest


def contact_sheet(observations, output):
    selected = {}
    for row in observations:
        if row["status"] == "ready":
            selected.setdefault(row["map_index"], row)
    rows = list(selected.values())
    sheet = Image.new("RGB", (6 * 220, max(1, (len(rows) + 5) // 6) * 180), "white")
    draw = ImageDraw.Draw(sheet)
    for k, row in enumerate(rows):
        with Image.open(row["crop_path"]) as im:
            tile = ImageOps.contain(im, (210, 150))
        x, y = k % 6 * 220, k // 6 * 180
        sheet.paste(tile, (x, y))
        draw.text(
            (x, y + 152), f"map {row['map_id']} / idx {row['map_index']}", fill="black"
        )
    sheet.save(output)


def minimum_forest(overlap, map_indices, threshold=0.01):
    """Release semantics: use upper triangle directly, without symmetrizing by max."""
    values = np.asarray(overlap)
    if (
        values.shape != (len(map_indices), len(map_indices))
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid overlap matrix")
    upper = np.triu(np.where(values > threshold, values, 0), 1)
    adjacency = csr_matrix(upper + upper.T)
    count, labels = connected_components(adjacency, directed=False)
    components, pairs = [], []
    for label in range(count):
        component = np.flatnonzero(labels == label)
        components.append([map_indices[int(i)] for i in component])
        tree = minimum_spanning_tree(adjacency[component][:, component])
        for u, v in zip(*tree.nonzero()):
            i, j = int(component[u]), int(component[v])
            pairs.append(
                {
                    "pair_id": f"{map_indices[i]}_{map_indices[j]}",
                    "graph_a": i,
                    "graph_b": j,
                    "map_a": map_indices[i],
                    "map_b": map_indices[j],
                    "weight": float(tree[u, v]),
                }
            )
    assert len(pairs) == len(map_indices) - count
    return {
        "metric": "public_main_point_overlap",
        "threshold": threshold,
        "components": components,
        "candidate_graph_edges": int(adjacency.nnz // 2),
        "pairs": pairs,
    }


def compute_release_overlap(root):
    """Local helper matches main's old cfg-based helper after unpacking its argument."""
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

    geometry = np.load(Path(root) / "geometry.npz")
    nodes = json.loads((Path(root) / "source_objects.json").read_text())
    objects = MapObjectList()
    objects.load_serializable(
        [
            {
                key: geometry[f"{key}_{i}"]
                for key in ("pcd_np", "pcd_color_np", "bbox_np", "clip_ft")
            }
            for i in range(len(nodes))
        ]
    )
    overlap = compute_overlap_matrix(objects, 0.025)
    np.savez_compressed(Path(root) / "release_overlap_all.npz", overlap=overlap)
    return overlap


def output_schema(task):
    if task == "summary":
        props = {
            "summary": {"type": "string"},
            "possible_tags": {"type": "array", "items": {"type": "string"}},
            "object_tag": {"type": "string", "minLength": 1},
        }
    elif task == "relation":
        props = {
            "reason": {"type": "string"},
            "object_relation": {"type": "string", "enum": RELATIONS},
        }
    else:
        return None
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def validated_output(task, text):
    if not isinstance(text, str):
        raise ValueError("Missing model response text")
    if task == "caption":
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty caption")
        return text.strip().replace("The central object in the image is ", "")
    payload, _ = parse_json_response(text)
    jsonschema.validate(payload, output_schema(task))
    return payload


def prohibited_transport(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if (
            isinstance(exc, PermissionError)
            or 300 <= getattr(exc, "status_code", 0) < 400
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class PaperClient:
    """Reuse verified local transport, with independent paper task contracts/cache."""

    def __init__(self, root, model="qwen3.5:35b", transport=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = LocalOllamaClient(
            base_url="http://127.0.0.1:11435/v1",
            model=model,
            log_dir=self.root,
            timeout=300,
            transport=transport,
        )
        self.digest, self.model = self.client.digest, model
        self.events = []

    def close(self):
        self.client.close()

    def request(self, task, messages, context, image=None):
        budget = {"caption": 512, "summary": 768, "relation": 768}[task]
        image_sha = file_hash(image) if image else None
        contract = {
            "task": task,
            "model": self.model,
            "digest": self.digest,
            "messages": messages,
            "image_sha256": image_sha,
            "temperature": 0.0,
            "max_tokens": budget,
            "schema": output_schema(task),
            "think": False,
        }
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        path = self.root / f"{task}_{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["contract"] != contract:
                raise ValueError("Cache contract mismatch")
            accepted = validated_output(
                task, record["response"]["choices"][0]["message"]["content"]
            )
            event = {
                "task": task,
                "status": "ok",
                "cache_hit": True,
                "context": context,
                "response_key": key,
                "elapsed_seconds": 0.0,
                "output": accepted,
            }
            self.events.append(event)
            return event
        wire = json.loads(json.dumps(messages))
        if image:
            data = Path(image).read_bytes()
            mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
            wire[-1]["content"] = [
                {"type": "text", "text": wire[-1]["content"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64," + base64.b64encode(data).decode()
                    },
                },
            ]
        record = {
            "contract": contract,
            "context": context,
            "image_path": str(image) if image else None,
            "response_key": key,
        }
        started = time.perf_counter()
        try:
            options = {}
            if contract["schema"]:
                options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "paper_" + task,
                        "strict": True,
                        "schema": contract["schema"],
                    },
                }
            response = self.client.sdk.chat.completions.create(
                model=self.model,
                messages=wire,
                temperature=0.0,
                max_tokens=budget,
                extra_body={"think": False, "reasoning_effort": "none"},
                **options,
            )
            record["response"] = response.model_dump(mode="json")
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("Missing or incomplete response")
            output = validated_output(task, response.choices[0].message.content)
            record.update(
                status="ok",
                elapsed_seconds=time.perf_counter() - started,
                output=output,
            )
            write_json(path, record)
        except (
            APIError,
            httpx.HTTPError,
            ValueError,
            jsonschema.ValidationError,
            VLMError,
        ) as exc:
            record.update(
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            write_json(self.root / f"failed_{task}_{uuid.uuid4().hex}.json", record)
            if prohibited_transport(exc):
                raise VLMError("Local transport policy violated") from exc
        event = {
            key: record[key] for key in ("status", "elapsed_seconds", "response_key")
        }
        event.update(task=task, cache_hit=False, context=context)
        event.update({k: record[k] for k in ("output", "error") if k in record})
        self.events.append(event)
        return event


def load_nodes(root):
    """Match MapObjectList.load_serializable's OBB reconstruction before using extents.

    Saved corner edge order can differ from the reconstructed OBB's PCA axis order.
    The public main relation prompt uses the reconstructed object's extent/center.
    """
    import open3d as o3d

    root = Path(root)
    nodes = json.loads((root / "source_objects.json").read_text())
    with np.load(root / "geometry.npz") as geometry:
        for node in nodes:
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(geometry[f"bbox_np_{node['map_index']}"])
            )
            node["saved_bbox_extent"] = node["bbox_extent"]
            node["bbox_extent"] = bbox.extent.tolist()
            node["bbox_center"] = bbox.center.tolist()
    return nodes


def relation_input(pair, nodes, summaries):
    result = {}
    for label, key in [("object1", "map_a"), ("object2", "map_b")]:
        index = pair[key]
        node = nodes[index]
        result[label] = {
            "id": pair["graph_a" if key == "map_a" else "graph_b"],
            "bbox_extent": np.round(node["bbox_extent"], 1).tolist(),
            "bbox_center": np.round(node["bbox_center"], 1).tolist(),
            "object_tag": summaries[index]["output"]["object_tag"],
        }
    return result


def normalized_relation(pair, relation):
    if relation == "none of these":
        return None
    a, b = pair["map_a"], pair["map_b"]
    return {
        "source": a if relation.startswith("a ") else b,
        "target": b if relation.startswith("a ") else a,
        "relation": "on" if " on " in relation else "in",
        "pair_id": pair["pair_id"],
    }


def run_inference(root, run_name, model="qwen3.5:35b", indices=None, maximum=10):
    root = Path(root).resolve()
    output = root / run_name
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    nodes = load_nodes(root)
    observations = read_rows(root / "observations.jsonl")
    indices = list(range(len(nodes))) if indices is None else list(indices)
    settings = {
        "model": model,
        "indices": indices,
        "maximum": maximum,
        "manifest_sha256": file_hash(root / "manifest.json"),
    }
    if (output / "settings.json").exists() and json.loads(
        (output / "settings.json").read_text()
    ) != settings:
        raise ValueError("Run settings changed; use a new run name")
    write_json(output / "settings.json", settings)
    prompts = manifest["prompts"]
    client = PaperClient(root / "vlm_cache", model)
    captions, summaries, relations = [], {}, []
    started = time.perf_counter()

    def progress(phase, **extra):
        write_json(
            output / "state.json",
            {
                "status": "running",
                "phase": phase,
                "elapsed_seconds": time.perf_counter() - started,
                "requests": len(client.events),
                **extra,
            },
        )

    try:
        for index in indices:
            views = [
                r
                for r in observations
                if r["map_index"] == index
                and r.get("confidence_rank", maximum) < maximum
                and r["status"] == "ready"
            ]
            if nodes[index]["num_observations"] < 2:
                continue
            for row in views:
                progress("captions", map_index=index, source=row["source_frame"])
                event = client.request(
                    "caption",
                    [{"role": "user", "content": prompts["caption"]}],
                    row,
                    row["crop_path"],
                )
                captions.append(dict(event, observation=row))
                write_rows(output / "captions.jsonl", captions)
            evidence = [
                r["output"]
                for r in captions
                if r["context"]["map_index"] == index and r["status"] == "ok"
            ]
            messages = prompts["summary"] + [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"id": index, "captions": evidence}, indent=0
                    ),
                }
            ]
            progress("summaries", map_index=index)
            summaries[index] = client.request("summary", messages, {"map_index": index})
            write_rows(
                output / "object_summaries.jsonl",
                [{"map_index": i, **r} for i, r in summaries.items()],
            )
        retained = [
            i
            for i in indices
            if i in summaries
            and summaries[i]["status"] == "ok"
            and summaries[i]["output"]["object_tag"].lower() not in ("invalid", "fail")
            and nodes[i]["num_observations"] >= 2
        ]
        mapping = [
            {
                "graph_id": retained.index(i) if i in retained else None,
                **node,
                "exclusion_reason": None
                if i in retained
                else (
                    "insufficient_views"
                    if node["num_observations"] < 2
                    else "summary_failed"
                    if i not in summaries or summaries[i]["status"] != "ok"
                    else "invalid_tag"
                ),
            }
            for i, node in enumerate(nodes)
            if i in indices
        ]
        write_json(output / "id_mapping.json", mapping)
        progress("overlap")
        overlap_path = root / "release_overlap_all.npz"
        all_overlap = (
            np.load(overlap_path)["overlap"]
            if overlap_path.exists()
            else compute_release_overlap(root)
        )
        overlap = all_overlap[np.ix_(retained, retained)]
        np.savez_compressed(
            output / "overlap_matrix.npz", overlap=overlap, map_indices=retained
        )
        forest = minimum_forest(overlap, retained)
        write_json(output / "candidate_pairs.json", forest)
        for pair in forest["pairs"]:
            data = relation_input(pair, nodes, summaries)
            progress("relations", pair_id=pair["pair_id"])
            event = client.request(
                "relation",
                [
                    {
                        "role": "user",
                        "content": prompts["relation"] + "\n\n" + json.dumps(data),
                    }
                ],
                pair,
            )
            relations.append(dict(event, pair=pair, input=data))
            write_rows(output / "relations.jsonl", relations)
        edges = []
        for row in relations:
            if row["status"] == "ok":
                edge = normalized_relation(
                    row["pair"], row["output"]["object_relation"]
                )
                if edge:
                    edges.append(dict(edge, reason=row["output"]["reason"]))
        graph = {
            "method": "public_main_point_overlap_mst_qwen",
            "nodes": [r for r in mapping if r["graph_id"] is not None],
            "edges": edges,
            "static_postprocessing": True,
        }
        write_json(output / "scene_graph.json", graph)
        validate_graph(graph, forest)
        comparison = compare_realtime(root, graph, forest)
        write_json(output / "comparison.json", comparison)
        failures = [e for e in client.events if e["status"] != "ok"]
        report = {
            "status": "completed_with_vlm_issues" if failures else "completed",
            "input_objects": len(indices),
            "retained_objects": len(retained),
            "edges": len(edges),
            "components": len(forest["components"]),
            "candidate_pairs": len(forest["pairs"]),
            "caption_calls": len(captions),
            "summary_calls": len(summaries),
            "relation_calls": len(relations),
            "new_requests": sum(not e["cache_hit"] for e in client.events),
            "cache_hits": sum(e["cache_hit"] for e in client.events),
            "failures": failures,
            "task_latency_seconds": {
                task: [
                    e["elapsed_seconds"]
                    for e in client.events
                    if e["task"] == task and not e["cache_hit"]
                ]
                for task in ("caption", "summary", "relation")
            },
            "elapsed_seconds": time.perf_counter() - started,
            "model": model,
            "digest": client.digest,
            "comparison": comparison,
            "source_map_hash_unchanged": file_hash(manifest["input"]["map"])
            == manifest["input"]["map_sha256"],
        }
        assert report["source_map_hash_unchanged"]
        write_json(output / "report.json", report)
        write_rows(output / "response_events.jsonl", client.events)
        write_json(output / "state.json", report)
        return report
    except Exception as exc:
        write_json(
            output / "state.json",
            {
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    finally:
        client.close()


def validate_graph(graph, forest):
    ids = {r["map_index"] for r in graph["nodes"]}
    assert len(ids) == len(graph["nodes"])
    assert len({r["uuid"] for r in graph["nodes"]}) == len(ids)
    candidates = {r["pair_id"] for r in forest["pairs"]}
    for p in forest["pairs"]:
        assert p["map_a"] in ids and p["map_b"] in ids and p["map_a"] != p["map_b"]
    assert len(candidates) == len(ids) - len(forest["components"])
    for e in graph["edges"]:
        assert e["source"] in ids and e["target"] in ids and e["source"] != e["target"]
        assert e["relation"] in ("on", "in") and e["pair_id"] in candidates


def compare_realtime(root, graph, forest):
    old = json.loads((Path(root) / "source_edges.json").read_text())
    old_on = set()
    for e in old:
        a, b = e["obj1_index"], e["obj2_index"]
        if e["rel_type"] == "under":
            a, b = b, a
        if e["rel_type"] in ("under", "on top of"):
            old_on.add((a, b))
    new_on = {
        (e["source"], e["target"]) for e in graph["edges"] if e["relation"] == "on"
    }
    candidate_sets = {frozenset((p["map_a"], p["map_b"])) for p in forest["pairs"]}
    retained = {n["map_index"] for n in graph["nodes"]}
    return {
        "realtime_edges": len(old),
        "realtime_unique_on": len(old_on),
        "new_on": len(new_on),
        "new_in": sum(e["relation"] == "in" for e in graph["edges"]),
        "shared_directed_on": sorted(old_on & new_on),
        "realtime_only_on": sorted(old_on - new_on),
        "new_only_on": sorted(new_on - old_on),
        "realtime_on_pairs_in_mst": sum(frozenset(e) in candidate_sets for e in old_on),
        "realtime_on_with_both_nodes_retained": sum(set(e) <= retained for e in old_on),
        "interpretation": "Same-map method comparison; in-relations absent from realtime vocabulary; no ground-truth accuracy claim.",
    }
