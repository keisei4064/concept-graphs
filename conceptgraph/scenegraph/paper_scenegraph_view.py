"""Static graph plus observation/pair evidence browsing in a separate Rerun file."""

import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from conceptgraph.scenegraph.local_paper_scenegraph import read_rows, write_json


def graph_diagram(graph, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    network = nx.DiGraph()
    by_index = {n["map_index"]: n for n in graph["nodes"]}
    network.add_nodes_from(by_index)
    for edge in graph["edges"]:
        network.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    total = len(network)
    linked = {i for i in network if network.degree(i)}
    if linked:
        network = network.subgraph(linked).copy()
    figure, axis = plt.subplots(figsize=(16, 12))
    positions = {}
    components = sorted(
        nx.weakly_connected_components(network), key=lambda c: (-len(c), min(c))
    )
    for k, component in enumerate(components):
        undirected = network.subgraph(component).to_undirected()
        anchor = min(component, key=lambda i: (-undirected.degree(i), i))
        depths = nx.single_source_shortest_path_length(undirected, anchor)
        for depth in sorted(set(depths.values())):
            level = sorted(i for i in component if depths[i] == depth)
            for j, i in enumerate(level):
                positions[i] = np.array(
                    [
                        (k % 3) * 8.5 + (j - (len(level) - 1) / 2) * 2.8,
                        -(k // 3) * 7.2 - depth * 3.4,
                    ]
                )
    nx.draw_networkx_nodes(
        network,
        positions,
        ax=axis,
        node_color="#dceafb",
        node_size=4400,
        edgecolors="#537494",
    )
    labels = {
        i: f"#{by_index[i]['map_id']}\n"
        + "\n".join(textwrap.wrap(by_index[i]["object_tag"], 15))
        for i in network
    }
    nx.draw_networkx_labels(network, positions, labels, ax=axis, font_size=11)
    nx.draw_networkx_edges(
        network,
        positions,
        ax=axis,
        arrows=True,
        arrowsize=22,
        node_size=4400,
        edge_color="#535e6b",
        connectionstyle="arc3,rad=0",
    )
    nx.draw_networkx_edge_labels(
        network,
        positions,
        edge_labels={(a, b): d["relation"] for a, b, d in network.edges(data=True)},
        ax=axis,
        font_size=14,
        font_weight="bold",
        rotate=False,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#c4ccd5",
            "alpha": 1.0,
        },
    )
    axis.set_title(
        f"Public main + Qwen: {total} objects, {len(graph['edges'])} relations\n"
        f"{len(network)} linked objects shown; {total - len(network)} isolated objects remain in 3D/evidence.\n"
        "Arrows: object on/in target. Labels: original map display IDs."
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)
    write_json(
        path.with_suffix(".json"),
        {
            "shown_map_indices": sorted(network),
            "isolated_map_indices": sorted(set(by_index) - set(network)),
            "total_graph_nodes": total,
        },
    )


def write_evidence_markdown(root, run_name):
    """Explain saved caption provenance without rerunning inference or the viewer."""
    import os

    root = Path(root).resolve()
    folder = root / run_name
    nodes = json.loads((root / "source_objects.json").read_text())
    graph = json.loads((folder / "scene_graph.json").read_text())
    retained = {n["map_index"] for n in graph["nodes"]}
    captions = read_rows(folder / "captions.jsonl")
    summaries = {
        r["map_index"]: r for r in read_rows(folder / "object_summaries.jsonl")
    }
    relations = read_rows(folder / "relations.jsonl")
    by_object = {}
    for row in captions:
        by_object.setdefault(row["observation"]["map_index"], []).append(row)
    lines = [
        "# Node Captioningの結果と根拠",
        "",
        "この文書は保存済みのQwen応答と画像から自動生成しています。説明文の意味的な正しさを保証するものではありません。",
        "",
        "## 読み方と生成順序",
        "",
        "1. 物体ごとにconfidence上位の観測を最大10件選び、各RGBから物体画像を切り出します。",
        "2. **画像ごとのCaption生成（LVLM・画像入力）**：各切り出し画像をQwenへ1枚ずつ渡し、自由文の説明を生成します。",
        "3. **物体単位の最終要約（LLM・テキスト入力）**：同じ物体の有効なCaption文の一覧をQwenへ渡し、JSONへ統合します。",
        "",
        "**最終要約を先に表示し、その下へ要約の根拠を並べています。生成順序は逆で、各画像のCaptionが先です。**",
        "",
        "要約フェーズに画像やBBoxは再入力していません。画像は閲覧者がCaptionの根拠を確認するために掲載しています。",
        "",
        "| 表示 | 意味 |",
        "|---|---|",
        "| 最終要約JSON | 物体単位のテキスト統合結果。`summary`は説明、`possible_tags`は名前の候補、`object_tag`は最終的な名前 |",
        "| 観測ごとの画像 | 最初のCaption生成で実際にQwenへ送った物体切り出し画像 |",
        "| 引用枠のCaption | その画像からQwenが生成した自由文。`In summary:`、太字、箇条書き等もモデルの出力 |",
        "| Map番号 / map index | Map番号は元の物体表示ID、indexは保存マップ内の配列位置 |",
        "| 観測番号 / 処理フレーム | 観測番号はその物体の観測履歴内の位置、処理フレームはstride後の入力番号。元画像番号とは別 |",
        "",
        f"画像別Captionは{len(captions)}件、物体単位の要約は{len(summaries)}件、最終グラフの採用ノードは{len(retained)}件です。",
        "対象外となった要約や、有効な切り出し画像が0枚の物体も区別して掲載します。",
        "",
        "## 最終シーングラフ",
        "",
        "![最終シーングラフ](diagram.png)",
        "",
        "## 物体ごとの最終要約と根拠",
        "",
    ]
    for index, summary in summaries.items():
        node = nodes[index]
        rows = by_object.get(index, [])
        valid = [r for r in rows if r["status"] == "ok"]
        status = (
            "採用"
            if index in retained
            else "対象外（要約のinvalid/fail判定、または要約失敗）"
        )
        lines += [
            f"### Map #{node['map_id']}（map index {index}）",
            "",
            f"検出器のクラス：`{node['class_name']}`。最終グラフ：**{status}**。",
            "",
            "#### 最終要約JSON — この物体のCaption文を統合した結果",
            "",
            f"入力は、この下に掲載する**有効な{len(valid)}件のCaption文の一覧**です。画像自体は再入力していません。",
            "",
            "```json",
            json.dumps(
                summary.get("output", {"error": summary.get("error", "missing")}),
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "#### 要約の根拠 — 各観測の画像と画像別Caption",
            "",
        ]
        if not rows:
            lines += [
                "サイズ条件等を通過した切り出し画像がなく、要約へ渡したCaption一覧は空です。",
                "",
            ]
        for row in rows:
            observation = row["observation"]
            relative_crop = "../crops/" + Path(observation["crop_path"]).name
            relative_rgb = Path(
                os.path.relpath(observation["source_image"], folder)
            ).as_posix()
            used = (
                "要約の入力に使用"
                if row["status"] == "ok"
                else "Caption生成失敗のため要約の入力から除外"
            )
            lines += [
                f"##### 観測 {observation['observation_index']} — {observation['source_frame']}",
                "",
                f"処理フレーム：{observation['processing_frame']} ／ confidence：{observation['confidence']:.4f} ／ "
                f"選択順位：{observation['confidence_rank'] + 1} ／ **{used}**。",
                "",
                f"[元のRGB画像を開く]({relative_rgb})",
                "",
                "**Caption生成時の画像入力（物体切り出し）**",
                "",
                f"![物体切り出し・観測{observation['observation_index']}]({relative_crop})",
                "",
                "**この画像に対するQwenのCaption（自由文）**",
                "",
            ]
            text = row.get("output", row.get("error", "failed"))
            lines += ["> " + line if line else ">" for line in text.splitlines()]
            lines += ["", "---", ""]
    lines += [
        "## 候補ペアの関係推定",
        "",
        "ここからはNode Captioningの後段です。物体の`object_tag`と3D BBoxの中心・寸法をテキストで渡して推定します。",
        "**横並び画像は閲覧用の合成で、関係推定の画像入力ではありません。** 2画像は別のフレームから選ばれる場合があります。",
        "`a`は入力の`object1`、`b`は`object2`です。`none of these`で不採用となった候補も掲載します。",
        "",
    ]
    for row in relations:
        a, b = row["pair"]["map_a"], row["pair"]["map_b"]
        lines += [
            f"### Pair {row['pair']['pair_id']} — Map #{nodes[a]['map_id']} と Map #{nodes[b]['map_id']}",
            "",
            "#### 閲覧用の参考画像（関係推定には送信していない）",
            "",
            f"![候補ペア](pair_images/{row['pair']['pair_id']}.jpg)",
            "",
            "#### 関係推定へ渡した入力（テキスト）",
            "",
            "```json",
            json.dumps(row["input"], ensure_ascii=False, indent=2),
            "```",
            "",
            "#### Qwenの返却 — 関係ラベルと理由",
            "",
            "```json",
            json.dumps(
                row.get("output", {"error": row.get("error")}),
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    (folder / "evidence.md").write_text("\n".join(lines))


def load_view_geometry(root):
    """Reconstruct saved OBBs once for both final and candidate viewers."""
    import open3d as o3d
    import rerun as rr
    from scipy.spatial.transform import Rotation

    root = Path(root)
    source_nodes = json.loads((root / "source_objects.json").read_text())
    with np.load(root / "geometry.npz") as saved:
        geometry = {key: saved[key] for key in saved.files}
    boxes = [
        o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(geometry[f"bbox_np_{i}"])
        )
        for i in range(len(source_nodes))
    ]
    quaternions = [
        rr.Quaternion(xyzw=Rotation.from_matrix(np.array(box.R, copy=True)).as_quat())
        for box in boxes
    ]
    return source_nodes, geometry, boxes, quaternions


def log_static_map(nodes, source_nodes, geometry, boxes, quaternions):
    """Keep point density, OBBs and object metadata consistent between viewers."""
    import rerun as rr

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    retained = {n["map_index"] for n in nodes}
    for i, node in enumerate(source_nodes):
        points = geometry[f"pcd_np_{i}"]
        colors = geometry[f"pcd_color_np_{i}"]
        color = (
            (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            if i in retained
            else [170, 180, 190, 45]
        )
        # Match the original map viewer: preserve saved point density and let
        # Rerun choose point radii, rather than imposing tiny world-space dots.
        rr.log(f"world/objects/{i}", rr.Points3D(points, colors=color), timeless=True)
    for node in nodes:
        i = node["map_index"]
        color = (np.clip(node["inst_color"], 0, 1) * 255).astype(np.uint8)
        rr.log(
            f"world/bboxes/{i}",
            rr.Boxes3D(
                centers=[boxes[i].center],
                half_sizes=[boxes[i].extent / 2],
                rotations=[quaternions[i]],
                colors=[color],
                labels=[f"#{node['map_id']} {node['object_tag']}"],
            ),
            rr.AnyValues(
                uuid=node["uuid"], map_index=i, detector_class=node["class_name"]
            ),
            timeless=True,
        )


def render(root, run_name):
    import rerun as rr
    import rerun.blueprint as blueprint

    root = Path(root).resolve()
    folder = root / run_name
    graph = json.loads((folder / "scene_graph.json").read_text())
    summaries = {
        r["map_index"]: r for r in read_rows(folder / "object_summaries.jsonl")
    }
    captions = read_rows(folder / "captions.jsonl")
    relations = read_rows(folder / "relations.jsonl")
    source_nodes, geometry, boxes, quaternions = load_view_geometry(root)
    for node in graph["nodes"]:
        node["object_tag"] = summaries[node["map_index"]]["output"]["object_tag"]
    graph_diagram(graph, folder / "diagram.png")
    rr.init("conceptgraphs_public_main_qwen", spawn=False)
    rr.save(str(folder / "viewer.rrd"))
    log_static_map(graph["nodes"], source_nodes, geometry, boxes, quaternions)
    edge_anchors = []
    for k, edge in enumerate(graph["edges"]):
        a = boxes[edge["source"]].center
        b = boxes[edge["target"]].center
        color = [38, 135, 200] if edge["relation"] == "on" else [165, 75, 175]
        midpoint = (a + b) / 2
        rr.log(
            f"world/relations/{k}/arrow",
            rr.Arrows3D(origins=[a], vectors=[b - a], radii=0.008, colors=color),
            timeless=True,
        )
        # Arrow labels are anchored by the viewer's arrow archetype. Explicitly
        # anchor the relation text at the same 3D midpoint as a line label.
        rr.log(
            f"world/relations/{k}/label",
            rr.Points3D(
                [midpoint], radii=0.004, colors=color, labels=[edge["relation"]]
            ),
            rr.AnyValues(
                source_map_id=source_nodes[edge["source"]]["map_id"],
                target_map_id=source_nodes[edge["target"]]["map_id"],
            ),
            timeless=True,
        )
        edge_anchors.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "start": a.tolist(),
                "end": b.tolist(),
                "label_anchor": midpoint.tolist(),
                "relation": edge["relation"],
            }
        )
    write_json(
        folder / "three_d_render_metadata.json",
        {
            "saved_points": sum(
                len(geometry[f"pcd_np_{i}"]) for i in range(len(source_nodes))
            ),
            "point_downsampling": False,
            "bbox_count": len(graph["nodes"]),
            "edge_label_anchors": edge_anchors,
            "bbox_source": "Saved OBB corners reconstructed identically to MapObjectList.load_serializable",
        },
    )
    rr.log(
        "diagram/graph",
        rr.ImageEncoded(path=str(folder / "diagram.png")),
        timeless=True,
    )
    evidence = []
    by_object = {}
    for row in captions:
        index = row["observation"]["map_index"]
        by_object.setdefault(index, []).append(row)
        node = source_nodes[index]
        summary = summaries.get(index, {})
        details = (
            f"# Map #{node['map_id']} (map index {index})\n"
            f"UUID: `{node['uuid']}`\n\nOriginal detector class: {node['class_name']}\n\n"
            f"Source: {row['observation']['source_frame']} / sampled frame {row['observation']['processing_frame']}\n\n"
            f"Confidence: {row['observation']['confidence']:.4f}\n\n"
            f"## Crop caption\n{row.get('output', row.get('error', 'failed'))}\n\n"
            f"## Multi-view summary\n{json.dumps(summary.get('output', summary.get('error', 'missing')), ensure_ascii=False, indent=2)}"
        )
        evidence.append(
            {
                "kind": "observation",
                "map_indices": [index],
                "image": row["observation"]["crop_path"],
                "rgb": row["observation"]["source_image"],
                "text": details,
                "response_key": row["response_key"],
                "observation_index": row["observation"]["observation_index"],
            }
        )
    pair_dir = folder / "pair_images"
    pair_dir.mkdir(exist_ok=True)
    for row in relations:
        a, b = row["pair"]["map_a"], row["pair"]["map_b"]
        sheet = Image.new("RGB", (720, 330), "white")
        draw = ImageDraw.Draw(sheet)
        for k, index in enumerate((a, b)):
            examples = by_object.get(index, [])
            if examples:
                with Image.open(examples[0]["observation"]["crop_path"]) as im:
                    tile = ImageOps.contain(im, (350, 290))
                    sheet.paste(tile, (k * 360, 0))
            draw.text(
                (k * 360, 300),
                f"Map #{source_nodes[index]['map_id']} (index {index})",
                fill="black",
            )
        image = pair_dir / f"{row['pair']['pair_id']}.jpg"
        sheet.save(image)
        details = (
            f"# Candidate {row['pair']['pair_id']}\nStatic post-processing; pair crops may come from different frames.\n\n"
            f"Overlap: {row['pair']['weight']:.5f}\n\n"
            f"## Input (text only for relationship inference)\n```json\n{json.dumps(row['input'], indent=2)}\n```\n\n"
            f"## Response\n```json\n{json.dumps(row.get('output', row.get('error')), indent=2)}\n```"
        )
        evidence.append(
            {
                "kind": "relation",
                "map_indices": [a, b],
                "image": str(image),
                "rgb": None,
                "text": details,
                "response_key": row["response_key"],
            }
        )
    if not evidence:
        raise ValueError("No evidence to display")
    # Open with an actual source frame visible, rather than the last text-only pair.
    overview = dict(evidence[0], kind="overview")
    overview["text"] = (
        "# Evidence browser\nThis is a static final graph. "
        "Use the evidence timeline to browse individual observations, followed by MST pairs.\n\n"
        + overview["text"]
    )
    evidence.append(overview)
    display_dir = folder / "display_images"
    display_dir.mkdir(exist_ok=True)
    for step, event in enumerate(evidence):
        rr.set_time_sequence("evidence", step)
        # A fixed display canvas prevents small crops shrinking against the bounds
        # of a larger image from another timeline step. Inference images stay intact.
        with Image.open(event["image"]) as im:
            tile = ImageOps.contain(im.convert("RGB"), (720, 450))
        canvas = Image.new("RGB", (720, 450), "#15191d")
        canvas.paste(tile, ((720 - tile.width) // 2, (450 - tile.height) // 2))
        display = display_dir / f"{step:04d}.jpg"
        canvas.save(display, quality=92)
        event["display_image"] = str(display)
        rr.log("evidence/crop", rr.ImageEncoded(path=str(display)))
        if event["rgb"]:
            rr.log("evidence/rgb", rr.ImageEncoded(path=event["rgb"]))
        else:
            rr.log("evidence/rgb", rr.Clear(recursive=True))
        rr.log(
            "evidence/details",
            rr.TextDocument(event["text"], media_type="text/markdown"),
        )
        selected = event["map_indices"]
        rr.log(
            "world/selected",
            rr.Boxes3D(
                centers=[boxes[i].center for i in selected],
                half_sizes=[boxes[i].extent / 2 for i in selected],
                rotations=[quaternions[i] for i in selected],
                colors=[235, 80, 55],
                radii=0.006,
            ),
        )
    rr.send_blueprint(
        blueprint.Blueprint(
            blueprint.Horizontal(
                blueprint.Vertical(
                    blueprint.Spatial2DView(origin="evidence/rgb", name="Source RGB"),
                    blueprint.Spatial2DView(
                        origin="evidence/crop", name="Actual crop / pair evidence"
                    ),
                ),
                blueprint.Vertical(
                    blueprint.Spatial2DView(
                        origin="diagram", name="Static scene graph"
                    ),
                    blueprint.Spatial3DView(
                        origin="world", name="Static 3D objects + relations"
                    ),
                ),
                blueprint.TextDocumentView(
                    origin="evidence/details",
                    name="Caption, summary and relation reason",
                ),
                column_shares=[2, 3, 2],
            ),
            collapse_panels=True,
        )
    )
    rr.disconnect()
    write_json(folder / "evidence.json", evidence)
    write_evidence_markdown(root, run_name)
    report = json.loads((folder / "report.json").read_text())
    lines = [
        "# Public main + Qwen result",
        "",
        f"Status: {report['status']}",
        "",
        f"Input objects: {report['input_objects']}; retained: {report['retained_objects']}; "
        f"MST candidates: {report['candidate_pairs']}; relations: {report['edges']}.",
        "",
        "[Evidence images, captions and reasons](evidence.md)",
        "",
        "![Scene graph](diagram.png)",
        "",
        "## Same-map comparison",
        "",
        "```json",
        json.dumps(report["comparison"], indent=2),
        "```",
        "",
        "These counts are not ground-truth accuracy. Point overlap follows public main, not paper-text BBox IoU.",
    ]
    (folder / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "viewer": str(folder / "viewer.rrd"),
                "evidence_steps": len(evidence),
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
            }
        )
    )
