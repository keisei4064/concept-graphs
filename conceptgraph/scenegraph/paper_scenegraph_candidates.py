"""Visualize persisted edge-estimation stages; never rerun MST or an LLM."""

import json
from pathlib import Path
import textwrap

import networkx as nx
import numpy as np

from conceptgraph.scenegraph.local_paper_scenegraph import (
    file_hash,
    read_rows,
    write_json,
)
from conceptgraph.scenegraph.paper_scenegraph_view import (
    load_view_geometry,
    log_static_map,
)

STAGES = [
    ("before_mst", "Overlap candidates before MST"),
    ("after_mst", "Saved minimum spanning forest"),
    ("llm_input", "Pairs sent to the LLM (no relation direction yet)"),
    ("llm_result", "LLM decisions: accepted / none of these"),
    ("final", "Final accepted directed relations"),
]
COLORS = {
    "overlap": "#7b8795",
    "mst": "#da8a16",
    "accepted": "#16845d",
    "none": "#cc5260",
    "not_selected": "#cbd1d8",
    "selected": "#cf5e12",
}


def candidate_data(folder):
    """Read the actual saved matrix, forest, requests and final decisions."""
    folder = Path(folder)
    graph = json.loads((folder / "scene_graph_with_evidence.json").read_text())
    forest = json.loads((folder / "candidate_pairs.json").read_text())
    responses = read_rows(folder / "relations.jsonl")
    with np.load(folder / "overlap_matrix.npz") as saved:
        matrix = saved["overlap"]
        ids = saved["map_indices"].tolist()
    by_pair = {
        frozenset((r["pair"]["map_a"], r["pair"]["map_b"])): r for r in responses
    }
    selected = {frozenset((p["map_a"], p["map_b"])): p for p in forest["pairs"]}
    final = {e["pair_id"]: e for e in graph["edges"]}
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if matrix[i, j] <= forest["threshold"]:
                continue
            key = frozenset((ids[i], ids[j]))
            pair = selected.get(key)
            response = by_pair.get(key)
            edge = {
                "a": ids[i],
                "b": ids[j],
                "overlap": float(matrix[i, j]),
                "mst_selected": pair is not None,
                "pair_id": pair["pair_id"] if pair else None,
                "response": response,
                "final_edge": final.get(pair["pair_id"]) if pair else None,
            }
            edges.append(edge)
    assert len(edges) == forest["candidate_graph_edges"]
    assert len(selected) == len(responses) == len(forest["pairs"])
    assert len(final) == sum(e["final_edge"] is not None for e in edges)
    assert len(selected) == sum(e["mst_selected"] for e in edges)
    assert all(key in selected for key in by_pair)
    return graph, edges, responses


def layout(nodes, edges):
    network = nx.Graph()
    network.add_nodes_from(n["map_index"] for n in nodes)
    network.add_edges_from((e["a"], e["b"]) for e in edges)
    active = {i for i in network if network.degree(i)}
    shown = network.subgraph(active).copy() if active else network
    positions = {}
    y = 0.0
    components = sorted(nx.connected_components(shown), key=lambda c: (-len(c), min(c)))
    # Lay out components independently; reuse the same positions for every stage.
    for k in range(0, len(components), 2):
        depths_in_row = []
        for col, component in enumerate(components[k : k + 2]):
            if len(component) == 2:
                # A horizontal pair leaves a readable line between the nodes.
                for j, i in enumerate(sorted(component)):
                    positions[i] = np.array([col * 13.0 + (j - 0.5) * 6.0, y])
                depths_in_row.append(0)
                continue
            sub = shown.subgraph(component)
            anchor = min(component, key=lambda i: (-sub.degree(i), i))
            depths = nx.single_source_shortest_path_length(sub, anchor)
            depths_in_row.append(max(depths.values()))
            for depth in sorted(set(depths.values())):
                level = sorted(i for i in component if depths[i] == depth)
                for j, i in enumerate(level):
                    positions[i] = np.array(
                        [col * 13.0 + (j - (len(level) - 1) / 2) * 3.0, y - depth * 3.4]
                    )
        y -= (max(depths_in_row, default=0) + 1) * 3.4 + 2.0
    return shown, positions


def draw_stage(nodes, edges, stage, output, positions, shown):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(19, 16))
    node_size = 2800
    node_by_id = {n["map_index"]: n for n in nodes}
    nx.draw_networkx_nodes(
        shown,
        positions,
        ax=axis,
        node_color="#e3edf8",
        edgecolors="#6584a3",
        node_size=node_size,
    )
    labels = {
        i: f"#{node_by_id[i]['map_id']}\n"
        + "\n".join(textwrap.wrap(node_by_id[i]["object_tag"], 14))
        for i in shown
    }
    nx.draw_networkx_labels(shown, positions, labels, ax=axis, font_size=10)
    active = 0
    edge_labels = []
    for edge in edges:
        keep = edge["mst_selected"]
        final = edge["final_edge"]
        response = edge["response"]
        if stage == 0:
            color = COLORS["overlap"]
            text = f"{edge['overlap']:.3f}"
            style = "solid"
            directed = False
        elif stage in (1, 2):
            color = COLORS["mst"] if keep else COLORS["not_selected"]
            text = (
                (f"{edge['overlap']:.3f}" if stage == 1 else edge["pair_id"])
                if keep
                else ""
            )
            style = "solid" if keep else "dashed"
            directed = False
        else:
            if not keep:
                if stage == 4:
                    continue
                color = COLORS["not_selected"]
                text = ""
                style = "dashed"
                directed = False
            elif final:
                color = COLORS["accepted"]
                text = final["relation"]
                style = "solid"
                directed = True
            else:
                if stage == 4:
                    continue
                color = COLORS["none"]
                text = (
                    response["output"]["object_relation"]
                    if response["status"] == "ok"
                    else "failed"
                )
                style = "dashed"
                directed = False
        if stage == 0 or keep:
            active += 1
        a, b = (
            (final["source"], final["target"]) if directed else (edge["a"], edge["b"])
        )
        network = nx.DiGraph() if directed else nx.Graph()
        network.add_edge(a, b)
        nx.draw_networkx_edges(
            network,
            positions,
            ax=axis,
            arrows=directed,
            node_size=node_size,
            edge_color=color,
            width=2.4 if keep or stage == 0 else 1.2,
            style=style,
            **({"arrowsize": 22} if directed else {}),
        )
        if text:
            edge_labels.append((a, b, text, color))
    subtitle = (
        "Edge labels are overlap scores."
        if stage < 2
        else "Edge labels are saved pair IDs; these are undirected candidates."
        if stage == 2
        else "Green: accepted direction. Red dashed: no relation / failure. Gray: not selected by MST."
    )
    axis.set_title(
        f"{STAGES[stage][1]}\n{len(nodes)} nodes; {active} active pairs. "
        f"Same node positions across all stages.\n{subtitle}\n"
        f"{len(nodes) - len(shown)} isolated objects omitted from diagram, retained in 3D.",
        fontsize=12,
    )
    axis.axis("off")
    figure.tight_layout()
    figure.canvas.draw()
    for a, b, text, color in edge_labels:
        # Offset in screen space so labels cannot hide short or vertical edges.
        delta = axis.transData.transform(positions[b]) - axis.transData.transform(
            positions[a]
        )
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        axis.annotate(
            text,
            xy=(positions[a] + positions[b]) / 2,
            xytext=normal * 12,
            textcoords="offset points",
            ha="left"
            if normal[0] > 0.35
            else "right"
            if normal[0] < -0.35
            else "center",
            va="bottom"
            if normal[1] > 0.35
            else "top"
            if normal[1] < -0.35
            else "center",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": color},
        )
    figure.savefig(output, dpi=140)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def render_candidates(root, run_name):
    import rerun as rr
    import rerun.blueprint as blueprint

    root = Path(root).resolve()
    folder = root / run_name
    target = folder / "candidate_visualization"
    target.mkdir(exist_ok=True)
    protected = [
        folder / n
        for n in (
            "scene_graph.json",
            "candidate_pairs.json",
            "overlap_matrix.npz",
            "relations.jsonl",
            "captions.jsonl",
            "object_summaries.jsonl",
            "viewer.rrd",
        )
    ]
    before = {str(p): file_hash(p) for p in protected}
    graph, edges, responses = candidate_data(folder)
    shown, positions = layout(graph["nodes"], edges)
    for stage, (name, _) in enumerate(STAGES):
        draw_stage(
            graph["nodes"], edges, stage, target / f"{name}.png", positions, shown
        )
    source, geometry, boxes, quats = load_view_geometry(root)
    rr.init("conceptgraphs_edge_candidates", spawn=False)
    rr.save(str(folder / "candidate_stages.rrd"))
    log_static_map(graph["nodes"], source, geometry, boxes, quats)

    def log_stage(step, stage, selected=None):
        rr.set_time_sequence("candidate_step", step)
        rr.log("world/candidates", rr.Clear(recursive=True))
        rr.log("world/selected", rr.Clear(recursive=True))
        rr.log("diagram", rr.ImageEncoded(path=str(target / f"{STAGES[stage][0]}.png")))
        for k, e in enumerate(edges):
            keep = e["mst_selected"]
            final = e["final_edge"]
            if stage in (1, 2) and not keep:
                continue
            if stage == 4 and not final:
                continue
            color = (
                COLORS["overlap"]
                if stage == 0
                else COLORS["mst"]
                if stage in (1, 2)
                else COLORS["accepted"]
                if final
                else COLORS["none"]
                if keep
                else COLORS["not_selected"]
            )
            color = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
            a, b = (
                (final["source"], final["target"])
                if stage >= 3 and final
                else (e["a"], e["b"])
            )
            p, q = boxes[a].center, boxes[b].center
            if stage >= 3 and final:
                rr.log(
                    f"world/candidates/{k}/edge",
                    rr.Arrows3D(
                        origins=[p], vectors=[q - p], colors=color, radii=0.008
                    ),
                )
                label = final["relation"]
            else:
                rr.log(
                    f"world/candidates/{k}/edge",
                    rr.LineStrips3D([[p, q]], colors=color, radii=0.004),
                )
                label = f"{e['overlap']:.3f}" if stage < 2 else e["pair_id"] or ""
                if stage == 3 and keep and not final:
                    label = "none" if e["response"]["status"] == "ok" else "failed"
            if label:
                rr.log(
                    f"world/candidates/{k}/label",
                    rr.Points3D(
                        [(p + q) / 2], radii=0.003, colors=color, labels=[label]
                    ),
                )
        if selected:
            a, b = selected["pair"]["map_a"], selected["pair"]["map_b"]
            p, q = boxes[a].center, boxes[b].center
            rr.log(
                "world/selected/boxes",
                rr.Boxes3D(
                    centers=[boxes[i].center for i in (a, b)],
                    half_sizes=[boxes[i].extent / 2 for i in (a, b)],
                    rotations=[quats[i] for i in (a, b)],
                    colors=[255, 205, 53],
                    radii=0.009,
                ),
            )
            rr.log(
                "world/selected/pair",
                rr.LineStrips3D([[p, q]], colors=[255, 205, 53], radii=0.014),
            )
            pair_image = folder / "pair_images" / f"{selected['pair']['pair_id']}.jpg"
            rr.log("pair/image", rr.ImageEncoded(path=str(pair_image)))
            text = (
                f"# LLM input pair {selected['pair']['pair_id']}\n"
                f"Map #{source[a]['map_id']} / Map #{source[b]['map_id']}\n\n"
                f"Overlap: {selected['pair']['weight']:.6f}\n\n"
                "Yellow: selected pair. Orange: all saved LLM candidates. No direction is assigned yet.\n\n"
                "The pair image is for inspection; the LLM received only the following text.\n\n"
                "## Actual input\n```json\n"
                + json.dumps(selected["input"], indent=2)
                + "\n```\n\n"
                "## Saved LLM response (shown separately)\n```json\n"
                + json.dumps(selected.get("output", selected.get("error")), indent=2)
                + "\n```"
            )
        else:
            rr.log("pair/image", rr.Clear(recursive=True))
            text = (
                f"# Step {stage}: {STAGES[stage][1]}\n\n"
                f"Overlap graph: {len(edges)} pairs. Saved MST / actual LLM input: {len(responses)} pairs. "
                f"Accepted: {len(graph['edges'])}; not accepted: {len(responses) - len(graph['edges'])}.\n\n"
                "Gray: overlap candidate. Orange: saved minimum spanning forest / LLM candidate. "
                "Green arrow: accepted relation. Red: none of these / failure. Yellow: currently inspected pair.\n\n"
                f"Steps 0-4 show pipeline stages. Steps 5-{4 + len(responses)} inspect individual LLM inputs. "
                "The last step repeats the first LLM input pair for orientation.\n\n"
                "These are persisted public-main minimum-spanning-forest results. No maximum-tree substitution or new inference. "
                "Node positions in the 2D diagrams are fixed across stages. The 3D map and BBoxes are also fixed."
            )
        rr.log("details", rr.TextDocument(text, media_type="text/markdown"))

    for stage in range(5):
        log_stage(stage, stage)
    timeline = [
        {"step": i, "stage": name, "pair_id": None}
        for i, (name, _) in enumerate(STAGES)
    ]
    for i, row in enumerate(responses, 5):
        log_stage(i, 2, row)
        timeline.append(
            {"step": i, "stage": "llm_input_pair", "pair_id": row["pair"]["pair_id"]}
        )
    log_stage(
        5 + len(responses), 2 if responses else 0, responses[0] if responses else None
    )
    timeline.append(
        {
            "step": 5 + len(responses),
            "stage": "overview_llm_input",
            "pair_id": responses[0]["pair"]["pair_id"] if responses else None,
        }
    )
    rr.send_blueprint(
        blueprint.Blueprint(
            blueprint.Horizontal(
                blueprint.Spatial2DView(
                    origin="diagram", name="Pipeline stage: fixed node positions"
                ),
                blueprint.Spatial3DView(origin="world", name="3D candidates + BBoxes"),
                blueprint.Vertical(
                    blueprint.TextDocumentView(
                        origin="details", name="Stage / actual LLM input"
                    ),
                    blueprint.Spatial2DView(
                        origin="pair/image", name="Pair crops (reference only)"
                    ),
                ),
                column_shares=[3, 3, 2],
            ),
            collapse_panels=True,
        )
    )
    rr.disconnect()
    assert before == {str(p): file_hash(p) for p in protected}
    write_json(target / "timeline.json", timeline)
    write_json(
        target / "stages.json",
        {
            "overlap_candidates": len(edges),
            "mst_candidates": len(responses),
            "llm_inputs": len(responses),
            "accepted": len(graph["edges"]),
            "not_accepted": len(responses) - len(graph["edges"]),
            "source_data_sha256": before,
            "timeline_steps": len(timeline),
            "edges": edges,
            "note": "Read-only visualization of saved public-main results; no MST/LLM rerun.",
        },
    )
    markdown = [
        "# Edge estimationの候補を段階ごとに見る",
        "",
        "保存済み結果を可視化したもので、MST計算やLLM推論は再実行していません。",
        "",
        f"**重複候補{len(edges)}組 → MST選択{len(responses)}組 → LLM入力{len(responses)}組 → 採用{len(graph['edges'])}件**。",
        f"MST後とLLM直前は同じ{len(responses)}候補です。この間に別の候補選別はありません。",
        "",
        "全図でノード位置を固定。灰色は重複候補、橙はMST/LLM候補、緑は採用された有向関係、赤は該当なし。",
        "MST前後の候補にはon/inの意味や方向を付けていません。LLMの方向誤りも元の応答のまま表示します。",
        "",
    ]
    for i, (name, title) in enumerate(STAGES):
        markdown += [f"## {i}. {title}", "", f"![{title}]({name}.png)", ""]
    markdown += [
        "## LLMへ実際に渡した候補",
        "",
        "入力JSON・応答・画像は[根拠一覧](../evidence.md#候補ペアの関係推定)にも掲載しています。",
        f"GUIのcandidate_step 5〜{4 + len(responses)}で各ペアを黄色に強調し、入力JSONを表示します。",
        "",
        "| step | pair ID | 物体A | 物体B | 重複率 | LLM返却 |",
        "|---|---|---|---|---:|---|",
    ]
    for step, r in enumerate(responses, 5):
        a, b = r["pair"]["map_a"], r["pair"]["map_b"]
        markdown.append(
            f"| {step} | {r['pair']['pair_id']} | #{source[a]['map_id']} | #{source[b]['map_id']} | {r['pair']['weight']:.4f} | {r.get('output', {}).get('object_relation', 'failed')} |"
        )
    (target / "stages.md").write_text("\n".join(markdown))
    print(
        json.dumps(
            {
                "overlap_candidates": len(edges),
                "mst_candidates": len(responses),
                "llm_inputs": len(responses),
                "accepted": len(graph["edges"]),
                "recording": str(folder / "candidate_stages.rrd"),
            }
        )
    )
