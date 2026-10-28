"""Export recorded slide assets and reproduce geometry examples without inference."""
# The offline guard must precede third-party imports.
# ruff: noqa: E402

import copy
from functools import lru_cache
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ROOT = REPO / "outputs/local-only/paper_scenegraph_room0/attempt1"
RUN = ROOT / "full_main"
DATA = REPO / "outputs/local-only/assets/data/Replica/room0"
CACHE = DATA / "exps/local_room0_full_v1/detections"
LOGS = REPO / "outputs/local-only/presentation_review/20261027/asset_build"
LOGS.mkdir(parents=True, exist_ok=True)

# Native dependencies are also isolated. Torch needs device files for /dev/urandom.
namespace = os.readlink("/proc/self/ns/net")
if os.environ.get("CG_SLIDE_HOST_NETNS", namespace) == namespace:
    if not shutil.which("bwrap"):
        raise RuntimeError("bwrap is required for offline rendering")
    environment = dict(os.environ, CG_SLIDE_HOST_NETNS=namespace)
    subprocess.run(
        [
            "bwrap",
            "--unshare-net",
            "--bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            sys.executable,
            str(Path(__file__).resolve()),
        ],
        env=environment,
        check=True,
    )
    sys.exit(0)

sys.path.insert(0, str(REPO))
os.environ["MPLCONFIGDIR"] = str(LOGS / "mpl-config")
logging.basicConfig(filename=LOGS / "build.log", level=logging.INFO)
from conceptgraph.utils.local_runtime import install_local_network_guard

install_local_network_guard(LOGS / "network.jsonl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial import cKDTree
import torch
import yaml
from conceptgraph.slam.utils import (
    detections_to_obj_pcd_and_bbox,
    filter_gobs,
    get_bounding_box,
    init_process_pcd,
    merge_obj2_into_obj1,
)
from conceptgraph.utils.ious import mask_subtract_contained

SOURCES = {}
ASSETS = {}
PACKS = [
    ("A_chair", 41, "Wingback armchair"),
    ("B_sofa", 53, "White sofa"),
    ("C_planter", 2, "Planting container"),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(REPO))


def source(path):
    path = Path(path)
    SOURCES.setdefault(relative(path), sha(path))
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def rows(path):
    return [
        json.loads(line)
        for line in source(path).read_text().splitlines()
        if line.strip()
    ]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def register(path, kind, **context):
    ASSETS[str(Path(path).relative_to(OUT))] = dict(kind=kind, **context)


def save_image(image, path, kind="extracted", **context):
    image.save(path)
    register(path, kind, **context)


def text_card(text, path, mono=False, columns=55, font_size=30):
    font = ImageFont.truetype(MONO if mono else FONT, font_size)
    lines = []
    for line in text.splitlines():
        lines.extend(
            textwrap.wrap(
                line, width=columns, replace_whitespace=False, drop_whitespace=False
            )
            or [""]
        )
    width = max(
        400, int(max((font.getlength(line) for line in lines), default=100) + 60)
    )
    line_height = int(font_size * 1.42)
    height = max(100, len(lines) * line_height + 50)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((25, 20 + i * line_height), line, font=font, fill="#142236")
    save_image(image, path, "rendered_saved_text")
    # Editable text version for vector-aware presentation tools.
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for i, line in enumerate(lines):
        svg.append(
            f'<text x="25" y="{20 + font_size + i * line_height}" font-family="{"DejaVu Sans Mono" if mono else "DejaVu Sans"}" font-size="{font_size}" fill="#142236" xml:space="preserve">{escape(line)}</text>'
        )
    svg.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(svg))
    register(path.with_suffix(".svg"), "rendered_saved_text")


def array(path):
    with np.load(source(path)) as saved:
        return saved[saved.files[0]]


def pickled(path):
    with gzip.open(source(path), "rb") as stream:
        return pickle.load(stream)  # Trusted local experiment outputs only.


CFG = read_json(DATA / "exps/room0_reference_graph_attempt1/config_params.json")
CAM = yaml.safe_load(
    source(REPO / "conceptgraph/dataset/dataconfigs/replica/replica.yaml").read_text()
)["camera_params"]
K = np.array([[CAM["fx"], 0, CAM["cx"]], [0, CAM["fy"], CAM["cy"]], [0, 0, 1.0]])
POSES = np.loadtxt(source(DATA / "traj.txt")).astype(np.float32).reshape(-1, 4, 4)
NODES = read_json(ROOT / "source_objects.json")
OBS = rows(ROOT / "observations.jsonl")
CAPTIONS = rows(RUN / "captions.jsonl")
SUMMARIES = {r["map_index"]: r for r in rows(RUN / "object_summaries.jsonl")}
RELATIONS = rows(RUN / "relations.jsonl")


@lru_cache(maxsize=3)
def frame(name):
    folder = CACHE / Path(name).stem
    image = np.array(Image.open(source(DATA / "results" / name)).convert("RGB"))
    raw = {
        key: array(folder / f"{key}.npz")
        for key in ("xyxy", "mask", "confidence", "class_id")
    }
    raw["classes"] = pickled(folder / "classes.pkl.gz")
    raw["raw_index"] = np.arange(len(raw["xyxy"]))
    raw["sam_mask"] = raw["mask"].copy()
    # filter_gobs also aligns semantic fields; geometry needs only empty records.
    raw["detection_class_labels"] = [
        f"{raw['classes'][class_id]} {i}" for i, class_id in enumerate(raw["class_id"])
    ]
    raw["captions"] = [
        {"id": str(i), "name": raw["classes"][class_id], "caption": None}
        for i, class_id in enumerate(raw["class_id"])
    ]
    feats = array(folder / "image_feats.npz")
    # The saved image features are per detection; this cache has no per-detection
    # text features, and the original fusion function does not update text_ft.
    texts = None
    crops = pickled(folder / "image_crops.pkl.gz")
    filtered = filter_gobs(
        raw,
        image,
        skip_bg=CFG["skip_bg"],
        BG_CLASSES=["wall", "floor", "ceiling"],
        mask_area_threshold=CFG["mask_area_threshold"],
        max_bbox_area_ratio=CFG["max_bbox_area_ratio"],
        mask_conf_threshold=CFG["mask_conf_threshold"],
    )
    filtered["mask"] = mask_subtract_contained(filtered["xyxy"], filtered["mask"])
    return image, filtered, feats, texts, crops


def observation_data(row):
    image, gobs, feats, texts, crops = frame(row["source_frame"])
    distances = np.linalg.norm(gobs["xyxy"] - row["bbox_xyxy"], axis=1)
    index = int(distances.argmin())
    assert distances[index] < 1e-3, row["source_frame"]
    x1, y1, x2, y2 = row["crop_xyxy"]
    mask = gobs["mask"][index]
    with np.load(source(Path(row["crop_path"]).with_suffix(".mask.npz"))) as saved:
        np.testing.assert_array_equal(mask[y1:y2, x1:x2], saved["mask"])
    raw_index = int(gobs["raw_index"][index])
    feature = torch.as_tensor(feats[raw_index]).float().clone()
    text_feature = None
    return image, mask, crops[raw_index].copy(), feature, text_feature, gobs, index


POINTS = {}


def observation_cloud(row):
    key = (row["map_index"], row["observation_index"])
    if key in POINTS:
        return copy.deepcopy(POINTS[key])
    image, mask, _, feature, text_feature, _, _ = observation_data(row)
    number = int(Path(row["source_frame"]).stem.removeprefix("frame"))
    depth = (
        np.asarray(
            Image.open(source(DATA / "results" / f"depth{number:06d}.png"))
        ).astype(float)
        / CAM["png_depth_scale"]
    )
    item = detections_to_obj_pcd_and_bbox(
        depth_array=depth,
        masks=mask[None],
        cam_K=K,
        image_rgb=image,
        trans_pose=POSES[number],
        min_points_threshold=CFG["min_points_threshold"],
        spatial_sim_type=CFG["spatial_sim_type"],
        obj_pcd_max_points=CFG["obj_pcd_max_points"],
        device="cpu",
    )[0]
    if item is None:
        raise ValueError(f"Invalid point cloud for {key}")
    raw = item["pcd"]
    voxel = raw.voxel_down_sample(CFG["downsample_voxel_size"])
    clean = init_process_pcd(
        copy.deepcopy(raw),
        CFG["downsample_voxel_size"],
        CFG["dbscan_remove_noise"],
        CFG["dbscan_eps"],
        CFG["dbscan_min_points"],
    )
    POINTS[key] = {
        "raw": raw,
        "before": voxel,
        "clean": clean,
        "clip_ft": feature,
        "text_ft": text_feature,
    }
    return copy.deepcopy(POINTS[key])


def as_object(cloud, row):
    return {
        "id": NODES[row["map_index"]]["uuid"],
        "class_name": NODES[row["map_index"]]["class_name"],
        "pcd": copy.deepcopy(cloud["clean"]),
        "bbox": get_bounding_box(CFG["spatial_sim_type"], cloud["clean"]),
        "clip_ft": cloud["clip_ft"].clone(),
        "num_detections": 1,
        "n_points": len(cloud["clean"].points),
        "image_idx": [row["processing_frame"]],
        "color_path": [row["source_image"]],
    }


def merge(first, second):
    return merge_obj2_into_obj1(
        first,
        second,
        CFG["downsample_voxel_size"],
        CFG["dbscan_remove_noise"],
        CFG["dbscan_eps"],
        CFG["dbscan_min_points"],
        CFG["spatial_sim_type"],
        "cpu",
        run_dbscan=False,
    )


def view_points(points, pose):
    camera = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    return camera[:, [0, 2, 1]] * np.array([1, 1, -1])


def render_cloud(series, boxes, pose, bounds_points, path, oblique=False):
    fig = plt.figure(figsize=(6, 6), dpi=240)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    for points, colors, size in series:
        transformed = view_points(points, pose)
        if len(transformed):
            ax.scatter(
                *transformed.T,
                c=colors,
                s=size,
                depthshade=False,
                linewidths=0,
                alpha=0.95,
                rasterized=True,
            )
    for box, color in boxes:
        vertices = view_points(np.asarray(box.get_box_points()), pose)
        lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box).lines
        for a, b in np.asarray(lines):
            ax.plot(*vertices[[a, b]].T, color=color, linewidth=1.6, alpha=0.95)
    bounds = view_points(bounds_points, pose)
    center = (bounds.min(0) + bounds.max(0)) / 2
    half = max(np.ptp(bounds, axis=0).max() / 2, 0.02) * 1.1
    ax.set(
        xlim=(center[0] - half, center[0] + half),
        ylim=(center[1] - half, center[1] + half),
        zlim=(center[2] - half, center[2] + half),
    )
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15 if oblique else 6, azim=-60 if oblique else -90)
    ax.set_axis_off()
    fig.savefig(path, transparent=True, pad_inches=0)
    plt.close(fig)
    register(path, "reconstructed_geometry", oblique=oblique)
    with Image.open(path) as image:
        white = Image.new("RGB", image.size, "white")
        white.paste(image, mask=image.getchannel("A"))
        white_path = path.with_name(path.stem + "_white.png")
        save_image(white, white_path, "reconstructed_geometry")


def preview(folder, items, output, title):
    width, cell_height = 420, 330
    sheet = Image.new(
        "RGB", (width * 4, 70 + cell_height * ((len(items) + 3) // 4)), "#f3f5f7"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 18)
    draw.text((20, 20), title, font=ImageFont.truetype(FONT, 26), fill="#172638")
    for k, (name, label) in enumerate(items):
        with Image.open(folder / name) as image:
            image = image.convert("RGBA")
            tile = ImageOps.contain(image, (width - 20, cell_height - 50))
            x, y = k % 4 * width, k // 4 * cell_height + 70
            sheet.paste(
                tile,
                (
                    x + (width - tile.width) // 2,
                    y + (cell_height - 50 - tile.height) // 2,
                ),
                tile,
            )
        draw.text((x + 12, y + cell_height - 40), label, font=font, fill="#172638")
    sheet.save(output, quality=94)
    register(output, "preview")


def crop_geometry_groups(folder):
    """Remove empty margins with one shared pixel crop per comparison/view."""
    for family in ("projection", "fusion"):
        states = (
            ("before", "after", "rgb")
            if family == "projection"
            else ("before", "after", "after_rgb")
        )
        for view in ("front", "oblique"):
            paths = [folder / f"{family}_{state}_{view}.png" for state in states]
            images = [Image.open(path).copy() for path in paths]
            boxes = [im.getchannel("A").getbbox() for im in images]
            box = (
                max(0, min(b[0] for b in boxes) - 25),
                max(0, min(b[1] for b in boxes) - 25),
                min(images[0].width, max(b[2] for b in boxes) + 25),
                min(images[0].height, max(b[3] for b in boxes) + 25),
            )
            for path, image in zip(paths, images):
                image = image.crop(box)
                image.save(path)
                ASSETS[str(path.relative_to(OUT))]["shared_comparison_crop"] = list(box)
                white = Image.new("RGB", image.size, "white")
                white.paste(image, mask=image.getchannel("A"))
                white.save(path.with_name(path.stem + "_white.png"))


def caption_assets(folder, index):
    captions = [
        r
        for r in CAPTIONS
        if r["observation"]["map_index"] == index and r["status"] == "ok"
    ]
    records = []
    for k, row in enumerate(captions, 1):
        name = f"caption_{k:02d}"
        source_path = source(row["observation"]["crop_path"])
        shutil.copy2(source_path, folder / f"{name}_crop.png")
        register(folder / f"{name}_crop.png", "extracted", source=relative(source_path))
        (folder / f"{name}_full.txt").write_text(row["output"] + "\n")
        register(folder / f"{name}_full.txt", "saved_model_output")
        plain = re.sub(r"\*\*|^>\s?", "", row["output"], flags=re.MULTILINE)
        sentence = re.split(r"(?<=[.!?])\s+", plain.strip())[0]
        shortened = len(sentence) > 200
        excerpt = sentence if not shortened else sentence[:200].rsplit(" ", 1)[0] + "…"
        (folder / f"{name}_display.txt").write_text(excerpt + "\n")
        register(
            folder / f"{name}_display.txt",
            "excerpt_of_saved_output",
            truncated=shortened,
        )
        if k in (1, 2, len(captions)):
            text_card(excerpt, folder / f"{name}_card.png", columns=30, font_size=32)
        records.append(
            {
                "caption_number": k,
                "source_frame": row["observation"]["source_frame"],
                "observation_index": row["observation"]["observation_index"],
                "response_key": row["response_key"],
                "excerpt": excerpt,
                "excerpt_truncated": shortened,
            }
        )
    write_json(folder / "caption_sources.json", records)
    summary = SUMMARIES[index]["output"]
    write_json(folder / "summary.json", summary)
    register(
        folder / "summary.json",
        "saved_model_output",
        response_key=SUMMARIES[index]["response_key"],
    )
    text_card(
        json.dumps(summary, ensure_ascii=False, indent=2),
        folder / "summary_card.png",
        mono=True,
        columns=62,
        font_size=27,
    )
    inline = (
        "{\n"
        + ",\n".join(
            f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in summary.items()
        )
        + "\n}"
    )
    (folder / "summary_inline.txt").write_text(inline + "\n")
    text_card(
        inline, folder / "summary_inline_card.png", mono=True, columns=120, font_size=26
    )
    preview(
        folder,
        [
            (f"caption_{k:02d}_crop.png", f"View {k}: {r['source_frame']}")
            for k, r in enumerate(records, 1)
        ],
        folder / "caption_choices.jpg",
        "Recorded LVLM inputs; confidence order",
    )
    return records


def sam_assets(folder, image, gobs, selection):
    """Keep original SAM masks separate from containment-subtracted map inputs."""
    choices = []
    for k, i in enumerate(selection, 1):
        mask = gobs["sam_mask"][i]
        ys, xs = np.nonzero(mask)
        box = (
            max(0, int(xs.min()) - 10),
            max(0, int(ys.min()) - 10),
            min(image.shape[1], int(xs.max()) + 11),
            min(image.shape[0], int(ys.max()) + 11),
        )
        binary = mask.astype(np.uint8) * 255
        black = np.where(mask[..., None], image, 0).astype(np.uint8)
        white = np.where(mask[..., None], image, 255).astype(np.uint8)
        rgba = np.dstack((image, binary))
        overlay = image.copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([25, 119, 214])).astype(
            np.uint8
        )
        variants = {
            "binary": binary,
            "rgb_black": black,
            "rgb_white": white,
            "rgba": rgba,
            "overlay": overlay,
        }
        for suffix, pixels in variants.items():
            save_image(
                Image.fromarray(pixels),
                folder / f"sam_{k:02d}_{suffix}.png",
                "original_cached_sam_mask_visualization",
                raw_detection_index=int(gobs["raw_index"][i]),
            )
        for suffix, pixels in (
            ("rgb_crop", image),
            ("binary_crop", binary),
            ("masked_crop_black", black),
            ("masked_crop_white", white),
            ("crop_rgba", rgba),
        ):
            save_image(
                Image.fromarray(pixels).crop(box),
                folder / f"sam_{k:02d}_{suffix}.png",
                "original_cached_sam_mask_visualization",
                crop_xyxy=list(box),
            )
        # These are pixelwise identities, not visual judgments.
        np.testing.assert_array_equal(black[mask], image[mask])
        assert not black[~mask].any()
        label = gobs["classes"][gobs["class_id"][i]]
        choices += [
            (f"sam_{k:02d}_overlay.png", f"Choice {k}: {label}"),
            (f"sam_{k:02d}_binary.png", "Original SAM mask (binary)"),
            (f"sam_{k:02d}_rgb_black.png", "RGB x mask: black background"),
            (f"sam_{k:02d}_masked_crop_white.png", "Masked crop: white background"),
        ]
    preview(
        folder,
        choices,
        folder / "masks_choices.jpg",
        "SAM output choices; same-frame foreground objects; no containment subtraction",
    )
    preview(
        folder,
        [
            ("sam_01_rgb_crop.png", "Input RGB crop"),
            ("sam_01_binary_crop.png", "Aligned binary mask crop"),
            ("sam_01_masked_crop_black.png", "Pixelwise RGB x mask"),
            ("clip_same_01.png", "Actual CLIP input (BBox crop)"),
        ],
        folder / "p12_mask_and_clip.jpg",
        "p12: three aligned crops, plus the separate recorded CLIP input",
    )


def build_pack(name, index, title):
    print("Building", name, flush=True)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    views = [r for r in OBS if r["map_index"] == index and r["status"] == "ready"]
    incoming = views[0]
    image, mask, clip, feature, _, gobs, selected = observation_data(incoming)
    save_image(
        Image.fromarray(image),
        folder / "rgb.png",
        source=relative(incoming["source_image"]),
    )
    # Reviewed foreground examples, rather than selecting large background masks.
    reviewed = {41: [5, 0, 4, 10], 53: [0, 2, 10, 1], 2: [0, 1, 8, 9]}
    selection = [
        int(np.flatnonzero(gobs["raw_index"] == i)[0]) for i in reviewed[index]
    ]
    assert selection[0] == selected
    palette = [
        np.array([25, 119, 214]),
        np.array([239, 126, 35]),
        np.array([31, 160, 105]),
        np.array([145, 80, 190]),
    ]
    segmentation = []
    for k, i in enumerate(selection, 1):
        m = gobs["mask"][i]
        rgba = np.dstack((image, m.astype(np.uint8) * 255))
        binary = Image.fromarray(m.astype(np.uint8) * 255)
        save_image(
            binary,
            folder / f"mask_{k:02d}_binary.png",
            "recorded_mask_after_original_filtering",
        )
        save_image(
            Image.fromarray(rgba),
            folder / f"mask_{k:02d}_rgba.png",
            "recorded_mask_after_original_filtering",
        )
        overlay = image.copy()
        overlay[m] = (0.4 * overlay[m] + 0.6 * palette[k - 1]).astype(np.uint8)
        save_image(
            Image.fromarray(overlay),
            folder / f"mask_{k:02d}_overlay.png",
            "mask_visualization",
        )
        ys, xs = np.nonzero(m)
        box = (
            max(0, int(xs.min()) - 10),
            max(0, int(ys.min()) - 10),
            min(image.shape[1], int(xs.max()) + 11),
            min(image.shape[0], int(ys.max()) + 11),
        )
        save_image(
            Image.fromarray(rgba).crop(box),
            folder / f"mask_{k:02d}_crop_rgba.png",
            "mask_visualization",
        )
        segmentation.append(
            {
                "slot": k,
                "raw_detection_index": int(gobs["raw_index"][i]),
                "detector_class": gobs["classes"][gobs["class_id"][i]],
                "mask_pixels": int(m.sum()),
                "sam_mask_pixels": int(gobs["sam_mask"][i].sum()),
                "map_input_pixels_removed_from_sam": int(
                    (gobs["sam_mask"][i] & ~m).sum()
                ),
            }
        )
    shutil.copy2(folder / "mask_01_rgba.png", folder / "masked_rgb.png")
    register(folder / "masked_rgb.png", "mask_visualization")
    x1, y1, x2, y2 = incoming["crop_xyxy"]
    save_image(
        Image.fromarray(np.dstack((image, mask.astype(np.uint8) * 255))).crop(
            (x1, y1, x2, y2)
        ),
        folder / "masked_rgb_crop.png",
        "mask_visualization",
    )
    save_image(
        clip, folder / "clip_same_01.png", source="Original CLIP image_crops cache"
    )
    _, _, clip2, feature2, _, _, _ = observation_data(views[1])
    save_image(
        clip2, folder / "clip_same_02.png", source="Original CLIP image_crops cache"
    )
    different_index = 2 if index != 2 else 41
    different = next(
        r for r in OBS if r["map_index"] == different_index and r["status"] == "ready"
    )
    _, _, other_clip, other_feature, _, _, _ = observation_data(different)
    save_image(
        other_clip,
        folder / "clip_different.png",
        source="Original CLIP image_crops cache",
    )
    sam_assets(folder, image, gobs, selection)

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    similarities = {
        "same_object_cosine": cos(feature, feature2),
        "different_object_cosine": cos(feature, other_feature),
        "feature_dimension": len(feature),
        "source": "Saved CLIP image features; no embedding inference",
        "frames": [
            incoming["source_frame"],
            views[1]["source_frame"],
            different["source_frame"],
        ],
    }
    write_json(folder / "clip_features.json", similarities)
    assert (
        similarities["same_object_cosine"] > similarities["different_object_cosine"]
    ), similarities
    cloud = observation_cloud(incoming)
    before, clean = cloud["before"], cloud["clean"]
    before_points = np.asarray(before.points)
    after_points = np.asarray(clean.points)
    kept = cKDTree(after_points).query(before_points)[0] < 1e-7
    before_colors = np.tile([0.62, 0.64, 0.67], (len(before_points), 1))
    before_colors[kept] = [0.06, 0.38, 0.78]
    frame_id = int(Path(incoming["source_frame"]).stem.removeprefix("frame"))
    pose = POSES[frame_id]
    for suffix, oblique in [("front", False), ("oblique", True)]:
        render_cloud(
            [(before_points, before_colors, 5)],
            [],
            pose,
            before_points,
            folder / f"projection_before_{suffix}.png",
            oblique,
        )
        render_cloud(
            [(after_points, "#126dcc", 5)],
            [],
            pose,
            before_points,
            folder / f"projection_after_{suffix}.png",
            oblique,
        )
        render_cloud(
            [(after_points, np.asarray(clean.colors), 5)],
            [],
            pose,
            before_points,
            folder / f"projection_rgb_{suffix}.png",
            oblique,
        )
    # Reconstruct a partial past map using only earlier recorded observations.
    earlier = sorted(
        (r for r in views if r["source_frame"] < incoming["source_frame"]),
        key=lambda r: r["source_frame"],
    )[-3:]
    if not earlier:
        raise ValueError("Need earlier observations for a causal fusion example")
    old = as_object(observation_cloud(earlier[0]), earlier[0])
    for row in earlier[1:]:
        old = merge(old, as_object(observation_cloud(row), row))
    new = as_object(cloud, incoming)
    merged = merge(copy.deepcopy(old), copy.deepcopy(new))
    old_points = np.asarray(old["pcd"].points)
    new_points = np.asarray(new["pcd"].points)
    merged_points = np.asarray(merged["pcd"].points)
    bounds = np.concatenate([old_points, new_points, merged_points])
    for suffix, oblique in [("front", False), ("oblique", True)]:
        render_cloud(
            [(old_points, "#2278d1", 4), (new_points, "#f28b23", 4)],
            [(old["bbox"], "#145fb0"), (new["bbox"], "#df7214")],
            pose,
            bounds,
            folder / f"fusion_before_{suffix}.png",
            oblique,
        )
        render_cloud(
            [(merged_points, "#16845d", 4)],
            [(merged["bbox"], "#08704c")],
            pose,
            bounds,
            folder / f"fusion_after_{suffix}.png",
            oblique,
        )
        render_cloud(
            [(merged_points, np.asarray(merged["pcd"].colors), 4)],
            [(merged["bbox"], "#08704c")],
            pose,
            bounds,
            folder / f"fusion_after_rgb_{suffix}.png",
            oblique,
        )
    crop_geometry_groups(folder)
    caption_records = caption_assets(folder, index)
    info = {
        "map_index": index,
        "map_id": NODES[index]["map_id"],
        "uuid": NODES[index]["uuid"],
        "title": title,
        "incoming_source_frame": incoming["source_frame"],
        "mask_verified_against_saved_caption_crop": True,
        "segmentation_outputs": segmentation,
        "clip": similarities,
        "projection": {
            "raw_points_after_original_dynamic_sampling": len(cloud["raw"].points),
            "voxel_points": len(before_points),
            "retained_points": len(after_points),
            "removed_points": int((~kept).sum()),
            "removed_point_color": "gray",
            "method": "Original detections_to_obj_pcd_and_bbox + init_process_pcd, CPU",
        },
        "fusion": {
            "earlier_source_frames": [r["source_frame"] for r in earlier],
            "incoming_source_frame": incoming["source_frame"],
            "old_points": len(old_points),
            "incoming_points": len(new_points),
            "union_before_downsampling": len(old_points) + len(new_points),
            "fused_points": len(merged_points),
            "observations": merged["num_detections"],
            "method": "Original merge_obj2_into_obj1 with run_dbscan=False, CPU",
            "is_historical_map_snapshot": False,
            "partial_reconstruction": True,
        },
        "captions": caption_records,
    }
    write_json(folder / "provenance.json", info)
    preview(
        folder,
        [
            ("rgb.png", "Source RGB"),
            ("mask_01_overlay.png", "3D input mask (not raw SAM)"),
            ("clip_same_01.png", "Actual CLIP input"),
            ("clip_same_02.png", "Same object, another view"),
            ("projection_before_front.png", "Before DBSCAN; removed = gray"),
            ("projection_after_front.png", "After DBSCAN"),
            ("fusion_before_front.png", "Existing = blue / incoming = orange"),
            ("fusion_after_front.png", "Fused = green"),
            ("caption_01_crop.png", "Actual LVLM input: view 1"),
            ("caption_02_crop.png", "Actual LVLM input: view 2"),
            ("projection_before_oblique.png", "Before DBSCAN: alternate view"),
            ("fusion_before_oblique.png", "Fusion: alternate view"),
        ],
        folder / "preview.jpg",
        f"{name}: {title}; Map #{NODES[index]['map_id']}",
    )
    print(name, "done:", info["projection"], info["fusion"], flush=True)
    return info


def relation_assets():
    folder = OUT / "relation_text"
    folder.mkdir(exist_ok=True)
    for pair_id in ("2_7", "19_29"):
        row = next(r for r in RELATIONS if r["pair"]["pair_id"] == pair_id)
        record = read_json(ROOT / "vlm_cache" / f"relation_{row['response_key']}.json")
        prompt = record["contract"]["messages"][0]["content"]
        (folder / f"{pair_id}_prompt_full.txt").write_text(prompt + "\n")
        register(
            folder / f"{pair_id}_prompt_full.txt",
            "saved_model_input",
            response_key=row["response_key"],
        )
        write_json(folder / f"{pair_id}_input.json", row["input"])
        write_json(folder / f"{pair_id}_response.json", row["output"])
        # Presentation excerpt: original JSON only, rather than inventing a new prompt.
        (folder / f"{pair_id}_prompt_display.txt").write_text(
            json.dumps(row["input"], indent=2) + "\n"
        )
        text_card(
            json.dumps(row["input"], indent=2),
            folder / f"{pair_id}_input_card.png",
            mono=True,
            columns=54,
            font_size=26,
        )
        instruction = 'Produce an "object_relation" field that best describes the relationship between the two objects.'
        assert instruction in prompt
        excerpt = instruction + "\n\n" + json.dumps(row["input"], indent=2)
        text_card(
            excerpt,
            folder / f"{pair_id}_prompt_card.png",
            mono=True,
            columns=54,
            font_size=26,
        )
        compact = []
        for key, value in row["input"].items():
            compact += [
                f"{key} (id={value['id']}): {value['object_tag']}",
                f"  center: {value['bbox_center']}",
                f"  extent: {value['bbox_extent']}",
            ]
        compact += ["Output: reason + object_relation"]
        compact = "\n".join(compact)
        (folder / f"{pair_id}_input_compact.txt").write_text(compact + "\n")
        text_card(
            compact,
            folder / f"{pair_id}_input_compact_card.png",
            columns=60,
            font_size=28,
        )
        for name in ("input.json", "response.json", "prompt_display.txt"):
            register(
                folder / f"{pair_id}_{name}",
                "excerpt_or_saved_record",
                response_key=row["response_key"],
            )


def overview_pdf():
    pages = []
    for name, _, _ in PACKS:
        for filename in (
            "preview.jpg",
            "masks_choices.jpg",
            "p12_mask_and_clip.jpg",
            "caption_choices.jpg",
        ):
            with Image.open(OUT / name / filename) as image:
                pages.append(image.convert("RGB"))
    path = OUT / "候補一覧.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150)
    register(path, "preview")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # Keep speaker-specific draft names out of published source files.
    pdf_override = os.environ.get("CG_SLIDES_SOURCE_PDF")
    pdf_candidates = sorted((REPO / "docs/発表").glob("*.pdf"))
    if pdf_override:
        source_pdf = Path(pdf_override).expanduser()
    elif len(pdf_candidates) == 1:
        source_pdf = pdf_candidates[0]
    else:
        raise ValueError("Set CG_SLIDES_SOURCE_PDF to the local presentation PDF")
    protected = [
        RUN / name
        for name in (
            "scene_graph.json",
            "candidate_pairs.json",
            "relations.jsonl",
            "captions.jsonl",
            "object_summaries.jsonl",
            "viewer.rrd",
            "candidate_stages.rrd",
        )
    ]
    protected_before = {relative(p): sha(p) for p in protected}
    reports = [build_pack(*pack) for pack in PACKS]
    relation_assets()
    overview_pdf()
    for path in (
        REPO / "conceptgraph/slam/utils.py",
        REPO / "conceptgraph/utils/ious.py",
        Path(__file__),
        source_pdf,
    ):
        source(path)
    # Verify original data were never modified; hash only files actually used.
    for name, digest in SOURCES.items():
        assert sha(REPO / name) == digest, name
    assert protected_before == {relative(p): sha(p) for p in protected}
    for path in OUT.rglob("*"):
        if (
            path.is_file()
            and path.suffix in (".png", ".svg", ".jpg", ".json", ".txt", ".pdf")
            and path.name not in ("manifest.json", "validation.json")
        ):
            ASSETS.setdefault(str(path.relative_to(OUT)), {"kind": "metadata"})
    for name, details in ASSETS.items():
        path = OUT / name
        details.update(sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix == ".png":
            with Image.open(path) as image:
                details.update(pixel_size=list(image.size), mode=image.mode)
                image.verify()
    manifest = {
        "source_pdf": relative(source_pdf),
        "new_model_inference": False,
        "network_namespace_isolated": True,
        "geometry_reconstructed": True,
        "historical_fusion_snapshots": False,
        "overlapping_placeholders": "Multiple alternatives requested, not a fixed image count",
        "image_request_groups": 20,
        "text_request_groups": 5,
        "parameters": {
            k: CFG[k]
            for k in (
                "downsample_voxel_size",
                "dbscan_eps",
                "dbscan_min_points",
                "obj_pcd_max_points",
            )
        },
        "camera_parameters": CAM,
        "packs": reports,
        "assets": ASSETS,
        "source_files_sha256": SOURCES,
        "protected_outputs_sha256": protected_before,
        "original_outputs_unchanged": True,
    }
    write_json(OUT / "manifest.json", manifest)
    print("All assets complete:", len(ASSETS), "registered files.", flush=True)
