"""Assemble page 15 assets, including a ready-to-paste tilted image plane."""
# Install the offline guard before loading third-party libraries.
# ruff: noqa: E402

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "p15_projection"
LOG = REPO / "outputs/local-only/presentation_review/20261027/asset_build"
namespace = os.readlink("/proc/self/ns/net")
if os.environ.get("CG_P15_HOST_NETNS", namespace) == namespace:
    env = dict(os.environ, CG_P15_HOST_NETNS=namespace)
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
        env=env,
        check=True,
    )
    sys.exit(0)
sys.path.insert(0, str(REPO))
os.environ["MPLCONFIGDIR"] = str(LOG / "mpl-config")
from conceptgraph.utils.local_runtime import install_local_network_guard

install_local_network_guard(LOG / "p15-network.jsonl")
import numpy as np
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def white(image):
    im = image.convert("RGBA")
    out = Image.new("RGB", im.size, "white")
    out.paste(im, mask=im.getchannel("A"))
    return out


def tilted_plane(image, path):
    # Fit the actual crop on a projective image plane; no image content is synthesized.
    im = image.convert("RGBA")
    texture = Image.new("RGBA", im.size, (0, 0, 0, 255))
    texture.alpha_composite(im)
    w, h = texture.size
    corners = np.array([[130, 110], [590, 35], [500, 825], [45, 675]], float)
    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    matrix = []
    target = []
    for (x, y), (u, v) in zip(corners, source):
        matrix += [
            [x, y, 1, 0, 0, 0, -u * x, -u * y],
            [0, 0, 0, x, y, 1, -v * x, -v * y],
        ]
        target += [u, v]
    coeff = np.linalg.solve(np.array(matrix), np.array(target))
    plane = texture.transform(
        (650, 860),
        Image.Transform.PERSPECTIVE,
        coeff,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(plane)
    draw.line(
        [tuple(x) for x in np.vstack([corners, corners[:1]])], fill="#263b50", width=4
    )
    plane.save(path)
    white(plane).save(path.with_name(path.stem + "_white.png"))


def put_image(ax, path, x, y, width, height):
    with Image.open(path) as image:
        im = image.convert("RGBA").copy()
    scale = min(width / im.width, height / im.height)
    actual_width, actual_height = im.width * scale, im.height * scale
    x += (width - actual_width) / 2
    y += (height - actual_height) / 2
    ax.imshow(
        im, extent=(x, x + actual_width, y, y + actual_height), aspect="auto", zorder=3
    )


def save_diagram(folder, depth=False):
    stem = "04_projection_overview_depth" if depth else "04_projection_overview"
    fig, ax = plt.subplots(figsize=(18, 4.8), dpi=180)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    # Camera/map coordinate axes are a schematic, independent of data poses.
    for origin, length, prefix in [
        ((0.7, 0.75), 0.65, "map"),
        ((3.0, 2.0), 0.45, "cam"),
    ]:
        x, y = origin
        for dx, dy, color, label in [
            (length, 0, "#e75a22", "x"),
            (0, length, "#249d69", "y"),
            (-0.55 * length, -0.5 * length, "#2777d3", "z"),
        ]:
            ax.annotate(
                "",
                (x + dx, y + dy),
                (x, y),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
            )
        ax.text(x - 0.15, y - 0.55, prefix, fontsize=12)
    ax.add_patch(
        Polygon(
            [[2.1, 2.15], [3.5, 2.15], [3.5, 3.15], [2.1, 3.15]],
            facecolor="#e8eef4",
            edgecolor="#334c62",
            lw=1.7,
        )
    )
    ax.add_patch(
        Polygon(
            [[3.5, 2.3], [4.0, 2.1], [4.0, 3.2], [3.5, 3.0]],
            facecolor="#d4e0e9",
            edgecolor="#334c62",
            lw=1.5,
        )
    )
    ax.annotate(
        "",
        (2.5, 2.15),
        (0.85, 0.85),
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#777777"},
    )
    ax.text(1.2, 1.6, r"$R_t,\ t_t$", fontsize=17)
    put_image(
        ax,
        folder / ("01_masked_depth_plane.png" if depth else "01_masked_plane.png"),
        4.5,
        1.0,
        2.1,
        2.8,
    )
    ax.annotate("", (7.25, 2.6), (6.7, 2.6), arrowprops={"arrowstyle": "->", "lw": 2})
    put_image(ax, folder / "02_projected_with_noise.png", 7.5, 1.0, 3.5, 2.8)
    ax.annotate("", (11.95, 2.6), (11.2, 2.6), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(11.05, 3.1, "DBSCAN", fontsize=11)
    put_image(ax, folder / "03_denoised.png", 12.3, 1.0, 3.5, 2.8)
    ax.text(
        5.55,
        4.0,
        "Masked depth on image plane" if depth else "Masked RGB on image plane",
        ha="center",
        fontsize=13,
    )
    ax.text(9.25, 4.0, "Projected points", ha="center", fontsize=13)
    ax.text(14.05, 4.0, "Retained points", ha="center", fontsize=13)
    ax.text(
        9.25, 0.6, "Gray: discarded points", ha="center", fontsize=12, color="#686e76"
    )
    ax.text(14.05, 0.6, r"$\mathbf{p}_{t,i}$", ha="center", fontsize=18)
    put_image(ax, folder / "05_clip_feature.png", 16.3, 1.0, 1.2, 2.8)
    ax.text(16.9, 0.6, r"$\mathbf{f}_{t,i}$", ha="center", fontsize=18)
    ax.text(16.9, 4.0, "RGB feature", ha="center", fontsize=11)
    ax.text(
        0.3,
        -0.40,
        "Schematic camera / layout; point clouds reconstructed from the same recorded observation.",
        fontsize=10,
        color="#566575",
    )
    fig.savefig(
        folder / f"{stem}.png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        folder / f"{stem}.svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    with Image.open(folder / f"{stem}.png") as im:
        white(im).save(folder / f"{stem}_white.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    manifest = {}
    source_hashes = {}
    recorded_root = REPO / "outputs/local-only/paper_scenegraph_room0/attempt1"
    observations = [
        json.loads(s)
        for s in (recorded_root / "observations.jsonl").read_text().splitlines()
    ]
    data_root = REPO / "outputs/local-only/assets/data/Replica/room0"
    trajectory = np.loadtxt(data_root / "traj.txt").astype(np.float32).reshape(-1, 4, 4)
    camera = json.loads((HERE / "manifest.json").read_text())["camera_parameters"]
    for name in ("A_chair", "B_sofa", "C_planter"):
        src = HERE / name
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        provenance = json.loads((src / "provenance.json").read_text())
        frame_number = int(
            Path(provenance["incoming_source_frame"]).stem.removeprefix("frame")
        )
        depth_source = data_root / "results" / f"depth{frame_number:06d}.png"
        depth_hash = sha(depth_source)
        depth_raw = np.array(Image.open(depth_source)).astype(np.uint16)
        mask = np.array(Image.open(src / "mask_01_binary.png")) > 0
        depth_m = depth_raw.astype(float) / camera["png_depth_scale"]
        valid = depth_m > 0
        maximum = float(depth_m[valid].max())
        colored = (
            plt.get_cmap("turbo")(np.clip(depth_m / maximum, 0, 1)) * 255
        ).astype(np.uint8)
        colored[..., 3] = valid.astype(np.uint8) * 255
        shutil.copy2(depth_source, dest / "07_depth_raw.png")
        Image.fromarray(colored).save(dest / "07_depth_preview.png")
        Image.fromarray(np.where(mask, depth_raw, 0).astype(np.uint16)).save(
            dest / "08_depth_masked_raw.png"
        )
        colored[..., 3] = (valid & mask).astype(np.uint8) * 255
        Image.fromarray(colored).save(dest / "08_depth_masked_preview.png")
        obs = next(
            r
            for r in observations
            if r["map_index"] == provenance["map_index"]
            and r.get("source_frame") == provenance["incoming_source_frame"]
            and r["status"] == "ready"
        )
        crop = Image.fromarray(colored).crop(obs["crop_xyxy"])
        crop.save(dest / "08_depth_masked_crop.png")
        tilted_plane(crop, dest / "01_masked_depth_plane.png")
        pose_info = {
            "source_frame": provenance["incoming_source_frame"],
            "R_t": trajectory[frame_number, :3, :3].tolist(),
            "t_t": trajectory[frame_number, :3, 3].tolist(),
            "K": [
                [camera["fx"], 0, camera["cx"]],
                [0, camera["fy"], camera["cy"]],
                [0, 0, 1],
            ],
            "png_depth_scale": camera["png_depth_scale"],
            "depth_unit": "meters after division by png_depth_scale",
            "depth_preview_range_m": [0, maximum],
            "crop_xyxy": obs["crop_xyxy"],
            "map_pose": "Untransformed dataset camera-to-world pose, as used by the mapping pipeline",
        }
        (dest / "06_camera_to_map.json").write_text(
            json.dumps(pose_info, indent=2) + "\n"
        )
        assert sha(depth_source) == depth_hash
        for source_name, target_name in [
            ("masked_rgb_crop.png", "00_masked_rgb_flat.png"),
            ("projection_before_front.png", "02_projected_with_noise.png"),
            ("projection_after_front.png", "03_denoised.png"),
            ("projection_rgb_front.png", "03_denoised_rgb.png"),
            ("projection_before_oblique.png", "02_projected_with_noise_oblique.png"),
            ("projection_after_oblique.png", "03_denoised_oblique.png"),
        ]:
            source = src / source_name
            source_hashes[str(source.relative_to(HERE))] = sha(source)
            shutil.copy2(source, dest / target_name)
            with Image.open(dest / target_name) as im:
                white(im).save(dest / (Path(target_name).stem + "_white.png"))
        with Image.open(src / "masked_rgb_crop.png") as image:
            tilted_plane(image, dest / "01_masked_plane.png")
        with (
            Image.open(dest / "02_projected_with_noise.png") as a,
            Image.open(dest / "03_denoised.png") as b,
        ):
            assert a.size == b.size
        feature_cache = (
            REPO
            / "outputs/local-only/assets/data/Replica/room0/exps/local_room0_full_v1/detections"
            / Path(provenance["incoming_source_frame"]).stem
            / "image_feats.npz"
        )
        feature_hash = sha(feature_cache)
        raw_index = provenance["segmentation_outputs"][0]["raw_detection_index"]
        with np.load(feature_cache) as saved:
            feature = saved[saved.files[0]][raw_index].copy()
        np.save(dest / "05_clip_feature.npy", feature)
        fig, ax = plt.subplots(figsize=(1.2, 6), dpi=200)
        ax.imshow(
            feature[:, None],
            cmap="YlGn",
            aspect="auto",
            vmin=-max(abs(feature)),
            vmax=max(abs(feature)),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(
            dest / "05_clip_feature.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
        assert sha(feature_cache) == feature_hash
        save_diagram(dest)
        save_diagram(dest, depth=True)
        manifest[name] = {
            "source_frame": provenance["incoming_source_frame"],
            "map_id": provenance["map_id"],
            "removed_points": provenance["projection"]["removed_points"],
            "masked_image": "3D input mask after containment subtraction",
            "camera_icon_and_axes": "schematic, not measured camera geometry",
            "cloud_geometry": "reconstructed from the recorded RGB-D and original processing",
            "source_sha256": {
                k: v for k, v in source_hashes.items() if k.startswith(name + "/")
            },
            "clip_feature": {
                "source": str(feature_cache.relative_to(REPO)),
                "sha256": feature_hash,
                "raw_detection_index": raw_index,
                "dimension": len(feature),
            },
            "depth": {
                "source": str(depth_source.relative_to(REPO)),
                "sha256": depth_hash,
                "preview_range_m": [0, maximum],
                "mask_source": f"{name}/mask_01_binary.png",
            },
        }
    for path, digest in source_hashes.items():
        assert sha(HERE / path) == digest
    files = {
        str(p.relative_to(OUT)): sha(p)
        for p in OUT.rglob("*")
        if p.suffix in (".png", ".svg", ".npy", ".json")
        and p.name not in ("manifest.json", "validation.json")
    }
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "packs": manifest,
                "files_sha256": files,
                "source_files_unchanged": True,
                "new_inference": False,
            },
            indent=2,
        )
        + "\n"
    )
    pages = [
        white(Image.open(OUT / name / f"{stem}.png"))
        for name in manifest
        for stem in ("04_projection_overview", "04_projection_overview_depth")
    ]
    pages[0].save(
        OUT / "p15候補一覧.pdf", save_all=True, append_images=pages[1:], resolution=160
    )
    print("p15 assets ready:", len(files), "images/SVGs; 3 candidate packs.")
