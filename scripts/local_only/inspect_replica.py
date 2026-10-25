"""Validate posed RGB-D geometry before mapping (no model inference)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from conceptgraph.dataset.datasets_common import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--end", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/local-only/m1/input.json"))
    args = parser.parse_args()
    dataset = get_dataset(dataconfig="conceptgraph/dataset/dataconfigs/replica/replica.yaml",
                          basedir=args.root, sequence="room0", start=0, end=args.end, stride=1,
                          desired_height=680, desired_width=1200, device="cpu", dtype=torch.float)
    assert len(dataset) == args.end
    frames = []
    for index in range(len(dataset)):
        rgb, depth, intrinsics, _ = dataset[index]
        raw = cv2.imread(dataset.depth_paths[index], cv2.IMREAD_UNCHANGED)
        assert tuple(rgb.shape[:2]) == tuple(depth.shape[:2]) == raw.shape == (680, 1200)
        assert Path(dataset.color_paths[index]).stem[5:] == Path(dataset.depth_paths[index]).stem[5:]
        z = depth[..., 0].numpy()
        assert np.isfinite(z).all() and (z > 0).any()
        assert np.allclose(z, raw / 6553.5, atol=1e-6)
        pose = dataset.poses[index].numpy()
        assert np.isfinite(pose).all() and np.allclose(pose[3], [0, 0, 0, 1])
        assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
        assert np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
        k = intrinsics[:3, :3].numpy()
        assert np.allclose(k, [[600, 0, 599.5], [0, 600, 339.5], [0, 0, 1]])
        y, x = np.argwhere(z > 0)[len(np.argwhere(z > 0)) // 2]
        camera_point = np.linalg.inv(k) @ np.array([x, y, 1]) * z[y, x]
        world = pose @ np.r_[camera_point, 1]
        recovered = np.linalg.inv(pose) @ world
        projected = k @ recovered[:3]
        assert np.allclose(projected[:2] / projected[2], [x, y], atol=1e-3)
        frames.append({"frame": index, "depth_m_range": [float(z[z > 0].min()), float(z.max())],
                       "camera_to_world": pose.tolist(), "intrinsics": k.tolist(),
                       "roundtrip_pixel": [int(x), int(y)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "frames": frames,
        "conventions": "NICE-SLAM c2w; +x right,+y down,+z forward; raw depth / 6553.5 m; original dataset.poses used by ali-dev"}, indent=2))
    print(f"PASS: {len(frames)} aligned RGB-D/c2w frames, depth scale and projection roundtrip")


if __name__ == "__main__":
    main()
