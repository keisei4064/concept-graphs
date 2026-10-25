"""Run inside a newly created dedicated Conda environment; writes real evidence."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not Path(prefix).name.startswith("conceptgraph-local") or Path(sys.prefix) != Path(prefix):
        raise RuntimeError("Activate the dedicated conceptgraph-local Conda environment first")
    report = {"python": platform.python_version(), "cuda_home": os.getenv("CUDA_HOME"),
              "packages": {}, "commands": {}}
    for command in (["git", "rev-parse", "HEAD"], ["nvidia-smi"], ["nvcc", "--version"],
                    ["conda", "--version"], ["ollama", "--version"], ["ollama", "list"]):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        report["commands"][" ".join(command)] = result.stdout.strip()
    for module, distribution in {
        "torch": "torch", "torchvision": "torchvision", "pytorch3d.ops": "pytorch3d",
        "ultralytics": "ultralytics", "open_clip": "open_clip_torch", "open3d": "open3d",
        "faiss": "faiss-cpu", "rerun": "rerun-sdk", "hydra": "hydra-core",
        "supervision": "supervision", "kornia": "kornia", "openai": "openai",
        "conceptgraph.slam.rerun_realtime_mapping": "conceptgraph",
    }.items():
        importlib.import_module(module)
        report["packages"][distribution] = importlib.metadata.version(distribution)
    import torch
    import numpy as np
    import open3d as o3d
    from pytorch3d.ops import box3d_overlap
    assert torch.cuda.is_available(), "CUDA unavailable"
    report["torch_cuda"] = torch.version.cuda
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))
    a = torch.arange(256 * 256, device="cuda", dtype=torch.float32).reshape(256, 256) / 65536
    product = a @ a.T
    torch.cuda.synchronize()
    assert product.isfinite().all()
    report["gpu_matmul_checksum"] = product.sum().item()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.eye(3)))
    assert len(pcd.points) == 3
    box = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
    for device in ("cpu", "cuda"):
        volumes, iou = box3d_overlap(box[None].to(device), box[None].to(device))
        assert torch.allclose(iou, torch.ones_like(iou))
        report[f"pytorch3d_{device}_iou"] = iou.item()
    # Exercise the actual ali-dev reorder/expansion wrapper too.
    from conceptgraph.utils.ious import compute_3d_iou_accurate_batch
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.zeros(3), np.ones(3))
    corners = torch.tensor(np.asarray(bbox.get_box_points()), device="cuda").unsqueeze(0)
    assert torch.allclose(compute_3d_iou_accurate_batch(corners, corners),
                          torch.ones((1, 1), device="cuda"))
    report["status"] = "PASS"
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/local-only/m0/smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "commands"}, indent=2))


if __name__ == "__main__":
    main()
