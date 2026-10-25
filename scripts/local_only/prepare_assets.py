"""Explicit ONLINE setup phase; download only missing original ali-dev assets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/local-only/assets"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from remotezip import RemoteZip
    url = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
    data = root / "data"
    required = ["Replica/room0/traj.txt"]
    required += [f"Replica/room0/results/{kind}{idx:06d}.{extension}"
                 for idx in range(args.frames) for kind, extension in (("frame", "jpg"), ("depth", "png"))]
    missing = [name for name in required if not (data / name).exists()]
    if missing:
        # HTTP ranges avoid downloading the 12.4GB multi-scene archive.
        with RemoteZip(url, timeout=90) as archive:
            for index, name in enumerate(missing):
                member = archive.getinfo(name)
                target = data / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source:
                    payload = source.read()  # zipfile also verifies the member CRC
                with target.open("xb") as stream:
                    stream.write(payload)
                if index % 20 == 0:
                    print(f"Replica: {index + 1}/{len(missing)} missing members downloaded", flush=True)
    manifest = {"replica_source": url, "frames": args.frames}
    if not args.data_only:
        import clip
        import open_clip
        from ultralytics.utils.downloads import attempt_download_asset
        weights = root / "weights"
        weights.mkdir(exist_ok=True)
        for name in ("yolov8l-world.pt", "sam_l.pt"):
            target = weights / name
            if not target.is_file():
                attempt_download_asset(str(target))
            if not target.is_file():
                raise FileNotFoundError(target)
            manifest[name] = str(target.relative_to(root))
        # YOLO-World's set_classes uses exactly this original CLIP cache.
        clip.load("ViT-B/32", device="cpu")
        checkpoint = open_clip.pretrained.download_pretrained(
            open_clip.get_pretrained_cfg("ViT-H-14", "laion2b_s32b_b79k"),
            cache_dir=str(weights / "open_clip"))
        manifest["clip_checkpoint"] = str(Path(checkpoint).relative_to(root))
        (root / "clip_checkpoint.txt").write_text(str(checkpoint) + "\n")
    manifest["sha256"] = {}
    for path in sorted(data.glob("Replica/room0/results/*")):
        manifest["sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Setup assets verified: {root}", flush=True)


if __name__ == "__main__":
    main()
