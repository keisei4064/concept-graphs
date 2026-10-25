"""Offline entry point: python -m conceptgraph.local_only [Hydra overrides]."""
import os
from pathlib import Path
import sys

from conceptgraph.utils.local_runtime import install_local_network_guard


def main():
    install_local_network_guard(os.environ.get(
        "CONCEPTGRAPH_NETWORK_LOG", "outputs/local-only/network.jsonl"))
    # Prevent optional Ultralytics analytics without touching global settings.
    config = Path(os.environ["YOLO_CONFIG_DIR"])
    config.mkdir(parents=True, exist_ok=True)
    from ultralytics import settings
    settings.update({"sync": False})
    from conceptgraph.slam.rerun_realtime_mapping import main as mapping_main
    if not any(arg.startswith("--config-name") for arg in sys.argv[1:]):
        sys.argv.insert(1, "--config-name=rerun_local_replica")
    mapping_main()


if __name__ == "__main__":
    main()
