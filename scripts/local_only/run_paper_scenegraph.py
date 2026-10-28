"""Explicit stages for the public-main scenegraph comparison, without mapping."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from conceptgraph.utils.local_runtime import install_local_network_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["prepare", "infer", "render", "render-candidates", "verify"]
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/local-only/paper_scenegraph_room0/attempt1"),
    )
    p.add_argument(
        "--map",
        type=Path,
        default=Path(
            "outputs/local-only/assets/data/Replica/room0/exps/room0_reference_graph_attempt1/pcd_room0_reference_graph_attempt1.pkl.gz"
        ),
    )
    p.add_argument("--reference", type=Path)
    p.add_argument("--run", default="full_main")
    p.add_argument("--indices", type=int, nargs="+")
    p.add_argument("--max-observations", type=int, default=10)
    p.add_argument("--model", default="qwen3.5:35b")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("render", "render-candidates"):
        # Python audit hooks cannot intercept Rust SDK telemetry. Isolate native
        # rendering too, using a task-specific config instead of the user's one.
        host_ns = os.environ.get("CONCEPTGRAPH_PAPER_HOST_NETNS")
        isolated = host_ns and os.readlink("/proc/self/ns/net") != host_ns
        if not isolated:
            if not shutil.which("bwrap"):
                raise RuntimeError(
                    "Rendering requires bwrap for native network isolation"
                )
            config = args.root.resolve() / "logs/gui-config"
            config.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                XDG_CONFIG_HOME=str(config),
                RERUN_ANALYTICS_ENABLED="false",
                CONCEPTGRAPH_PAPER_HOST_NETNS=os.readlink("/proc/self/ns/net"),
            )
            subprocess.run(
                [
                    "bwrap",
                    "--unshare-net",
                    "--bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    sys.executable,
                    *sys.argv,
                ],
                env=env,
                check=True,
            )
            return
        subprocess.run(
            ["rerun", "analytics", "disable"], check=True, capture_output=True
        )
    install_local_network_guard(args.root / "logs" / "python-network.jsonl")
    from conceptgraph.scenegraph.local_paper_scenegraph import prepare, run_inference

    if args.stage == "prepare":
        result = prepare(
            args.map,
            args.root,
            args.reference or args.root / "source_refs/official_main_scenegraph.py",
            args.max_observations,
        )
        print(
            json.dumps({k: result[k] for k in ("prepared", "objects", "observations")})
        )
    elif args.stage == "infer":
        result = run_inference(
            args.root, args.run, args.model, args.indices, args.max_observations
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "retained_objects",
                        "edges",
                        "candidate_pairs",
                        "elapsed_seconds",
                        "new_requests",
                        "cache_hits",
                    )
                }
            )
        )
    elif args.stage == "render":
        from conceptgraph.scenegraph.paper_scenegraph_view import render

        render(args.root, args.run)
    elif args.stage == "render-candidates":
        from conceptgraph.scenegraph.paper_scenegraph_candidates import (
            render_candidates,
        )

        render_candidates(args.root, args.run)
    else:
        from conceptgraph.scenegraph.paper_scenegraph_validation import verify

        result = verify(args.root, args.run)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
