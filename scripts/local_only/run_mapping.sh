#!/usr/bin/env bash
# Run from the repository root. Each run gets syscall and Python network logs.
set -euo pipefail
run_dir=${1:?Usage: bash scripts/local_only/run_mapping.sh RUN_DIR [Hydra overrides...]}
shift
mkdir -p "$run_dir"
run_dir=$(realpath "$run_dir")
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export CONCEPTGRAPH_REPO_ROOT=$PWD
export CONCEPTGRAPH_DATA_ROOT=${CONCEPTGRAPH_DATA_ROOT:-$PWD/outputs/local-only/assets/data}
export CONCEPTGRAPH_MODEL_ROOT=${CONCEPTGRAPH_MODEL_ROOT:-$PWD/outputs/local-only/assets/weights}
export CONCEPTGRAPH_CLIP_CHECKPOINT=${CONCEPTGRAPH_CLIP_CHECKPOINT:-$(cat outputs/local-only/assets/clip_checkpoint.txt)}
export CONCEPTGRAPH_NETWORK_LOG=$run_dir/network.jsonl
export YOLO_CONFIG_DIR=$run_dir/yolo-config
export RERUN_ANALYTICS_ENABLED=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
unset OPENAI_API_KEY ANTHROPIC_API_KEY
date --iso-8601=seconds > "$run_dir/started.txt"
# User requested no Git operations. Capture the executed sources directly instead.
python3 - "$run_dir/source-state.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
paths = list(Path('conceptgraph').rglob('*.py')) + list(Path('conceptgraph/hydra_configs').rglob('*.yaml'))
paths += list(Path('scripts/local_only').glob('*.py')) + list(Path('scripts/local_only').glob('*.sh'))
Path(sys.argv[1]).write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(paths)}, indent=2))
PY
nvidia-smi > "$run_dir/nvidia-before.txt"
printf '%s\n' "$@" > "$run_dir/overrides.txt"
echo "Starting mapping; logs: $run_dir; launcher PID: $$"
conda run --no-capture-output -n "${CONCEPTGRAPH_CONDA_ENV:-conceptgraph-local}" \
    strace -f -e trace=network -s 160 -o "$run_dir/network.strace" \
    python -m conceptgraph.local_only "$@" > "$run_dir/mapping.log" 2>&1 &
mapping_pid=$!
echo "$mapping_pid" > "$run_dir/launcher.pid"
wait "$mapping_pid"
conda run --no-capture-output -n "${CONCEPTGRAPH_CONDA_ENV:-conceptgraph-local}" \
    python scripts/local_only/audit_network.py "$run_dir/network.strace" > "$run_dir/network-audit.json"
nvidia-smi > "$run_dir/nvidia-after.txt"
echo "Mapping and syscall network audit passed: $run_dir"
