#!/usr/bin/env bash
# Foreground temporary server. Run in a separate terminal; Ctrl-C ends only it.
set -euo pipefail
run_dir=${1:?Usage: CONCEPTGRAPH_OLLAMA_MODELS=existing/models bash scripts/local_only/serve_ollama.sh RUN_DIR}
model_root=${CONCEPTGRAPH_OLLAMA_MODELS:?Point to existing readable models; do not download them again}
test -d "$model_root/manifests"
test -d "$model_root/blobs"
mkdir -p "$run_dir"
# Refuse a busy port rather than altering/stopping its owner.
python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 11435))
PY
echo "Temporary cloud-disabled Ollama at 127.0.0.1:11435; trace launcher PID $$"
echo "$$" > "$run_dir/trace.pid"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=$model_root
export OLLAMA_NO_CLOUD=1
export OLLAMA_NOPRUNE=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=${CONCEPTGRAPH_OLLAMA_CONTEXT_LENGTH:-8192}
export OLLAMA_KEEP_ALIVE=2m
exec strace -f -e trace=network -s 160 -o "$run_dir/network.strace" \
    ollama serve > "$run_dir/ollama.log" 2>&1
