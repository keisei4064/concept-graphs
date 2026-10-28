#!/usr/bin/env bash
# View saved results with native network isolation; never touch global Rerun config.
set -euo pipefail
paper_root=${1:-outputs/local-only/paper_scenegraph_room0/attempt1}
run_name=${2:-full_main}
recording_name=${3:-viewer.rrd}
paper_root=$(realpath "$paper_root")
recording="$paper_root/$run_name/$recording_name"
test -f "$recording"
command -v bwrap >/dev/null
mkdir -p "$paper_root/logs/gui-config"
viewer_logs=$(mktemp -d "$paper_root/logs/viewer-XXXXXX")
export XDG_CONFIG_HOME="$paper_root/logs/gui-config"
export RERUN_ANALYTICS_ENABLED=false
export PYTHONPATH=
export PYTHONNOUSERSITE=1
printf 'Offline viewer logs: %s\n' "$viewer_logs"
exec bwrap --unshare-net --bind / / --proc /proc bash -c '
  set -euo pipefail
  rerun analytics disable
  if [[ ${CONCEPTGRAPH_PAPER_TRACE:-0} == 1 ]]; then
    exec strace -f -e trace=network -s 160 -o "$2/network.strace" \
      rerun --bind 127.0.0.1 --port 0 --threads 8 --window-size 1800x1000 "$1"
  fi
  exec rerun --bind 127.0.0.1 --port 0 --threads 8 --window-size 1800x1000 "$1"
' _ "$recording" "$viewer_logs"
