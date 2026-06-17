#!/usr/bin/env bash
# fd-leak-check.sh — measure file-descriptor growth across RTSP connect/disconnect
# cycles, to verify the narrow concern from upstream PR #400: whether neolink
# leaks UNIX-STREAM socket fds on per-client GStreamer pipeline teardown.
#
# Context: our audit found we are NOT affected by #400's headline claims
#   - separate Baichuan session per client  -> we share one BcCamera per camera
#   - buffer-pool fd/mem leak (#373/#340)    -> already fixed via bucketed pools
# The only theoretically-open item is a gst-rtsp-server pipeline teardown/rebuild
# fd leak under set_shared(false) + SuspendMode::Reset. It only manifests over
# many connect/disconnect cycles, so MEASURE before changing anything (and do NOT
# blindly port #400 — set_shared(true) is incompatible with our per-client model).
#
# Run this on the host/container where neolink actually runs (the HA add-on
# container, or a local `cargo run -- rtsp --config ...`). A flat fd count across
# cycles = no leak; a steady ~1-2 fds/cycle climb = a real teardown leak to fix
# at the unref/teardown path (keeping the per-client design).
#
# Usage:
#   scripts/fd-leak-check.sh <rtsp-url> [cycles] [hold-secs] [neolink-pid]
# Example:
#   scripts/fd-leak-check.sh rtsp://127.0.0.1:8558/front/sub 50 3
#
# Requires: ffprobe (ffmpeg). fd count via /proc on Linux, lsof on macOS.
set -uo pipefail

URL="${1:?usage: fd-leak-check.sh <rtsp-url> [cycles] [hold-secs] [neolink-pid]}"
CYCLES="${2:-50}"
HOLD="${3:-3}"
PID="${4:-$(pgrep -n -x neolink 2>/dev/null || pgrep -n neolink 2>/dev/null || true)}"

[ -n "${PID:-}" ] || { echo "ERROR: could not find a neolink pid; pass it as arg 4." >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe (ffmpeg) is required." >&2; exit 1; }

fdcount() {
  if [ -d "/proc/$PID/fd" ]; then
    ls "/proc/$PID/fd" 2>/dev/null | wc -l | tr -d ' '
  else
    lsof -p "$PID" 2>/dev/null | tail -n +2 | wc -l | tr -d ' '
  fi
}

echo "neolink pid=$PID  url=$URL  cycles=$CYCLES  hold=${HOLD}s"
base="$(fdcount)"
echo "cycle    fd     d_base  d_step"
printf "%-8s %-6s %-7s %s\n" "start" "$base" "0" "0"
prev="$base"
for i in $(seq 1 "$CYCLES"); do
  # one full connect -> hold -> disconnect cycle (builds + tears down one pipeline)
  ffprobe -rtsp_transport tcp -i "$URL" -v error -show_entries format=duration >/dev/null 2>&1 &
  fp=$!
  sleep "$HOLD"
  kill "$fp" 2>/dev/null
  wait "$fp" 2>/dev/null
  sleep 1   # allow neolink to tear the per-client pipeline down
  c="$(fdcount)"
  printf "%-8s %-6s %-7s %s\n" "$i" "$c" "$((c - base))" "$((c - prev))"
  prev="$c"
done
final="$(fdcount)"
growth="$((final - base))"
echo "----"
echo "base=$base final=$final growth=$growth over $CYCLES cycles"
if [ "$growth" -gt "$((CYCLES / 5))" ]; then
  echo "VERDICT: fd count climbed (~>=0.2/cycle) -> likely a pipeline-teardown leak."
  echo "         Fix at the teardown path (fully unref/remove media bin + appsrc on client-gone),"
  echo "         keeping the per-client model. Do NOT adopt #400's set_shared(true) rewrite."
else
  echo "VERDICT: fd count stable -> no per-cycle teardown leak observed; #400 does not apply."
fi
