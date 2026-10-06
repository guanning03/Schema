#!/usr/bin/env bash
set -u
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <reasoning> <game> [game ...]" >&2
  exit 2
fi
REASONING=$1; shift
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs"
PIDS=()
LABELS=()
for G in "$@"; do
  ( cd "$ROOT/schema" && "$PY" -m agent.world_model.solve --game "$G" --reasoning "$REASONING" \
      > "$ROOT/logs/${G}_schema_${REASONING}_$TS.log" 2>&1 ) &
  PIDS+=("$!"); LABELS+=("$G schema")
  ( cd "$ROOT/basic_harness" && "$PY" run_codex.py "$G" --effort "$REASONING" --run-dir runs \
      > "$ROOT/logs/${G}_basic_${REASONING}_$TS.log" 2>&1 ) &
  PIDS+=("$!"); LABELS+=("$G basic")
  sleep 3
done
FAILED=0
for I in "${!PIDS[@]}"; do
  if wait "${PIDS[$I]}"; then
    continue
  else
    RC=$?
    echo "${LABELS[$I]} failed (exit $RC); see $ROOT/logs/" >&2
    FAILED=1
  fi
done
exit "$FAILED"
