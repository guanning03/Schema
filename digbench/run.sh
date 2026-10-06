#!/usr/bin/env bash
set -u
REASONING=$1; shift
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs"
for G in "$@"; do
  ( cd "$ROOT/schema" && $PY -m agent.world_model.solve --game "$G" --reasoning "$REASONING" \
      > "$ROOT/logs/${G}_schema_${REASONING}_$TS.log" 2>&1 ) &
  ( cd "$ROOT/basic_harness" && $PY run_codex.py "$G" --effort "$REASONING" --run-dir runs \
      > "$ROOT/logs/${G}_basic_${REASONING}_$TS.log" 2>&1 ) &
  sleep 3
done
wait
