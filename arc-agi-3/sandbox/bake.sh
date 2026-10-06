#!/usr/bin/env bash
set -euo pipefail

PY_IMAGE="docker://python:3.12-slim-bookworm"
NODE_VERSION="22.22.2"
CLAUDE_CODE_VERSION="2.1.216"
PIP_PKGS="arc-agi==0.9.8 arcengine==0.9.3 numpy scipy pydantic rich httpx"

CACHE="${ARC_SANDBOX_CACHE:-$HOME/.cache/arc-sandbox}"
self_hash=$(sha256sum "${BASH_SOURCE[0]}" | cut -c1-12)
OUT="$CACHE/arc-agent-$self_hash.sqsh"

if [[ -f "$OUT" && "${ARC_SANDBOX_REBUILD:-}" != "1" ]]; then
    echo "bake: image already cached: $OUT" >&2
    echo "$OUT"
    exit 0
fi

WORK="${TMPDIR:-/tmp}/arc-bake-$$"
export ENROOT_DATA_PATH="$WORK/data" ENROOT_CACHE_PATH="$WORK/cache" \
       ENROOT_RUNTIME_PATH="$WORK/run" ENROOT_TEMP_PATH="$WORK/tmp"
mkdir -p "$WORK"/{data,cache,run,tmp} "$CACHE"
trap 'rm -rf "$WORK"' EXIT

echo "bake: importing $PY_IMAGE …" >&2
enroot import -o "$WORK/base.sqsh" "$PY_IMAGE" >&2
enroot create -n arc-bake-$$ "$WORK/base.sqsh" >&2

echo "bake: installing toolchain inside image …" >&2
enroot start --root --rw arc-bake-$$ bash -eu <<EOS >&2
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends git ca-certificates curl xz-utils procps
curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
  | tar -xJ --no-same-owner -C /usr/local --strip-components=1
npm install -g --no-fund --no-audit "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
pip install --no-cache-dir ${PIP_PKGS}
apt-get clean && rm -rf /var/lib/apt/lists/* /root/.npm /tmp/*
node --version && claude --version && python3 -c "import arc_agi, arcengine, numpy, scipy; print('py deps ok')"
EOS

echo "bake: exporting image …" >&2
enroot export -o "$OUT.tmp.$$" arc-bake-$$ >&2
mv "$OUT.tmp.$$" "$OUT"
enroot remove -f arc-bake-$$ >&2 || true
echo "bake: done: $OUT" >&2
echo "$OUT"
