#!/bin/bash
#SBATCH -J arc3
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 3-00:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-arc3}"

export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export ARC_SANDBOX_PODMAN_ROOT="$TMPDIR/arc-podman"
export XDG_RUNTIME_DIR="$TMPDIR/arc-podman/run"
mkdir -p "$XDG_RUNTIME_DIR"
export ARC_SANDBOX_CACHE="${ARC_SANDBOX_CACHE:-$HOME/.cache/arc-sandbox}"
mkdir -p "$ARC_SANDBOX_CACHE"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

python -m agent.world_model.solve --provider claude --jail "$@"
