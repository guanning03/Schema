#!/bin/bash
#SBATCH -J mazebench
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
set -uo pipefail
RUN=$(realpath -m "$1"); DEADLINE=$2; shift 2; EXTRA=("$@")
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SELF="$REPO/run.sbatch"
FAILS="${RUN%/}.chain_short_exits"
log() { echo "[chain $(date '+%m-%d %T')] $*"; }

left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) ))
if [ "$left" -lt 1800 ]; then log "less than 30 minutes to the deadline $DEADLINE; not continuing"; exit 0; fi

NEXT=$(cd "$REPO" && sbatch --parsable -J "$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
       "$SELF" "$RUN" "$DEADLINE" "${EXTRA[@]}")
log "node=$(hostname) run=$RUN next leg queued: $NEXT"

MH=$(python3 -c "print(round(min(69.0, ($left - 1200) / 3600), 2))")
log "this leg: --max-hours $MH (deadline $DEADLINE)"

cd "$REPO" || exit 1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-arc3}"
export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

if [ -s "$RUN/events.jsonl" ]; then MODE=(--resume "$RUN"); else MODE=(--workdir "$RUN"); fi
log "solve ${MODE[*]}"
t0=$(date +%s)
python -m agent.world_model.solve --max-hours "$MH" --service-tier ultrafast "${MODE[@]}" "${EXTRA[@]}"
rc=$?
ran=$(( $(date +%s) - t0 ))
log "solve exited rc=$rc after $((ran / 60)) min"
python3 -c "
p = '$RUN/events.jsonl'; d = open(p, 'rb').read()
if d and not d.endswith(b'\n'): open(p, 'ab').write(b'\n')
"

if [ "$ran" -lt 1200 ]; then
  n=$(( $(cat "$FAILS" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAILS"
  if [ "$n" -ge 3 ]; then scancel "$NEXT"; log "$n legs in a row exited within 20 minutes; cancelled $NEXT, chain stopped"; exit 1; fi
else
  echo 0 > "$FAILS"
fi
if [ $(( $(date -d "$DEADLINE" +%s) - $(date +%s) )) -lt 1800 ]; then
  scancel "$NEXT"; log "deadline $DEADLINE reached; cancelled $NEXT, chain finished"
fi
