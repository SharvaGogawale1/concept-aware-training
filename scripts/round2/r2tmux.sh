#!/usr/bin/env bash
# Start one round-2 Python command in a detached tmux session, with the round-2 environment
# and a log at results/round2/logs/<session>.log that ends with the exit code.
# A tmux server that is already running does not pass this shell's exports to new sessions,
# so the environment is set here rather than inherited.
#
#   bash scripts/round2/r2tmux.sh r2holdout scripts/round2/build_sealed_holdout.py
#   bash scripts/round2/r2tmux.sh r2qA scripts/round2/run_round2.py train --tier 1 --queue A --gpu 2 --approved
set -euo pipefail
SELF=$(readlink -f "$0")
if [ "${1:-}" = "--inner" ]; then
  shift; LOG=$1; shift
  set +e
  echo "[$(date '+%F %T')] start: $*" >> "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] finished with exit code $code" | tee -a "$LOG"
  exit "$code"
fi
[ $# -ge 2 ] || { echo "usage: r2tmux.sh <session-name> <script.py> [args...]"; exit 2; }
NAME=$1; shift
BASE=${CONCEPT_BASE:-$HOME/fos_retrieval/task15}
REPO=$BASE/concept_aware/concept-aware-training
PY=${PY:-$HOME/miniconda3/envs/concept/bin/python}
# concept_aware/hf_cache is the cache's path before the NAS move and a link to it after.
HFH=${HF_HOME:-$BASE/concept_aware/hf_cache}
LOG=$REPO/results/round2/logs/$NAME.log
mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "=$NAME" 2>/dev/null; then
  echo "a tmux session named $NAME is already running (tmux attach -t $NAME)"; exit 1
fi
printf -v INNER '%q ' bash "$SELF" --inner "$LOG" "$PY" "$@"
tmux new-session -d -s "$NAME" -c "$REPO" \
  "env CONCEPT_BASE=$BASE HF_HOME=$HFH PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 $INNER"
echo "started tmux session '$NAME'  log: $LOG"
