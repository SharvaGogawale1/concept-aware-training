#!/usr/bin/env bash
# Read-only snapshot of every running job: GPUs, disk, Falcon, the downstream fine-tunes
# and the Iyer replication.  Safe to run any time; it only reads logs and result files.
#
#   cd ~/fos_retrieval/task15 && bash concept_aware/concept-aware-training/scripts/status.sh
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE" || exit 1
PY=${CONCEPT_PY:-$HOME/miniconda3/envs/concept/bin/python}
S=concept_aware/concept-aware-training/scripts
T=falcon3-1b-base
hr() { printf '\n== %s\n' "$*"; }
last() { [ -f "$1" ] && tr '\r' '\n' < "$1" | grep -v '^\s*$' | tail -"${2:-2}"; }

hr "GPUs (index, memory used, utilisation) and disk"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
df -h "$BASE" | tail -1

hr "Our processes (pid, running time, command)"
ps -u "$USER" -o pid=,etime=,args= | grep -E "papermill|run_new_family|eval_seqcls|eval_iyer|run_clm|train\.py|embedding_synonyms|get_content_words|lm_eval|eval_swords|eval_mteb|eval_classification|eval_word" \
  | grep -v grep | sed -E 's#/home/[^ ]*/(python[0-9.]*|bin/python)#python#' | cut -c1-160

hr "tmux sessions: busy ones with their last line, then the idle ones (finished or never used)"
if command -v tmux >/dev/null && tmux ls >/dev/null 2>&1; then
  idle=""
  while read -r name cmd; do
    case $cmd in
      bash|zsh|sh|"") idle+="$name " ;;
      *) line=$(tmux capture-pane -pt "$name" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-140)
         printf '  %-12s %s\n' "$name" "$line" ;;
    esac
  done < <(tmux list-panes -a -F '#{session_name} #{pane_current_command}' | sort -u -k1,1)
  echo "  idle: $idle"
fi

hr "Falcon ($T)"
[ -e family_$T.failed ] && echo "!! FAILED marker present: family_$T.failed -- read the gpuA/gpuB logs"
last family_$T.log 2
for f in family_${T}_gpuA.log family_${T}_gpuB.log; do [ -f "$f" ] && { echo "-- $f"; last "$f" 2; }; done
python3 - "$T" <<'PY'
import json, sys
from pathlib import Path
m = Path("outputs") / sys.argv[1] / "run_manifests"
def load(name):
    p = m / f"{name}.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except json.JSONDecodeError:
        return {}
print(f"extraction shards done: {len(load('task15_shards'))} of 8")
for name in ("task15", "task15b"):
    runs = load(name)
    done = [k for k, v in runs.items() if (Path(v) / "adapter_config.json").is_file()]
    print(f"{name}: {len(done)} trained" + (f"  ({', '.join(sorted(done))})" if done else ""))
PY
current=$(ls -t verified_${T}_*.log 2>/dev/null | head -1)
[ -n "$current" ] && { echo "-- newest pass log: $current"; last "$current" 3 | cut -c1-200; }

hr "Downstream fine-tunes (finished task x arm runs; 7 systems x 13 tasks = 91 when complete)"
for f in outputs/{llama-3.2-1b,qwen3-1.7b-base,$T}/{seqcls,iyer}_finetune{,_replica}.json; do
  [ -f "$f" ] || continue
  line=$("$PY" $S/summarize_finetune.py "$f" 2>/dev/null | grep "runs finished") || line="(file is being written; run again)"
  printf '%-52s %s\n' "$f" "$(echo "$line" | sed 's/^ *//')"
done

hr "EMO in the frozen probe (systems scored, of 21)"
for t in llama-3.2-1b qwen3-1.7b-base $T; do
  python3 - "outputs/$t/classification_probe.json" "$t" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (FileNotFoundError, json.JSONDecodeError):
    print(f"  {sys.argv[2]}: no readable probe file"); sys.exit()
arms = [k for k in d if not k.startswith("_")]
print(f"  {sys.argv[2]:18s} {sum('emo' in d[a] for a in arms)}/{len(arms)}")
PY
done

hr "Iyer replication"
if [ -f iyer_replication.log ]; then
  grep -E "training iyer|trained|checkpoints go|classification-head|generated-answer|left no model|done$" iyer_replication.log | tail -3 | cut -c1-160
  tr '\r' '\n' < iyer_replication.log | grep -E "[0-9]+/[0-9]+ \[" | tail -1 | cut -c1-100
else
  echo "not started (no iyer_replication.log)"
fi
