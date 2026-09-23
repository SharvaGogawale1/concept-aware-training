#!/usr/bin/env bash
# One family, end to end, unattended: mine the verified data unless a valid copy
# exists, then set the flags, name the model, and run the notebook (data gate ->
# smoke + lambda rule -> six arms -> evaluation -> two-claim gate).
#
#   run_verified_pipeline.sh <model-tag> <model-id> <gpu> [set_15b_flags options...]
#   run_verified_pipeline.sh qwen3-1.7b-base Qwen/Qwen3-1.7B-Base 0
#   run_verified_pipeline.sh llama-3.2-1b   meta-llama/Llama-3.2-1B 3
#
# Extra options go to set_15b_flags.py and override the defaults (e.g. --smoke-steps 0,
# --seeds 42,123,2024, --arms verified_mass,verified_alt_uniform, --no-eval).  To run
# two passes of one family at once, give each its own RUN_NAME: it names the notebook
# copy, the executed notebook and the log, so the two never edit the same file.
#   RUN_NAME=confirm run_verified_pipeline.sh llama-3.2-1b meta-llama/Llama-3.2-1B 1 \
#       --smoke-steps 0 --seeds 42,123,2024 --arms verified_zhang,verified_hybrid --no-eval
#
# This script never deletes data.  "Valid" means: the miner report exists, was
# written by the CURRENT miner version, and records the SHA-256 of the file on
# disk.  Anything else is re-mined with --overwrite, which replaces the file in
# place.  A miner already running for this family is waited on, never touched.
set -euo pipefail
TAG=$1; MODEL=$2; GPU=$3; shift 3
EXTRA=("$@")
# STAGE=verified (default) mines the data first; STAGE=hybrid runs the original-data
# hybrid stage of 15b; STAGE=confirm15 runs Task 15's extra seeds.  The last two
# never touch the verified data.
STAGE=${STAGE:-verified}
SUFFIX=${RUN_NAME:+_$RUN_NAME}
case $STAGE in
  confirm15) SRC=reproducibilty_15.ipynb;  NB=reproducibilty_15_$TAG$SUFFIX.ipynb ;;
  *)         SRC=reproducibilty_15b.ipynb; NB=reproducibilty_15b_$TAG$SUFFIX.ipynb ;;
esac
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
REPO=concept_aware/concept-aware-training
D=concept_aware/data/c4/$TAG
OUT=outputs/$TAG
DATA=$D/embedding/synonyms_train_verified.jsonl
REPORT=$OUT/verified_report.json
export HF_HOME=$BASE/concept_aware/hf_cache PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CONCEPT_BASE=$BASE
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

# Deliberately no git pull here: a family already running this script must never
# have its script file rewritten underneath it.  Pull once, before launching.

miner_running() {
  pgrep -f "build_verified_negatives.py.*--source $D/embedding/synonyms_train.jsonl" >/dev/null
}

# Exit 0 only when the report is from the current miner, for exactly this file.
data_is_valid() {
  python3 - "$DATA" "$REPORT" "$REPO/scripts/build_verified_negatives.py" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
data, report, miner = map(Path, sys.argv[1:4])
if not data.is_file():   print("  validation: no verified file");            sys.exit(1)
if not report.is_file(): print("  validation: no miner report");            sys.exit(1)
want = re.search(r'MINER_VERSION = "([^"]+)"', miner.read_text()).group(1)
r = json.loads(report.read_text())
if r.get("miner_version") != want:
    print(f"  validation: report is from miner {r.get('miner_version')!r}, current is {want!r}"); sys.exit(2)
digest = hashlib.sha256()
with data.open("rb") as h:
    for chunk in iter(lambda: h.read(1 << 20), b""):
        digest.update(chunk)
if r.get("output_sha256") != digest.hexdigest():
    print("  validation: report does not describe the file on disk"); sys.exit(3)
if not r.get("complete"):
    print("  validation: report is from a partial run"); sys.exit(4)
print(f"  validation: OK (miner {want}, {r.get('rows')} rows, sha {digest.hexdigest()[:12]})")
PY
}

# ---- 1. verified data ------------------------------------------------------------
if [ "$STAGE" != verified ]; then
  log "STAGE=$STAGE: skipping the verified data"
elif miner_running; then
  log "a miner for $TAG is already running; waiting for it to exit"
  while miner_running; do sleep 120; done
fi
if data_is_valid; then
  log "verified data for $TAG is valid; not re-mining"
else
  log "mining verified negatives for $TAG on GPU $GPU (~10 h)"
  CUDA_VISIBLE_DEVICES=$GPU python $REPO/scripts/build_verified_negatives.py \
    --source $D/embedding/synonyms_train.jsonl --topk "$D/prompting/topk_*.jsonl" \
    --model "$MODEL" --ext concept_aware/learning-concepts/data \
    --output "$DATA" --report "$REPORT" --sample $OUT/verified_sample_50.csv \
    --prune-positives --overwrite 2>&1 | tee -a verify_$TAG.log
  data_is_valid || { log "the miner finished but its output does not validate; stopping"; exit 1; }
fi

# ---- 2. flags, then the notebook: gate -> smoke + lambda rule -> train -> eval --------
cp $REPO/notebooks/$SRC "$NB"
# Defaults first: argparse keeps the LAST value, so anything in EXTRA overrides them.
python3 $REPO/scripts/set_15b_flags.py "$NB" --gpu "$GPU" --base "$BASE" \
  --model-tag "$TAG" --stage "$STAGE" --smoke-steps 40 ${EXTRA[@]+"${EXTRA[@]}"}
# The control cell names the model; the flag script keeps whatever is there, so set it.
python3 - "$MODEL" "$NB" <<'PY'
import json, re, sys
model, path = sys.argv[1], sys.argv[2]
nb = json.load(open(path))
hits = 0
for c in nb["cells"]:
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if "THE ONLY CELL YOU EDIT" in src:
        src, n = re.subn(r'^MODEL    = "[^"]*"', f'MODEL    = "{model}"', src, flags=re.M)
        hits += n
        c["source"] = src
assert hits == 1, f"expected to set MODEL once, set it {hits} times"
json.dump(nb, open(path, "w"), indent=1)
print("MODEL set to", model)
PY
log "running the notebook for $TAG$SUFFIX on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU papermill "$NB" executed_15b_$TAG$SUFFIX.ipynb -k concept --log-output 2>&1 | tee -a verified_$TAG$SUFFIX.log
log "done: $TAG$SUFFIX ($STAGE)"
