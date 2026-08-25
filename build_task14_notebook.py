#!/usr/bin/env python3
"""Mechanically build the canonical Task-14 Colab notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
markdown(r"""# Task 14 — External NTP vs NCP audit (canonical)

This notebook runs the smallest experiment capable of answering the central question without
reusing training targets for evaluation.

**Primary hypothesis.** On held-out SWORDS targets, gold-inclusive exact sequence marginalization
(`M-I`) improves acceptable-alternative probability and official ranking GAP over matched
gold-continuation NTP (`C0`) and gold-exclusive marginalization (`M-X`), while preserving the
observed target.

Benchmark roles:

- **SWORDS:** primary human-labelled contextual substitution benchmark.
- **HyperLex:** directional hypernym evaluation; not synthetic NCP training data.
- **bm-semlex:** 200-row wrong-sense guardrail, not a headline benchmark.

Arms, all trained on the same matched volume (`BUDGET` = 1,607 rows, section 2b):

| Arm | Owner | Training signal |
|---|---|---|
| B0 | base | untouched base model |
| A0 | paper | NTP on repeated originals |
| A1 | paper | NCP data augmentation, one sentence per alternative |
| P1 | paper | written mean-log NCP loss, no CLM term |
| C0 | ours | matched gold-continuation NTP |
| M-X / M-I | ours | set marginal, gold-exclusive / gold-inclusive |

SWORDS concept sets are human acceptability judgements, so the paper's Appendix-A LLM extraction
step does not apply here; `A1`'s corpus is a mechanical substitution over those sets.

The default path evaluates untouched 1B/3B bases, then trains the six 1B SWORDS arms with seed 42.
Official SWORDS test, extra seeds, hypernym transfer, and 3B training are gated off.

Storage invariant: models/checkpoints/caches live only under ephemeral `/content`. Drive receives
JSON/CSV reports only. Training writes one final weight copy, evaluates it, and deletes it in a
`finally` block."""),

markdown("## 0. Runtime, authentication, and repository"),
code(r"""import os, sys, subprocess, shutil, json, glob, hashlib
from pathlib import Path

import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
!pip install -q "transformers>=4.57,<5" datasets accelerate pytest pandas scipy bitsandbytes huggingface_hub

from google.colab import drive, userdata
from huggingface_hub import login
drive.mount('/content/drive')
hf_token = userdata.get('HF_TOKEN')
assert hf_token, 'Add HF_TOKEN to Colab Secrets; Llama-3.2 is gated.'
login(token=hf_token, add_to_git_credential=False)
"""),
code(r"""REPO_URL = 'https://github.com/SharvaGogawale1/concept-aware-training.git'
REPO_DIR = Path('/content/concept_aware_training')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)

SCRIPTS = REPO_DIR / 'transformers/examples/pytorch/language-modeling'
required = [
    'sequence_ncp_trainer.py', 'run_clm_sequence_ncp.py', 'eval_concept_ppl_v3.py',
    'eval_swords.py', 'eval_hyperlex.py', 'eval_bm_semlex.py',
    'compare_task14_results.py', 'test_sequence_ncp.py', 'test_task14_external.py',
]
missing = [name for name in required if not (SCRIPTS / name).exists()]
for name in ['prepare_swords_concept_sets.py', 'verify_task14_data.py',
             'rebuild_gold_inclusive.py']:
    if not (REPO_DIR / name).exists(): missing.append(name)
assert not missing, f'Push Task-14 files to GitHub before Colab: {missing}'
os.chdir(REPO_DIR)
print('Repository:', REPO_DIR)
"""),

markdown("## 1. Locked configuration"),
code(r"""DATA = REPO_DIR / 'data'
SCRATCH = Path('/content/task14_scratch')
DRIVE_ROOT = Path('/content/drive').resolve()
RESULTS = DRIVE_ROOT / 'MyDrive/concept_aware_outputs/task14_controlled'
SCRATCH.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)

PRIMARY_SEED = 42
RUN_EXTRA_SEEDS = False
SEEDS = [PRIMARY_SEED] + ([7, 123] if RUN_EXTRA_SEEDS else [])
RUN_PHASE_A = True
RUN_GPU_SMOKE = True
RUN_SWORDS_PILOT = True
RUN_PAPER_ARMS = True              # Iyer et al. A0/A1/P1 on the external benchmarks
RUN_PHASE_C_SWORDS_TEST = False    # enable only after all three 1B seeds pass
RUN_PHASE_C_HYPERNYM = False       # enable only after the SWORDS gate passes
TRAIN_3B_AFTER_PASS = False        # requires A100 and a replicated 1B result

MODEL_REPOS = {'1B': 'meta-llama/Llama-3.2-1B', '3B': 'meta-llama/Llama-3.2-3B'}
MODEL_PATHS = {key: Path(f'/content/Llama-3.2-{key}') for key in MODEL_REPOS}

def assert_ephemeral(path):
    resolved = Path(path).resolve()
    assert os.path.commonpath([str(resolved), str(DRIVE_ROOT)]) != str(DRIVE_ROOT), \
        f'weights/checkpoints may not be written to Drive: {resolved}'

print('seeds:', SEEDS, '| results:', RESULTS)
"""),

markdown("## 2. Fetch, verify, split, and test data"),
code(r"""# Pinned URLs and SHA-256 checks; downloads only when a file is absent.
subprocess.run([
    sys.executable, 'verify_task14_data.py', '--download_missing',
    '--report_json', str(RESULTS / 'data_integrity.json')
], check=True)

SWORDS_DEV = DATA / 'swords/swords-v1.1_dev.json.gz'
SWORDS_TEST = DATA / 'swords/swords-v1.1_test.json.gz'
SW_DIR = DATA / 'swords/concept_dev'
subprocess.run([
    sys.executable, 'prepare_swords_concept_sets.py',
    '--swords_json', str(SWORDS_DEV), '--test_json', str(SWORDS_TEST),
    '--out_dir', str(SW_DIR), '--split_seed', '42', '--train_fraction', '0.8'
], check=True)

# Rebuild both synonym and hypernym gold-inclusive views with the boundary-safe recovery code.
subprocess.run([sys.executable, 'rebuild_gold_inclusive.py'], check=True)
report = json.load(open(SW_DIR / 'prepare_report.json'))
assert report['slots_train'] == 251 and report['slots_val'] == 62
assert report['augmentation_lines']['train'] == 1607
assert report['dev_test_disjointness'] == {
    'target_id_overlap_with_dev': 0, 'context_id_overlap_with_dev': 0}
print(json.dumps(report, indent=2))
"""),

markdown(r"""### 2b. Matched training volume for the Iyer et al. arms

Iyer et al. hold training-instance volume approximately fixed across variants (paper section 2.2),
and the augmentation corpus is what fixes it: A1 is defined as one sentence per alternative, so its
size is not a free parameter. Here that is **1,607** rows.

Every other arm is therefore padded to 1,607 with deterministically repeated originals. Without
this, A1 would train on 3.7x the rows of `C0`/`M_X`/`M_I` and any A1 advantage would be
indistinguishable from seeing more data — the same confound Task 6 was written to rule out.

This matches example count, not token count, which is the paper's own protocol."""),
code(r"""import pandas as pd, numpy as np

def read_lines(path):
    return [line.strip() for line in open(path, encoding='utf-8') if line.strip()]

def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')

def cycle_to_size(rows, size):
    assert rows and size >= 0
    return [rows[index % len(rows)] for index in range(size)]

SW_AUG_TRAIN = SW_DIR / 'context_syn_train.txt'
BUDGET = len(read_lines(SW_AUG_TRAIN))
assert BUDGET == report['augmentation_lines']['train']

originals = read_lines(SW_DIR / 'vanilla_train.txt')
concept_rows = pd.read_csv(SW_DIR / 'context_loss_train_goldexcl.csv')
N_CONCEPT = len(concept_rows)
assert N_CONCEPT < BUDGET, 'replay padding would be negative'

# A0: repeated originals only.  Matched replay: what the hybrid arms add on top of their concept
# rows to reach the same budget.  P1: the paper's gold-exclusive concept rows, cycled to budget.
A0_TRAIN = SCRATCH / 'swords_A0_repeated_original.txt'
MATCHED_REPLAY = SCRATCH / 'swords_matched_replay.txt'
P1_TRAIN = SCRATCH / 'swords_P1_repeated_concepts.csv'
write_lines(A0_TRAIN, cycle_to_size(originals, BUDGET))
write_lines(MATCHED_REPLAY, cycle_to_size(originals, BUDGET - N_CONCEPT))

repeated = concept_rows.iloc[np.arange(BUDGET) % N_CONCEPT].copy()
repeated['row_id'] = [f'{row_id}:paper-repeat:{index:05d}'
                      for index, row_id in enumerate(repeated.row_id.astype(str))]
repeated.to_csv(P1_TRAIN, index=False)

print(f'budget N={BUDGET} | concept rows={N_CONCEPT} | replay={BUDGET - N_CONCEPT}')
"""),
code(r"""# CPU regression tests: exact scores, gold-slot labels, caching, gradients, GAP fixtures,
# split disjointness, external file hashes, and candidate microbatch equivalence.
subprocess.run([
    sys.executable, '-m', 'pytest', '-q', '--confcutdir=.',
    str(SCRIPTS / 'test_sequence_ncp.py'), str(SCRIPTS / 'test_task14_external.py')
], check=True)
"""),

markdown("## 3. Download untouched base models to ephemeral storage"),
code(r"""from huggingface_hub import snapshot_download
for key in ['1B', '3B']:
    snapshot_download(
        repo_id=MODEL_REPOS[key], local_dir=str(MODEL_PATHS[key]), token=hf_token,
        ignore_patterns=['*.msgpack', '*.h5', '*.ot', 'original/*'],
    )
    print(key, MODEL_PATHS[key])
"""),

markdown("## 4. External and intrinsic evaluation functions"),
code(r"""HYPERLEX_LEX = DATA / 'hyperlex-data/splits/lexical/hyperlex_test_all_lexical.txt'
HYPERLEX_RANDOM = DATA / 'hyperlex-data/splits/random/hyperlex_test_all_random.txt'
BM_TSV = DATA / 'bm_semlex/curated_200.tsv'

def run_command(parts):
    print(' '.join(map(str, parts)))
    subprocess.run([str(part) for part in parts], check=True)

def external_eval(tag, checkpoint, model_key, swords_json, target_ids_file=None,
                  full_swords=False, include_hyperlex=True):
    tokenizer = MODEL_PATHS[model_key]
    batch = 32 if model_key == '1B' else 16
    swords_out = RESULTS / f'swords__{tag}.json'
    command = [sys.executable, SCRIPTS / 'eval_swords.py', '--checkpoints', checkpoint,
               '--tokenizer_path', tokenizer, '--swords_json', swords_json,
               '--modes', 'left']
    if full_swords: command += ['full']
    if target_ids_file: command += ['--target_ids_file', target_ids_file]
    command += ['--block_size', '256', '--batch_size', batch,
                '--n_bootstrap', '2000', '--results_json', swords_out]
    run_command(command)

    if include_hyperlex:
        for split, path in [('lexical', HYPERLEX_LEX), ('random', HYPERLEX_RANDOM)]:
            run_command([sys.executable, SCRIPTS / 'eval_hyperlex.py', '--checkpoints', checkpoint,
                         '--tokenizer_path', tokenizer, '--hyperlex', path, '--pos', 'N',
                         'V',
                         '--batch_size', batch, '--n_bootstrap', '2000',
                         '--results_json', RESULTS / f'hyperlex_{split}__{tag}.json'])

    run_command([sys.executable, SCRIPTS / 'eval_bm_semlex.py', '--checkpoints', checkpoint,
                 '--tokenizer_path', tokenizer, '--data', BM_TSV, '--modes', 'left',
                 '--batch_size', batch, '--results_json', RESULTS / f'bm_semlex__{tag}.json'])

def intrinsic_eval(tag, checkpoint, model_key, concept_csv, relation='swords'):
    run_command([sys.executable, SCRIPTS / 'eval_concept_ppl_v3.py',
                 '--checkpoints', checkpoint, '--tokenizer_path', MODEL_PATHS[model_key],
                 '--concept_csv', f'{relation}={concept_csv}',
                 '--vanilla_val', DATA / 'syn/youtube_clean_gold/vanilla_val.txt',
                 '--batch_size', '16', '--n_bootstrap', '2000',
                 '--results_json', RESULTS / f'intrinsic__{tag}.json'])
"""),

markdown("## 5. Phase A — untouched 1B/3B evaluation validation"),
code(r"""if RUN_PHASE_A:
    for model_key in ['1B', '3B']:
        tag = f'base_{model_key}'
        external_eval(tag, MODEL_PATHS[model_key], model_key, SWORDS_DEV,
                      SW_DIR / 'target_ids_val.txt')
        intrinsic_eval(tag, MODEL_PATHS[model_key], model_key,
                       SW_DIR / 'context_loss_val.csv')
print('Phase A complete. Scale differences are descriptive, not a metric-validity gate.')
"""),

markdown("## 6. Phase B — 1B C0/M-X/M-I pilot"),
code(r"""# `data` selects which matched view an arm trains on:
#   hybrid       concept rows + repeated-original replay, padded to BUDGET  (ours)
#   vanilla      repeated originals only                                    (paper NTP baseline)
#   augmentation one sentence per alternative                               (paper NCP data aug.)
#   paper        gold-exclusive concept rows cycled to BUDGET               (paper NCP loss)
#
# base_weight is passed explicitly for every arm.  P1 is the paper's written equation, which
# carries no CLM term at all; leaving base_weight at its 1.0 default would silently train this
# project's hybrid CLM+NCP objective and label it as the paper reconstruction.
ARMS = {
    'C0':  {'objective': 'none', 'alpha': 0.0, 'base_weight': 1.0,
            'view': 'inclusive', 'data': 'hybrid', 'owner': 'ours'},
    'M_X': {'objective': 'set_marginal', 'alpha': 0.5, 'base_weight': 1.0,
            'view': 'exclusive', 'data': 'hybrid', 'owner': 'ours'},
    'M_I': {'objective': 'set_marginal', 'alpha': 0.5, 'base_weight': 1.0,
            'view': 'inclusive', 'data': 'hybrid', 'owner': 'ours'},
    'A0':  {'objective': 'none', 'alpha': 0.0, 'base_weight': 1.0,
            'view': 'inclusive', 'data': 'vanilla', 'owner': 'paper'},
    'A1':  {'objective': 'none', 'alpha': 0.0, 'base_weight': 1.0,
            'view': 'inclusive', 'data': 'augmentation', 'owner': 'paper'},
    'P1':  {'objective': 'paper_mean', 'alpha': 1.0, 'base_weight': 0.0,
            'view': 'exclusive', 'data': 'paper', 'owner': 'paper'},
}
PAPER_ARMS = [arm for arm, spec in ARMS.items() if spec['owner'] == 'paper']
OUR_ARMS = [arm for arm, spec in ARMS.items() if spec['owner'] == 'ours']

# Intrinsic evaluation is held fixed per source so every arm is scored on identical rows.
INTRINSIC_VAL = {
    'swords': SW_DIR / 'context_loss_val.csv',
    'swords_full': SW_DIR / 'context_loss_val.csv',
    'hypernym': DATA / 'hyp/youtube_clean_gold/context_loss_val.csv',
}

def arm_files(arm, source='swords'):
    spec = ARMS[arm]
    suffix = '_goldexcl' if spec['view'] == 'exclusive' else ''
    if spec['data'] != 'hybrid':
        # The budget-matched paper views are built for the SWORDS pilot only. Phase C sources
        # would need their own matched corpora; fail loudly rather than train on a mismatch.
        if source != 'swords':
            raise ValueError(f'paper arm {arm} has no matched view for source={source}')
        train = {'vanilla': A0_TRAIN, 'augmentation': SW_AUG_TRAIN, 'paper': P1_TRAIN}[spec['data']]
        val = (SW_DIR / f'context_loss_val{suffix}.csv' if spec['data'] == 'paper'
               else SW_DIR / 'vanilla_val.txt')
        return {'train': train, 'val': val, 'replay': None}
    if source == 'swords':
        root, train_name, replay = SW_DIR, 'train', MATCHED_REPLAY
    elif source == 'swords_full':
        root, train_name, replay = SW_DIR, 'full', SW_DIR / 'vanilla_full.txt'
    elif source == 'hypernym':
        root = DATA / 'hyp/youtube_clean_gold'
        train_name, replay = 'train', root / 'vanilla_train.txt'
    else:
        raise ValueError(source)
    return {
        'train': root / f'context_loss_{train_name}{suffix}.csv',
        'val': root / f'context_loss_val{suffix}.csv',
        'replay': replay,
    }

SCHEDULE = [
    '--num_train_epochs', '2', '--learning_rate', '1e-5', '--warmup_ratio', '0.1',
    '--block_size', '128', '--per_device_train_batch_size', '1',
    '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '16',
    '--logging_steps', '25', '--save_strategy', 'no', '--save_only_model',
    '--bf16', '--overwrite_output_dir',
    '--do_train', '--do_eval', '--report_to', 'none',
    '--forbidden_output_root', str(DRIVE_ROOT),
    # Required, not cosmetic: text-row dedup defaults ON, and would collapse A0's 1,607 repeated
    # originals and the matched replay back to 251 unique lines, silently undoing the volume
    # matching with no error raised. The rows_raw assertion in train_one is the tripwire.
    '--no-deduplicate_text_rows',
]

def train_one(arm, seed, source='swords', smoke=False, model_key='1B', evaluation='dev'):
    spec, files = ARMS[arm], arm_files(arm, source)
    tag = f'{model_key}_{source}_{arm}_s{seed}' + ('_smoke' if smoke else '')
    output = SCRATCH / tag
    cache = SCRATCH / f'cache_{source}'
    assert_ephemeral(output); assert_ephemeral(cache)
    optimizer = 'adamw_bnb_8bit' if model_key == '3B' else 'adamw_torch'
    # Both settings are numerics-preserving, so they do not affect the comparison:
    # gradient checkpointing only trades recompute for memory, and candidate microbatching is
    # a pure forward-splitting loop (test_candidate_microbatching_matches_one_full_forward).
    # At batch 1 x 128 tokens a 1B model peaks near 11 GiB, so checkpointing buys nothing and
    # costs ~25%. The gated 3B path keeps both conservative settings.
    heavy = model_key == '3B'
    candidate_microbatch = '1' if heavy else '8'
    checkpointing = '--gradient_checkpointing' if heavy else '--no-gradient_checkpointing'
    command = [sys.executable, SCRIPTS / 'run_clm_sequence_ncp.py',
               '--model_name_or_path', MODEL_PATHS[model_key], '--tokenizer_name', MODEL_PATHS[model_key],
               '--train_file', files['train'], '--validation_file', files['val'],
               '--objective', spec['objective'],
               '--ncp_alpha', spec['alpha'], '--base_loss_weight', spec['base_weight'],
               '--required_coverage', '0.99',
               '--preprocessing_cache_dir', cache, '--optim', optimizer,
               '--candidate_microbatch_size', candidate_microbatch, checkpointing,
               '--seed', seed, '--output_dir', output] + SCHEDULE
    if files['replay']:
        command += ['--replay_file', files['replay']]
    if smoke:
        command += ['--max_train_samples', '8', '--max_eval_samples', '8',
                    '--num_train_epochs', '1']
    if evaluation == 'test':
        command += ['--no-do_eval']  # fixed epochs; never select on the official test
    try:
        run_command(command)
        summary = json.load(open(output / 'sequence_ncp_run.json'))
        if spec['objective'] != 'none':
            assert summary['training_concept_coverage']['coverage'] >= 0.99
        if not smoke and source == 'swords':
            rows_raw = summary['train_preprocessing']['rows_raw']
            assert rows_raw == BUDGET, (
                f'{arm} trained on {rows_raw} rows, expected the matched budget {BUDGET}; '
                'check that --no-deduplicate_text_rows is still in SCHEDULE')
        shutil.copy2(output / 'sequence_ncp_run.json', RESULTS / f'train__{tag}.json')
        if not smoke:
            if evaluation == 'test':
                external_eval(tag, output, model_key, SWORDS_TEST, include_hyperlex=False)
            else:
                # HyperLex on every trained arm: Phase A reports base-model numbers, so without
                # this the hypernym benchmark has nothing to be compared against. ~926 pairs.
                external_eval(tag, output, model_key, SWORDS_DEV, SW_DIR / 'target_ids_val.txt',
                              include_hyperlex=True)
                # One fixed concept set per source, never the arm's own validation file:
                # A0/A1 validate on plain text, and M_X on the gold-exclusive view, so passing
                # files['val'] here would both crash the txt arms and score the rest on
                # different data. Phase A uses this same file, so bases stay comparable too.
                relation = 'hypernym' if source == 'hypernym' else 'swords'
                intrinsic_eval(tag, output, model_key, INTRINSIC_VAL[source], relation=relation)
        return tag
    finally:
        shutil.rmtree(output, ignore_errors=True)
        print('deleted ephemeral model:', output)

if RUN_GPU_SMOKE:
    train_one('M_I', PRIMARY_SEED, smoke=True)
"""),
code(r"""PILOT_ARMS = OUR_ARMS + (PAPER_ARMS if RUN_PAPER_ARMS else [])

PILOT_TAGS = []
if RUN_SWORDS_PILOT:
    for seed in SEEDS:
        for arm in PILOT_ARMS:
            PILOT_TAGS.append(train_one(arm, seed))
print('Pilot arms:', PILOT_ARMS)
print('Pilot tags:', PILOT_TAGS)
"""),

markdown("## 7. Paired comparisons and predeclared gate"),
code(r"""def paired_compare(benchmark, baseline_tag, treatment_tag, mode='left'):
    prefix = 'bm_semlex' if benchmark == 'bm_semlex' else benchmark
    output = RESULTS / f'paired_{benchmark}__{treatment_tag}_vs_{baseline_tag}.json'
    run_command([sys.executable, SCRIPTS / 'compare_task14_results.py',
                 '--baseline_json', RESULTS / f'{prefix}__{baseline_tag}.json',
                 '--treatment_json', RESULTS / f'{prefix}__{treatment_tag}.json',
                 '--benchmark', benchmark, '--mode', mode, '--n_bootstrap', '5000',
                 '--results_json', output])
    return json.load(open(output))

def evaluate_seed_gate(seed):
    c0 = f'1B_swords_C0_s{seed}'
    mx = f'1B_swords_M_X_s{seed}'
    mi = f'1B_swords_M_I_s{seed}'
    mi_vs_c0 = paired_compare('swords', c0, mi)
    mi_vs_mx = paired_compare('swords', mx, mi)
    bm_mi_vs_c0 = paired_compare('bm_semlex', c0, mi)

    def metric(report, name): return report['metrics'].get(name, {})
    gap = metric(mi_vs_c0, 'gap_rat')
    alt = metric(mi_vs_c0, 'alternatives_nll')
    gold_vs_c0 = metric(mi_vs_c0, 'gold_nll')
    gold_vs_mx = metric(mi_vs_mx, 'gold_nll')
    single = metric(mi_vs_c0, 'acceptable_single_logp_mean')
    multi = metric(mi_vs_c0, 'acceptable_multi_logp_mean')
    bm = metric(bm_mi_vs_c0, 'correct_float')

    c0_intr = json.load(open(RESULTS / f'intrinsic__{c0}.json'))[0]
    mi_intr = json.load(open(RESULTS / f'intrinsic__{mi}.json'))[0]
    ntp_delta = mi_intr['ntp']['ntp_nll_mean'] - c0_intr['ntp']['ntp_nll_mean']
    gate = {
        'gap_ratio_positive': bool(gap and gap['ci95'][0] > 0),
        'alternatives_nll_improves': bool(alt and alt['ci95'][1] < 0),
        'observed_gold_nll_delta_le_0.20': bool(
            gold_vs_c0 and gold_vs_c0['treatment_minus_baseline'] <= 0.20
        ),
        'gold_retained_better_than_MX': bool(gold_vs_mx and gold_vs_mx['treatment_minus_baseline'] < 0),
        'general_ntp_delta_le_0.20': ntp_delta <= 0.20,
        'bm_wrong_sense_noninferior_5pp': bool(bm and bm['ci95'][0] > -0.05),
        'single_token_gain': bool(single and single['treatment_minus_baseline'] > 0),
        'multi_token_gain': bool(multi and multi['treatment_minus_baseline'] > 0),
        'ntp_delta': ntp_delta,
    }
    gate['passes_all'] = all(value for key, value in gate.items()
                              if key not in {'ntp_delta', 'passes_all'})
    json.dump(gate, open(RESULTS / f'swords_pilot_gate_s{seed}.json', 'w'), indent=2)
    return gate

GATES = {seed: evaluate_seed_gate(seed) for seed in SEEDS} if RUN_SWORDS_PILOT else {}
GATE = GATES.get(PRIMARY_SEED, {})
ALL_SEEDS_PASS = bool(GATES) and all(gate.get('passes_all') for gate in GATES.values())
print(json.dumps({'per_seed': GATES, 'all_seeds_pass': ALL_SEEDS_PASS}, indent=2))
"""),

markdown(r"""### 7b. Iyer et al. baselines on the external benchmarks

`M_I` is compared against each paper arm on SWORDS and bm-semlex. These are **reported, not
gates** — `evaluate_seed_gate` above keeps its predeclared criteria against `C0` and `M_X`, so
adding baselines cannot move the pass/fail line after the fact.

Arm ownership: `B0` untouched base, `A0`/`A1`/`P1` from Iyer et al., `C0`/`M_X`/`M_I` ours."""),
code(r"""def headline(prefix, tag, mode='left'):
    path = RESULTS / f'{prefix}__{tag}.json'
    if not path.exists():
        return {}
    record = json.load(open(path))[0]
    return record.get(mode, record)

PAPER_COMPARISONS = {}
if RUN_SWORDS_PILOT and RUN_PAPER_ARMS:
    for seed in SEEDS:
        mi = f'1B_swords_M_I_s{seed}'
        for arm in PAPER_ARMS:
            baseline = f'1B_swords_{arm}_s{seed}'
            PAPER_COMPARISONS[f'{arm}_s{seed}'] = {
                'swords': paired_compare('swords', baseline, mi),
                'bm_semlex': paired_compare('bm_semlex', baseline, mi),
            }

def summary_row(arm, seed):
    tag = 'base_1B' if arm == 'B0' else f'1B_swords_{arm}_s{seed}'
    swords, bm = headline('swords', tag), headline('bm_semlex', tag)
    intrinsic_path = RESULTS / f'intrinsic__{tag}.json'
    intrinsic = json.load(open(intrinsic_path))[0] if intrinsic_path.exists() else {}
    return {
        'arm': arm,
        'owner': 'base' if arm == 'B0' else ARMS[arm]['owner'],
        'seed': '-' if seed is None else seed,
        'GAP ratio': swords.get('gap_rat'),
        'spearman': swords.get('spearman'),
        'ALT NLL': swords.get('alternatives_nll'),
        'GOLD NLL': swords.get('gold_nll'),
        'bm correct': bm.get('correct_float') if isinstance(bm, dict) else None,
        'NTP NLL': (intrinsic.get('ntp') or {}).get('ntp_nll_mean'),
    }

# B0 is seed-independent, so it appears once rather than repeated per seed.
plan = ([('B0', None)] if RUN_PHASE_A else []) + (
    [(arm, seed) for seed in SEEDS for arm in PILOT_ARMS] if RUN_SWORDS_PILOT else [])
rows = [summary_row(arm, seed) for arm, seed in plan]

external_table = pd.DataFrame(rows)
external_table.to_csv(RESULTS / 'task14_external_summary.csv', index=False)
print(external_table.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
"""),

markdown(r"""## 8. Conditional Phase C — locked SWORDS test, hypernym transfer, and scale

Leave these flags disabled until all three 1B development seeds pass. The first conditional block
re-trains matched arms on all SWORDS dev and touches official test exactly once per trained model.
The second trains on boundary-safe `hyp/youtube_clean_gold` and reads HyperLex lexical/random as
relation-specific evaluation. bm-semlex remains a synonym guardrail, not hypernym evidence."""),
code(r"""FINAL_SWORDS_TAGS = []
if RUN_PHASE_C_SWORDS_TEST:
    assert RUN_EXTRA_SEEDS, 'Replicate the 1B dev result with seeds 7 and 123 before test access.'
    assert ALL_SEEDS_PASS, 'Every enabled 1B seed must pass before official SWORDS test access.'
    for seed in SEEDS:
        for arm in OUR_ARMS:
            FINAL_SWORDS_TAGS.append(
                train_one(arm, seed, source='swords_full', evaluation='test')
            )
print('Official-test tags:', FINAL_SWORDS_TAGS)

PHASE_C_TAGS = []
if RUN_PHASE_C_HYPERNYM:
    assert RUN_EXTRA_SEEDS and ALL_SEEDS_PASS, 'Replicate the 1B SWORDS result before expansion.'
    for seed in SEEDS:
        for arm in OUR_ARMS:
            PHASE_C_TAGS.append(train_one(arm, seed, source='hypernym'))
print('Phase C tags:', PHASE_C_TAGS)

SCALE_TAGS = []
if TRAIN_3B_AFTER_PASS:
    assert RUN_EXTRA_SEEDS and ALL_SEEDS_PASS, 'Replicate the 1B result before scaling.'
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    assert gpu_gb >= 35, f'3B concept training requires an A100-class GPU; found {gpu_gb:.1f} GiB.'
    for seed in SEEDS:
        for arm in OUR_ARMS:
            SCALE_TAGS.append(train_one(arm, seed, source='swords', model_key='3B'))
print('Conditional 3B tags:', SCALE_TAGS)
"""),

markdown("## 9. Final storage audit"),
code(r"""shutil.rmtree(SCRATCH, ignore_errors=True)
prohibited = ['*.safetensors', '*.bin', 'optimizer.pt', 'scheduler.pt',
              'rng_state.pth', 'scaler.pt']
leftovers = [path for pattern in prohibited
             for path in RESULTS.rglob(pattern)]
assert not leftovers, f'weight/optimizer artifacts reached Drive: {leftovers[:5]}'
print('Storage audit passed. Drive contains reports only:')
for path in sorted(RESULTS.iterdir()):
    print(f'{path.stat().st_size / 1024:9.1f} KB  {path.name}')
"""),

markdown(r"""## Interpretation

- Treat the one-seed run as a development screen, not a paper result.
- The decisive concept metric is **gold-excluded acceptable-alternative NLL**, not inclusive NLL.
- `gap_rat` is the official-style rater-ratio GAP; `gap` uses raw TRUE counts. Oracle-k F1 is
  secondary and is labelled as such because k is taken from gold labels.
- Do not infer an annotation-quality effect by contrasting unmatched SWORDS and YouTube corpora.
- If the gate passes, enable seeds 7/123, lock the schedule, retrain on all SWORDS dev, evaluate
  official SWORDS test once, and only then consider 3B training on an A100."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("research_tasks_14_external_ntp_vs_ncp.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
)
print(f"wrote research_tasks_14_external_ntp_vs_ncp.ipynb ({len(cells)} cells)")
