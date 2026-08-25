#!/usr/bin/env python3
"""Build the canonical, output-free Task-13 Colab notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
markdown(r"""# Task 13 - controlled 1B reconstruction of Iyer et al. and our extensions

This notebook asks whether the main Iyer et al. baselines and our exact sequence objectives
produce a repeatable NTP-versus-concept tradeoff under one three-epoch schedule.

For context `x`, continuation `c=(c_1,...,c_m)`, and
`s_c = sum_t log p(c_t | x,c_<t)`:

```text
Iyer written NCP loss:       L_paper = -(1/|C|) sum_c s_c
Our set marginal:            L_set   = -log sum_c exp(s_c)
Our mean-set metric:         L_mean  = -log[(1/|C|) sum_c exp(s_c)]
Our scaled training variant: L_scale = -(1/|C|) log sum_c exp(s_c)
```

`L_mean = L_set + log|C|`, so it has identical gradients to `L_set`; training it would duplicate
A2. It is reported as a normalized metric. The PI's non-redundant `1/|C|` training experiment is
`L_scale`, called A3.

The paper uses Llama-3-8B and about 8K examples per variant. This is a **controlled 1B
reconstruction**, not an exact scale replication. It repairs gradient flow, gold-slot alignment,
multi-token scoring, padding, leakage, and caching. The strongest paper method overall on ordinary
NTP was context-aware hypernym data augmentation; synonym augmentation is not universally best.

| Arm | Ownership | Training signal |
|---|---|---|
| B0 | paper | untouched base model |
| A0s/A0h | paper | standard CLM on repeated originals, volume-matched |
| A1s/A1h | paper | standard CLM on synonym/hypernym augmented sentences |
| P1s/P1h | paper equation | exact `L_paper`, gold-exclusive, base weight 0, concept weight 1 |
| D2s/D2h | control | exact A2 rows and replay, concept weight 0 |
| A2s/A2h | ours | gold-inclusive exact full-sequence `L_set` |
| A3s/A3h | ours | gold-inclusive `L_set/|C|` |

All headline arms use seeds `[42, 123, 2024]`; seed 7 is not used. The default first pass is the
synonym matrix. Hypernym and cross-family arms are configured behind explicit flags.
"""),
markdown("## 0. Runtime, authentication, and repository"),
code(r"""import os, sys, subprocess, shutil, json, glob, math, random, statistics, ast
from pathlib import Path
import torch, pandas as pd, numpy as np

assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
!pip install -q "transformers>=4.57,<5" datasets accelerate pytest pandas scipy matplotlib nltk huggingface_hub

from google.colab import drive, userdata
from huggingface_hub import login
drive.mount('/content/drive')
hf_token = userdata.get('HF_TOKEN')
assert hf_token, 'Add HF_TOKEN to Colab Secrets for gated Llama access.'
login(token=hf_token, add_to_git_credential=False)
"""),
code(r"""REPO_URL = 'https://github.com/SharvaGogawale1/concept-aware-training.git'
REPO_DIR = Path('/content/concept_aware_training')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)
os.chdir(REPO_DIR)

SCRIPTS = REPO_DIR / 'transformers/examples/pytorch/language-modeling'
required = ['sequence_ncp_trainer.py', 'run_clm_sequence_ncp.py',
            'eval_concept_ppl_v3.py', 'test_sequence_ncp.py']
missing = [name for name in required if not (SCRIPTS / name).exists()]
assert not missing, f'Push Task-13 files before Colab: {missing}'
source = (SCRIPTS / 'sequence_ncp_trainer.py').read_text() + \
         (SCRIPTS / 'run_clm_sequence_ncp.py').read_text()
for token in ['paper_mean', 'set_marginal_scaled', 'base_loss_weight',
              'candidate_microbatch_size', 'deduplicate_text_rows']:
    assert token in source, f'missing required implementation: {token}'
"""),
markdown("## 1. Locked configuration"),
code(r"""DATA_ROOT = REPO_DIR / 'data'
SCRATCH = Path('/content/task13_scratch')
DRIVE_ROOT = Path('/content/drive').resolve()
RESULTS = DRIVE_ROOT / 'MyDrive/concept_aware_outputs/task13_controlled'
SCRATCH.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
EPOCHS = 3
RUN_HYPERNYM = False
ACTIVE_RELATIONS = ['syn'] + (['hyp'] if RUN_HYPERNYM else [])
RUN_RELEASED_CODE_DIAGNOSTIC = False
RUN_CROSS_FAMILY = False
CROSS_FAMILY_MODELS = ['falcon1b', 'pythia14b']
CROSS_FAMILY_SEEDS = [42]

MODEL_REPOS = {
    'llama1b': 'meta-llama/Llama-3.2-1B',
    'falcon1b': 'tiiuae/falcon-rw-1b',
    'pythia14b': 'EleutherAI/pythia-1.4b',
}
PRIMARY_MODEL = 'llama1b'
ACTIVE_MODELS = [PRIMARY_MODEL] + (CROSS_FAMILY_MODELS if RUN_CROSS_FAMILY else [])
MODEL_PATHS = {key: Path(f'/content/model_{key}') for key in ACTIVE_MODELS}

def assert_ephemeral(path):
    resolved = Path(path).resolve()
    assert os.path.commonpath([str(resolved), str(DRIVE_ROOT)]) != str(DRIVE_ROOT), \
        f'weights/checkpoints may not be written to Drive: {resolved}'

print('relations:', ACTIVE_RELATIONS, '| seeds:', SEEDS, '| cross-family:', ACTIVE_MODELS[1:])
"""),
code(r"""from huggingface_hub import snapshot_download
for key in ACTIVE_MODELS:
    snapshot_download(repo_id=MODEL_REPOS[key], local_dir=str(MODEL_PATHS[key]), token=hf_token,
                      ignore_patterns=['*.msgpack', '*.h5', '*.ot', 'original/*'])
    print(key, MODEL_PATHS[key])
"""),
]

cells += [
markdown(r"""## 2. Matched paper and extension datasets

The paper explicitly inflates original sentences for NTP baselines and holds training-instance
volume approximately fixed. For each relation, augmentation determines budget `N`: A0 gets `N`
repeated originals; A1 gets the `N` augmented sentences; P1 gets concept rows deterministically
repeated to `N`; D2/A2/A3 get every concept row once plus `N-n_concept` repeated-original replay
rows. D2/A2/A3 are byte-for-byte data matched.

This matches example count, not exact token count. Both are reported. A1 versus A0 still includes
the intended lexical-diversity difference and should be described that way."""),
code(r"""subprocess.run([sys.executable, 'rebuild_gold_inclusive.py'], check=True)

RAW = {
    'syn': {'clean': DATA_ROOT / 'syn/youtube_clean',
            'gold': DATA_ROOT / 'syn/youtube_clean_gold'},
    'hyp': {'clean': DATA_ROOT / 'hyp/youtube_clean',
            'gold': DATA_ROOT / 'hyp/youtube_clean_gold'},
}

def read_lines(path):
    return [line.strip() for line in open(path, encoding='utf-8') if line.strip()]

def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')

def cycle_to_size(rows, size):
    assert rows and size >= 0
    return [rows[index % len(rows)] for index in range(size)]

DATA, budget_report = {}, []
for relation, roots in RAW.items():
    clean, gold_root = roots['clean'], roots['gold']
    augmented = read_lines(clean / 'context_syn_train.txt')
    originals = read_lines(clean / 'vanilla_train.txt')
    budget = len(augmented)
    inclusive = pd.read_csv(gold_root / 'context_loss_train.csv')
    exclusive = pd.read_csv(gold_root / 'context_loss_train_goldexcl.csv')
    assert list(inclusive.row_id) == list(exclusive.row_id)
    assert len(inclusive) <= budget

    matched_original = SCRATCH / f'{relation}_A0_repeated_original.txt'
    matched_replay = SCRATCH / f'{relation}_matched_replay.txt'
    repeated_concepts = SCRATCH / f'{relation}_P1_repeated_concepts.csv'
    write_lines(matched_original, cycle_to_size(originals, budget))
    write_lines(matched_replay, cycle_to_size(originals, budget - len(inclusive)))

    repeated = exclusive.iloc[np.arange(budget) % len(exclusive)].copy()
    repeated['row_id'] = [f'{row_id}:paper-repeat:{index:05d}'
                          for index, row_id in enumerate(repeated.row_id.astype(str))]
    repeated.to_csv(repeated_concepts, index=False)

    DATA[relation] = {
        'budget': budget, 'vanilla_train': matched_original,
        'aug_train': clean / 'context_syn_train.txt',
        'vanilla_val': clean / 'vanilla_val.txt',
        'gold_train': gold_root / 'context_loss_train.csv',
        'gold_val': gold_root / 'context_loss_val.csv',
        'excl_train': gold_root / 'context_loss_train_goldexcl.csv',
        'excl_val': gold_root / 'context_loss_val_goldexcl.csv',
        'paper_train': repeated_concepts, 'replay': matched_replay,
    }
    for path in DATA[relation].values():
        if isinstance(path, Path): assert path.exists(), path
    budget_report.append({
        'relation': relation, 'target_examples': budget,
        'A0_rows': len(read_lines(matched_original)), 'A1_rows': len(augmented),
        'concept_rows': len(inclusive), 'matched_replay_rows': len(read_lines(matched_replay)),
        'P1_rows': len(repeated), 'unique_original_sentences': len(set(originals)),
    })

budget_df = pd.DataFrame(budget_report)
display(budget_df); budget_df.to_csv(RESULTS / 'task13_data_budgets.csv', index=False)
assert all(budget_df.A0_rows == budget_df.A1_rows)
assert all(budget_df.P1_rows == budget_df.A1_rows)
assert all(budget_df.concept_rows + budget_df.matched_replay_rows == budget_df.A1_rows)
"""),
code(r"""for relation in ['syn', 'hyp']:
    inc = pd.read_csv(DATA[relation]['gold_train'])
    exc = pd.read_csv(DATA[relation]['excl_train'])
    assert list(inc.row_id) == list(exc.row_id)
    for left, right, gold in zip(inc.context_syn, exc.context_syn, inc.gold_surface):
        inclusive = {str(value).casefold() for value in ast.literal_eval(left)}
        exclusive = {str(value).casefold() for value in ast.literal_eval(right)}
        gold = str(gold).casefold()
        assert gold in inclusive and gold not in exclusive
        assert inclusive == exclusive | {gold}
print('Gold-inclusive/exclusive alignment passed.')
"""),
markdown("## 3. Arm registry and CPU gates"),
code(r"""def arm_registry(relations):
    result = {'B0': dict(label='Untouched base', owner='paper', relation='none',
                         objective='base', alpha=0., base_weight=0., data=None)}
    for relation in relations:
        suffix = 's' if relation == 'syn' else 'h'
        pretty = 'synonym' if relation == 'syn' else 'hypernym'
        result.update({
            f'A0{suffix}': dict(label=f'A0 {pretty} repeated-original NTP', owner='paper',
                relation=relation, objective='none', alpha=0., base_weight=1., data='vanilla'),
            f'A1{suffix}': dict(label=f'A1 {pretty} data augmentation', owner='paper',
                relation=relation, objective='none', alpha=0., base_weight=1., data='augmentation'),
            f'P1{suffix}': dict(label=f'P1 {pretty} written mean-log NCP', owner='paper',
                relation=relation, objective='paper_mean', alpha=1., base_weight=0., data='paper'),
            f'D2{suffix}': dict(label=f'D2 {pretty} alpha-zero control', owner='control',
                relation=relation, objective='none', alpha=0., base_weight=1., data='hybrid'),
            f'A2{suffix}': dict(label=f'A2 {pretty} set marginal', owner='ours',
                relation=relation, objective='set_marginal', alpha=.5, base_weight=1., data='hybrid'),
            f'A3{suffix}': dict(label=f'A3 {pretty} scaled marginal / |C|', owner='ours',
                relation=relation, objective='set_marginal_scaled', alpha=.5,
                base_weight=1., data='hybrid'),
        })
        if RUN_RELEASED_CODE_DIAGNOSTIC:
            result[f'R1{suffix}'] = dict(
                label=f'R1 {pretty} repaired released-code hybrid', owner='diagnostic',
                relation=relation, objective='paper_mean', alpha=1., base_weight=1.,
                data='hybrid_excl')
    return result

ARMS = arm_registry(ACTIVE_RELATIONS)
for key, spec in ARMS.items(): print(f"{key:4s} [{spec['owner']:10s}] {spec['label']}")
"""),
code(r"""subprocess.run([sys.executable, '-m', 'pytest', '-q', '--confcutdir=.',
                str(SCRIPTS / 'test_sequence_ncp.py')], check=True)
"""),
code(r"""def files_for(spec):
    relation, view = spec['relation'], spec['data']
    if view == 'vanilla':
        return DATA[relation]['vanilla_train'], DATA[relation]['vanilla_val'], None
    if view == 'augmentation':
        return DATA[relation]['aug_train'], DATA[relation]['vanilla_val'], None
    if view == 'paper':
        return DATA[relation]['paper_train'], DATA[relation]['excl_val'], None
    if view == 'hybrid':
        return DATA[relation]['gold_train'], DATA[relation]['gold_val'], DATA[relation]['replay']
    if view == 'hybrid_excl':
        return DATA[relation]['excl_train'], DATA[relation]['excl_val'], DATA[relation]['replay']
    raise ValueError(view)

def preprocess_report(arm, overwrite):
    spec = ARMS[arm]; train, val, replay = files_for(spec)
    report = SCRATCH / f'preprocess_{arm}_{"fresh" if overwrite else "cached"}.json'
    command = [sys.executable, SCRIPTS / 'run_clm_sequence_ncp.py',
        '--model_name_or_path', MODEL_PATHS[PRIMARY_MODEL],
        '--tokenizer_name', MODEL_PATHS[PRIMARY_MODEL], '--train_file', train,
        '--validation_file', val, '--objective', spec['objective'],
        '--ncp_alpha', spec['alpha'], '--base_loss_weight', spec['base_weight'],
        '--output_dir', SCRATCH / f'preprocess_{arm}', '--preprocess_only',
        '--preprocessing_report', report, '--preprocessing_cache_dir', SCRATCH / f'cache_{arm}',
        '--no-deduplicate_text_rows']
    if replay: command += ['--replay_file', replay]
    if overwrite: command += ['--overwrite_cache']
    subprocess.run([str(value) for value in command], check=True)
    return json.load(open(report))

preprocess_rows = []
for arm in [key for key in ARMS if key != 'B0']:
    fresh, cached = preprocess_report(arm, True), preprocess_report(arm, False)
    left, right = fresh['train_preprocessing'], cached['train_preprocessing']
    for field in ['rows_raw','rows_kept','raw_rows_sha256','candidate_metadata_sha256']:
        assert left[field] == right[field], (arm, field)
    assert left['rows_raw'] == DATA[ARMS[arm]['relation']]['budget']
    if ARMS[arm]['objective'] != 'none': assert left['supervision_coverage'] >= .99
    preprocess_rows.append({'arm': arm, **left})
preprocess_table = pd.DataFrame(preprocess_rows)
preprocess_table.to_csv(RESULTS / 'task13_preprocessing_and_token_budgets.csv', index=False)
display(preprocess_table[['arm','rows_raw','rows_kept','base_supervised_tokens',
                          'candidate_sequences','candidate_supervised_tokens',
                          'supervision_coverage','duplicate_context_rows_preserved']])
print('Preprocessing/cache gates passed for every active arm.')
"""),
]

cells += [
markdown(r"""## 4. Train three epochs, evaluate identically, then delete weights

Every trained arm uses LR `1e-5`, 10% warmup, effective batch 16, block size 128, and three epochs.
No best-checkpoint selection is performed. Training curves and final JSON survive; weights and
optimizer state do not.

Ordinary NTP is reported separately on synonym and hypernym original validation sets. Concept
evaluation uses exact full candidate sequences. The headline concept metric is gold-excluded
alternatives-only NLL. Its `mean` counterpart removes set-size bias for reporting only."""),
code(r"""SCHEDULE = [
    '--num_train_epochs', str(EPOCHS), '--learning_rate', '1e-5', '--warmup_ratio', '.1',
    '--block_size', '128', '--per_device_train_batch_size', '1',
    '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '16',
    '--logging_steps', '10', '--save_strategy', 'no', '--save_only_model',
    '--bf16', '--gradient_checkpointing', '--overwrite_output_dir', '--do_train', '--do_eval',
    '--report_to', 'none', '--candidate_microbatch_size', '2',
    '--forbidden_output_root', DRIVE_ROOT, '--no-deduplicate_text_rows',
]
# Stream the child's output into the cell. `subprocess.run` inherits the kernel's file
# descriptors, which Colab does not forward into cell output, so the trainer's logging_steps
# lines were invisible. PYTHONUNBUFFERED is the half that actually matters: without it the pipe
# still arrives in 8 KB bursts.
def run_command(parts):
    parts = [str(value) for value in parts]
    print(' '.join(parts), flush=True)
    process = subprocess.Popen(
        parts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, parts)

def evaluate(tag, checkpoint, model_key):
    output = RESULTS / f'eval__{tag}.json'
    run_command([sys.executable, SCRIPTS / 'eval_concept_ppl_v3.py',
        '--checkpoints', checkpoint, '--tokenizer_path', MODEL_PATHS[model_key],
        '--concept_csv', f"syn={RAW['syn']['gold'] / 'context_loss_val.csv'}",
                         f"hyp={RAW['hyp']['gold'] / 'context_loss_val.csv'}",
        '--vanilla_val', f"syn={RAW['syn']['clean'] / 'vanilla_val.txt'}",
                         f"hyp={RAW['hyp']['clean'] / 'vanilla_val.txt'}",
        '--gold_column', 'gold_surface', '--block_size', '128', '--batch_size', '16',
        '--n_bootstrap', '2000', '--seed', '42', '--results_json', output])
    return output

def train_and_eval(arm, model_key, seed):
    spec = ARMS[arm]; tag = f'{model_key}_{arm}_s{seed}'
    output = SCRATCH / tag; assert_ephemeral(output)
    try:
        if spec['objective'] == 'base':
            evaluation = evaluate(tag, MODEL_PATHS[model_key], model_key)
            record = dict(tag=tag, arm=arm, model=model_key, seed=seed, **spec,
                          coverage=None, train_metrics={}, log_history=[],
                          eval_json=str(evaluation))
        else:
            train, val, replay = files_for(spec)
            command = [sys.executable, SCRIPTS / 'run_clm_sequence_ncp.py',
                '--model_name_or_path', MODEL_PATHS[model_key],
                '--tokenizer_name', MODEL_PATHS[model_key], '--train_file', train,
                '--validation_file', val, '--objective', spec['objective'],
                '--ncp_alpha', spec['alpha'], '--base_loss_weight', spec['base_weight'],
                '--contrast_beta', '0', '--required_coverage', '.99',
                '--preprocessing_cache_dir', SCRATCH / f'cache_{model_key}_{arm}',
                '--seed', seed, '--output_dir', output] + SCHEDULE
            if replay: command += ['--replay_file', replay]
            run_command(command)
            summary = json.load(open(output / 'sequence_ncp_run.json'))
            preprocessing = summary['train_preprocessing']
            assert preprocessing['rows_raw'] == DATA[spec['relation']]['budget']
            coverage = summary['training_concept_coverage']['coverage']
            if spec['objective'] != 'none': assert coverage >= .99
            evaluation = evaluate(tag, output, model_key)
            record = dict(tag=tag, arm=arm, model=model_key, seed=seed, **spec,
                          coverage=coverage, train_metrics=summary.get('train_metrics', {}),
                          log_history=summary.get('log_history', []), eval_json=str(evaluation),
                          preprocessing=preprocessing)
        json.dump(record, open(RESULTS / f'run__{tag}.json', 'w'), indent=2)
        return record
    finally:
        shutil.rmtree(output, ignore_errors=True)
        print('deleted ephemeral model:', output)

# Resume: a run whose Drive record exists is loaded, not recomputed.
#
# Colab Pro has no background execution, so a multi-hour cell will not survive a closed laptop.
# Every finished run writes run__{tag}.json to Drive before the next one starts, so the only work
# a disconnect can destroy is the single run in flight; re-running this cell in a fresh session
# picks up from there. Delete a run__*.json to force that one run to recompute.
def load_or_run(arm, model_key, seed):
    path = RESULTS / f'run__{model_key}_{arm}_s{seed}.json'
    if path.exists():
        print('skip (already in Drive):', path.name, flush=True)
        return json.load(open(path))
    return train_and_eval(arm, model_key, seed)
"""),
code(r"""import time

# Seed-major, so an interrupted session leaves whole seeds finished rather than fragments of
# every arm. Seed 42 alone is a complete six-arm result you can read; half of each arm is not.
PLAN = [('B0', PRIMARY_MODEL, 42)]
PLAN += [(arm, PRIMARY_MODEL, seed) for seed in SEEDS
         for arm in [key for key in ARMS if key != 'B0']]
if RUN_CROSS_FAMILY:
    cross_arms = [key for key in ['A0s','A1s','P1s','D2s','A2s','A3s'] if key in ARMS]
    for model_key in CROSS_FAMILY_MODELS:
        PLAN.append(('B0', model_key, 42))
        PLAN += [(arm, model_key, seed) for seed in CROSS_FAMILY_SEEDS for arm in cross_arms]

RUNS, started = [], time.time()
for index, (arm, model_key, seed) in enumerate(PLAN, start=1):
    print(f'\n=== [{index}/{len(PLAN)}] {model_key} {arm} seed={seed} '
          f'| elapsed {(time.time() - started) / 3600:.2f}h ===', flush=True)
    RUNS.append(load_or_run(arm, model_key, seed))

pd.DataFrame([{key: value for key, value in row.items()
               if key not in {'log_history','preprocessing'}} for row in RUNS]).to_csv(
    RESULTS / 'task13_runs.csv', index=False)
print(f'completed runs: {len(RUNS)}/{len(PLAN)} in {(time.time() - started) / 3600:.2f}h')
"""),
markdown("## 5. Master metrics and three-seed uncertainty"),
code(r"""def load_eval(record):
    return json.load(open(record['eval_json']))[0]

metric_rows = []
for run in RUNS:
    result = load_eval(run)
    for eval_relation in ['syn','hyp']:
        ntp, concept = result['ntp'][eval_relation], result['concept'][eval_relation]
        metric_rows.append({
            'model': run['model'], 'arm': run['arm'], 'label': run['label'],
            'owner': run['owner'], 'train_relation': run['relation'],
            'eval_relation': eval_relation, 'seed': run['seed'],
            'NTP NLL': ntp['ntp_nll_mean'], 'NTP PPL': ntp['ntp_ppl'],
            'NTP acc': ntp['ntp_accuracy'], 'ALT NLL': concept['alt_nll_mean'],
            'ALT mean NLL': concept['alt_mean_nll_mean'],
            'GOLD NLL': concept['gold_nll_mean'], 'SET NLL': concept['concept_nll_mean'],
            'SET mean NLL': concept['concept_mean_nll_mean'],
            'ALT mass': concept['alt_mass_mean'], 'coverage': concept['eval_slot_coverage_pct'],
        })
per_run = pd.DataFrame(metric_rows)
per_run.to_csv(RESULTS / 'task13_per_run_metrics.csv', index=False)

values = ['NTP NLL','NTP PPL','NTP acc','ALT NLL','ALT mean NLL','GOLD NLL',
          'SET NLL','SET mean NLL','ALT mass']
agg = (per_run.groupby(['model','owner','arm','label','train_relation','eval_relation'])[values]
              .agg(['mean','std','count']).reset_index())
agg.to_csv(RESULTS / 'task13_aggregate_numeric.csv', index=False)

keys = ['model','owner','arm','label','train_relation','eval_relation']
table = agg[keys].copy()
for metric in ['NTP NLL','NTP acc','ALT NLL','ALT mean NLL','GOLD NLL']:
    mean, std, count = agg[(metric,'mean')], agg[(metric,'std')], agg[(metric,'count')]
    table[metric] = [f'{m:.3f} ± {s:.3f}' if n > 1 else f'{m:.3f}'
                     for m, s, n in zip(mean, std, count)]
table['n seeds'] = agg[('NTP NLL','count')]
display(table); table.to_csv(RESULTS / 'task13_master_table.csv', index=False)
"""),
]

cells += [
markdown(r"""## 6. Loss trajectories

Raw loss values are **not comparable across objectives**: paper mean-log grows with candidate
sequence length, set marginal depends on total valid-set probability, and A3 deliberately rescales
the concept term. The first figure therefore shows within-arm raw optimization. The second divides
each run's logged total loss by its first finite value, which answers the PI's narrower question:
did each training objective actually decrease, and was the decrease stable across seeds?"""),
code(r"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

history_rows = []
for run in RUNS:
    for point in run.get('log_history', []):
        if point.get('loss') is None or point.get('epoch') is None:
            continue
        history_rows.append({
            'model': run['model'], 'arm': run['arm'], 'label': run['label'],
            'seed': run['seed'], 'epoch': float(point['epoch']),
            'total_loss': float(point['loss']),
            'clm_loss': point.get('clm_loss'),
            'weighted_clm_loss': point.get('weighted_clm_loss'),
            'concept_loss': point.get('concept_loss'),
        })
history = pd.DataFrame(history_rows)
history.to_csv(RESULTS / 'task13_loss_history.csv', index=False)
assert not history.empty, 'Trainer log history is empty; plots would be misleading.'

primary_history = history[history.model == PRIMARY_MODEL]
plot_arms = [arm for arm in ['A0s','A1s','P1s','D2s','A2s','A3s',
                             'A0h','A1h','P1h','D2h','A2h','A3h']
             if arm in set(primary_history.arm)]
ncols = 3; nrows = math.ceil(len(plot_arms) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8*nrows), squeeze=False)
for ax, arm in zip(axes.flat, plot_arms):
    subset = primary_history[primary_history.arm == arm]
    for seed, frame in subset.groupby('seed'):
        frame = frame.sort_values('epoch')
        ax.plot(frame.epoch, frame.total_loss, label=f'total s={seed}')
        if frame.concept_loss.notna().any():
            ax.plot(frame.epoch, frame.concept_loss, '--', alpha=.65,
                    label=f'concept s={seed}')
    ax.set(title=arm, xlabel='epoch', ylabel='raw logged loss')
    ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=2)
for ax in axes.flat[len(plot_arms):]: ax.axis('off')
fig.suptitle('Task 13 raw within-objective training trajectories', y=1.01)
fig.tight_layout(); fig.savefig(RESULTS / 'task13_loss_curves_raw.png', dpi=180,
                                bbox_inches='tight'); plt.show(); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
for (arm, seed), frame in primary_history.groupby(['arm','seed']):
    if arm == 'B0': continue
    frame = frame.sort_values('epoch')
    first = frame.total_loss.iloc[0]
    if np.isfinite(first) and first != 0:
        ax.plot(frame.epoch, frame.total_loss / first, label=f'{arm}, s={seed}', alpha=.8)
ax.axhline(1, color='black', linewidth=1, alpha=.4)
ax.set(xlabel='epoch', ylabel='total loss / first logged loss',
       title='Normalized optimization trajectories (shape only, not objective quality)')
ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
fig.tight_layout(); fig.savefig(RESULTS / 'task13_loss_curves_normalized.png', dpi=180)
plt.show(); plt.close(fig)
"""),
markdown("## 7. Final metric and Pareto plots"),
code(r"""OWNER_COLOR = {'paper':'#4C78A8', 'paper-equation':'#72A0C1',
               'control':'#9C9C9C', 'ours':'#F58518', 'diagnostic':'#B279A2'}

for relation in ACTIVE_RELATIONS:
    suffix = 's' if relation == 'syn' else 'h'
    arm_order = ['B0',f'A0{suffix}',f'A1{suffix}',f'P1{suffix}',
                 f'D2{suffix}',f'A2{suffix}',f'A3{suffix}']
    frame = per_run[(per_run.model == PRIMARY_MODEL) &
                    (per_run.eval_relation == relation) &
                    (per_run.arm.isin(arm_order))]
    summaries = []
    for arm in arm_order:
        rows = frame[frame.arm == arm]
        if rows.empty: continue
        summaries.append({'arm': arm, 'owner': rows.owner.iloc[0],
                          **{f'{metric}_mean': rows[metric].mean()
                             for metric in ['NTP NLL','ALT NLL','GOLD NLL']},
                          **{f'{metric}_std': rows[metric].std(ddof=1)
                             for metric in ['NTP NLL','ALT NLL','GOLD NLL']}})
    summary = pd.DataFrame(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ['NTP NLL','ALT NLL','GOLD NLL']):
        error = summary[f'{metric}_std'].fillna(0)
        ax.bar(summary.arm, summary[f'{metric}_mean'], yerr=error, capsize=3,
               color=[OWNER_COLOR.get(owner, '#777777') for owner in summary.owner])
        ax.set(title=metric + ' (lower is better)', xlabel='arm', ylabel='mean ± seed SD')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'{relation}: held-out ordinary-token and exact-sequence metrics')
    fig.tight_layout(); fig.savefig(RESULTS / f'task13_metrics_{relation}.png', dpi=180)
    plt.show(); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for row in summaries:
        ax.scatter(row['NTP NLL_mean'], row['ALT NLL_mean'], s=90,
                   color=OWNER_COLOR.get(row['owner'], '#777777'))
        ax.annotate(row['arm'], (row['NTP NLL_mean'], row['ALT NLL_mean']),
                    xytext=(5, 4), textcoords='offset points')
    ax.set(xlabel='ordinary NTP mean NLL (lower)',
           ylabel='gold-excluded alternative-set NLL (lower)',
           title=f'{relation}: retention/concept Pareto view')
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(RESULTS / f'task13_pareto_{relation}.png', dpi=180)
    plt.show(); plt.close(fig)
"""),
markdown(r"""## 8. Paired inference and claim gates

All comparisons use identical held-out `row_id`s. A negative difference means the challenger has
lower NLL. The decisive causal contrast is **A2 versus D2** because data, initialization family,
schedule, replay, and code path are matched; only the concept coefficient changes. A1 versus A0
tests the paper's augmentation intervention, but necessarily also tests lexical diversity."""),
code(r"""def concept_rows(run, relation):
    rows = load_eval(run)['concept'][relation]['per_row']
    return {str(row['row_id']): row for row in rows}

def paired_bootstrap(challenger, reference, relation, field, n_boot=10000):
    left, right = concept_rows(challenger, relation), concept_rows(reference, relation)
    assert set(left) == set(right), \
        f"paired rows differ: {challenger['tag']} vs {reference['tag']}"
    ids = sorted(left)
    delta = np.asarray([left[row_id][field] - right[row_id][field] for row_id in ids],
                       dtype=np.float64)
    assert np.isfinite(delta).all()
    rng = np.random.default_rng(13_000 + int(challenger['seed']))
    means = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        size = min(500, n_boot-start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start:start+size] = delta[indices].mean(axis=1)
    return {'n': len(delta), 'mean_delta': float(delta.mean()),
            'ci95': [float(value) for value in np.quantile(means, [.025,.975])],
            'win_rate': float(np.mean(delta < 0))}

run_index = {(run['model'], run['arm'], int(run['seed'])): run for run in RUNS}
comparison_defs = [
    ('paper_DA_vs_repeat','A1','A0'),
    ('paper_equation_vs_repeat','P1','A0'),
    ('ours_causal_vs_alpha0','A2','D2'),
    ('ours_vs_paper_DA','A2','A1'),
    ('ours_vs_paper_equation','A2','P1'),
    ('scaled_vs_unscaled','A3','A2'),
]
paired = []
for relation in ACTIVE_RELATIONS:
    suffix = 's' if relation == 'syn' else 'h'
    for name, challenger_prefix, reference_prefix in comparison_defs:
        challenger_arm = challenger_prefix + suffix
        reference_arm = reference_prefix + suffix
        for seed in SEEDS:
            left = run_index[(PRIMARY_MODEL, challenger_arm, seed)]
            right = run_index[(PRIMARY_MODEL, reference_arm, seed)]
            for field in ['nll_alternatives','nll_alternatives_mean','nll_gold']:
                paired.append({'comparison': name, 'relation': relation, 'seed': seed,
                               'field': field, 'challenger': challenger_arm,
                               'reference': reference_arm,
                               **paired_bootstrap(left, right, relation, field)})
json.dump(paired, open(RESULTS / 'task13_paired_bootstrap.json', 'w'), indent=2)
paired_table = pd.DataFrame([{**row, 'ci_low': row['ci95'][0], 'ci_high': row['ci95'][1]}
                             for row in paired])
display(paired_table); paired_table.to_csv(RESULTS / 'task13_paired_bootstrap.csv', index=False)

gate_rows = []
for (comparison, relation, field), rows in paired_table.groupby(
        ['comparison','relation','field']):
    direction_consistent = bool((rows.mean_delta < 0).all())
    independently_significant = bool((rows.ci_high < 0).all())
    gate_rows.append({'comparison': comparison, 'relation': relation, 'field': field,
                      'all_3_seeds_improve': direction_consistent,
                      'all_3_seed_CIs_below_zero': independently_significant,
                      'claim_allowed': direction_consistent and independently_significant})
claim_gate = pd.DataFrame(gate_rows)
claim_gate.to_csv(RESULTS / 'task13_claim_gate.csv', index=False)
display(claim_gate)
"""),
markdown(r"""## 9. How to interpret and stage the study

1. Run the default synonym matrix first. It contains A0, A1, P1, D2, A2, and A3 for three seeds;
   B0 is evaluation-only. Do not replace a noisy completed seed after looking at its result.
2. Inspect preprocessing coverage and loss curves before endpoint metrics. Raw loss magnitudes do
   not rank objectives.
3. Require A2 to improve alternative-only NLL over D2 on all three seeds while keeping ordinary
   NTP mean NLL within the preregistered tolerance (recommended: `+0.20`). This is the main causal
   gate for our method.
4. Then set `RUN_HYPERNYM=True`. This matters because the paper's strongest ordinary-NTP data
   augmentation result was hypernym-based; a synonym-only paper comparison is incomplete.
5. Only after a stable Llama-1B result, set `RUN_CROSS_FAMILY=True`. Falcon-RW-1B and Pythia-1.4B
   are one-seed portability checks, not confirmatory evidence. Mistral is omitted because its
   comparable official checkpoint is 7B, not roughly 1B.

`P1` is a faithful reconstruction of the paper's **written equation** at full sequence level, with
known implementation defects repaired. It is not an exact reproduction of the released trainer.
The optional `R1` diagnostic reconstructs the repaired CLM-plus-paper-loss hybrid, but should not
be mislabeled as the equation in the paper. The current paper story should lead with A2/D2, not
contrastive WordNet negatives; human-labelled rejected substitutes in the later SWORDS experiment
are a stronger place to revisit contrastive learning."""),
markdown("## 10. Storage cleanup audit"),
code(r"""# Results/Drive may contain only compact reports and figures.
for pattern in ['*.bin','*.safetensors','optimizer.pt','scheduler.pt','scaler.pt',
                'rng_state*.pth','checkpoint-*']:
    leaked = list(RESULTS.rglob(pattern))
    assert not leaked, f'weight/optimizer artifacts found in Drive: {leaked[:5]}'
print('Drive artifact audit passed. Ephemeral checkpoints were deleted after every evaluation.')
print('compact outputs:', sorted(path.name for path in RESULTS.iterdir()))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "research_tasks_13_controlled_1b_audit.ipynb",
                  "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parent / "research_tasks_13_controlled_1b_audit.ipynb"
output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"wrote {output} with {len(cells)} cells")
