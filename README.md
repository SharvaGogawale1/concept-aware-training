# Concept-Aware Training

This repository contains code for further training language models using concept-aware training objectives, including experiments with synonym and hypernym datasets.

## Setup

### 1. Download a base model

First, download the model that you would like to further train. This can be any Hugging Face-compatible causal language model.

The path to this model will be passed into the training command using:

```bash
--model_name_or_path /path/to/your/model
```

For example:

```bash
--model_name_or_path /juice2/scr2/laya/Meta-Llama-3-8B
```

### 2. Create the Conda environment

Create a Conda environment using the provided environment/spec file:

```bash
conda create --name new_env --file requirements.txt
```

Then activate the environment:

```bash
conda activate new_env
```

---

## Running Training

After setting up the environment, go into the language modeling example directory:

```bash
cd transformers/examples/pytorch/language-modeling/
```

From this directory, run `run_clm.py` with the appropriate model path, training file, validation file, and output directory.

Example command:

```bash
python3 run_clm.py \
  --model_name_or_path /juice2/scr2/laya/Meta-Llama-3-8B \
  --train_file /juice2/u/laya/scripts25/hyp_final/youtube/context_syn_train.txt \
  --validation_file /juice2/u/laya/scripts25/hyp_final/youtube/context_syn_val.txt \
  --save_total_limit 1 \
  --gradient_accumulation_steps 4 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --block_size 128 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./youtube_hyp/context
```

### Important Arguments

- `--model_name_or_path`: Path to the base model you want to further train.
- `--train_file`: Path to the training dataset.
- `--validation_file`: Path to the validation dataset.
- `--output_dir`: Directory where checkpoints and outputs will be saved.
- `--save_total_limit`: Maximum number of checkpoints to keep.
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating model weights.
- `--torch_dtype`: Data type used for model weights.
- `--bf16 True`: Uses bfloat16 training, if supported by the hardware.
- `--block_size`: Maximum sequence length used during training.
- `--auto_find_batch_size True`: Automatically adjusts batch size if memory issues occur.
- `--do_train`: Enables training.
- `--do_eval`: Enables evaluation on the validation set.

## Custom Loss Training Options

In addition to the standard `run_clm.py` script, this repository includes two additional `run_clm` options that use a custom loss function:

```bash
run_clm_hyp_custom_loss.py
run_clm_syn_custom_loss.py
```

These scripts are intended for experiments using concept-aware custom loss functions. You can read more about the NCP loss function in Section 2.2.2, **NCP Loss Function**, of the paper:

[https://arxiv.org/pdf/2601.11791](https://arxiv.org/pdf/2601.11791)

The relevant supporting files include:

```bash
custom_trainer.py
hierarchical_trainer.py
```

## Datasets

This repository contains concept-aware datasets under the `data/` directory.

### Synonym Datasets

Synonym datasets are located in:

```bash
data/syn
```

### Hypernym Datasets

Hypernym datasets are located in:

```bash
data/hyp
```

---

## Experimental Extensions

The following four extensions are implemented as part of ongoing research into differentiable and contrastive concept-aware training.

---

### Task 1: NTP Baseline Evaluation

**Script:** `eval_ntp_baselines.py`

Evaluates multiple checkpoints (vanilla CLM, synonym NCP, hypernym NCP) using **standard next-token prediction loss on vanilla validation data**. This separates the evaluation protocol from the training objective, allowing apples-to-apples comparison across methods.

#### Run

```bash
cd transformers/examples/pytorch/language-modeling/

python eval_ntp_baselines.py \
  --checkpoints \
    /path/to/vanilla_checkpoint \
    /path/to/syn_ncp_checkpoint \
    /path/to/hyp_ncp_checkpoint \
  --validation_file ../../../data/hyp/youtube/vanilla_val.txt \
  --results_json ntp_results.json
```

Output: a comparison table of `eval_loss`, `perplexity`, and `accuracy` for each checkpoint, saved to `ntp_results.json`.

#### In Colab

```python
!cd /content/concept-aware-training/transformers/examples/pytorch/language-modeling && \
  python eval_ntp_baselines.py \
    --checkpoints /content/vanilla_ckpt /content/syn_ckpt /content/hyp_ckpt \
    --validation_file /content/concept-aware-training/data/hyp/youtube/vanilla_val.txt \
    --results_json /content/ntp_results.json

import json
results = json.load(open("/content/ntp_results.json"))
for r in results:
    print(r)
```

---

### Task 2: Differentiable NCP Training

**Trainer:** `differentiable_ncp_trainer.py`  
**Script:** `run_clm_differentiable_ncp.py`

Replaces the original NCP loss (which runs N separate `torch.no_grad()` forward passes per concept set) with a **single-pass differentiable implementation**. Gradients flow fully through the concept-set log-sum-exp term:

```
L = L_CLM + alpha * mean( -log sum_{c in C} p(c | context) )
```

This is the correct marginal likelihood objective for a set-valued label, and it is **computationally cheaper** than the original (0 extra forward passes vs. N).

#### Run

```bash
cd transformers/examples/pytorch/language-modeling/

python run_clm_differentiable_ncp.py \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --train_file ../../../data/syn/youtube/context_loss_train.csv \
  --validation_file ../../../data/syn/youtube/context_loss_val.csv \
  --ncp_alpha 1.0 \
  --block_size 128 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --gradient_accumulation_steps 4 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output/diff_ncp_syn
```

For hypernym data:

```bash
python run_clm_differentiable_ncp.py \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --train_file ../../../data/hyp/youtube/context_loss_train.csv \
  --validation_file ../../../data/hyp/youtube/context_loss_val.csv \
  --ncp_alpha 1.0 \
  --block_size 128 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --gradient_accumulation_steps 4 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output/diff_ncp_hyp
```

**Key argument:** `--ncp_alpha` (default 1.0) — weight applied to the NCP loss term.

#### In Colab

```python
!cd /content/concept-aware-training/transformers/examples/pytorch/language-modeling && \
  python run_clm_differentiable_ncp.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --train_file /content/concept-aware-training/data/syn/youtube/context_loss_train.csv \
    --validation_file /content/concept-aware-training/data/syn/youtube/context_loss_val.csv \
    --ncp_alpha 1.0 \
    --block_size 128 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --gradient_accumulation_steps 4 \
    --auto_find_batch_size True \
    --do_train --do_eval \
    --output_dir /content/diff_ncp_output
```

---

### Task 3: Build Contrastive Dataset with Hard Negatives

**Script:** `build_contrastive_dataset.py` (at repository root)

Builds a YouTube-only contrastive dataset by mining **hard negatives** from WordNet for each positive concept set. Hard negatives are semantically plausible but wrong completions — they teach the model genuine conceptual boundaries.

**Negative types mined (in priority order):**
1. **Co-hyponyms** — siblings in the WordNet hierarchy (same parent, different subtree)
2. **Wrong-sense distractors** — other synsets of the same word (different semantic sense)
3. **Same-POS fallback** — other WordNet words with the same part-of-speech

**Output columns:** `text`, `positives`, `negatives`

#### Setup

```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### Run

```bash
# From repository root
python build_contrastive_dataset.py \
  --source both \
  --max_negatives 10 \
  --output_dir data/contrastive/youtube
```

This produces:
```
data/contrastive/youtube/syn_contrastive_train.csv
data/contrastive/youtube/syn_contrastive_val.csv
data/contrastive/youtube/hyp_contrastive_train.csv
data/contrastive/youtube/hyp_contrastive_val.csv
data/contrastive/youtube/contrastive_train.csv   ← merged (use this for training)
data/contrastive/youtube/contrastive_val.csv
```

#### In Colab

```python
!pip install nltk -q
!python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

!cd /content/concept-aware-training && \
  python build_contrastive_dataset.py \
    --source both \
    --max_negatives 10 \
    --output_dir data/contrastive/youtube

import pandas as pd
df = pd.read_csv("/content/concept-aware-training/data/contrastive/youtube/contrastive_train.csv")
print(df.head())
print(f"Total rows: {len(df)}")
```

---

### Task 4: Contrastive Training

**Trainer:** `contrastive_trainer.py`  
**Script:** `run_clm_contrastive.py`

Trains with an **InfoNCE-style contrastive loss** using both positive concept sets and hard negatives. The loss pushes the model to assign high probability to any valid concept and low probability to hard negatives, all computed from the existing base forward pass (zero extra compute):

```
L = L_CLM
  + alpha * mean( -log sum_{c+ in C+} p(c+ | context) )     [differentiable NCP]
  + beta  * mean( InfoNCE([log_p_pos, log_p_neg]) )          [contrastive]
```

Where `InfoNCE = log(p_pos + p_neg) - log_p_pos` — fully differentiable.

**Requires:** the contrastive CSV from Task 3.

#### Run

```bash
# Step 1: build dataset (see Task 3)
python build_contrastive_dataset.py --source both --output_dir data/contrastive/youtube

# Step 2: train
cd transformers/examples/pytorch/language-modeling/

python run_clm_contrastive.py \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --train_file ../../../data/contrastive/youtube/contrastive_train.csv \
  --validation_file ../../../data/contrastive/youtube/contrastive_val.csv \
  --ncp_alpha 0.5 \
  --contrast_beta 1.0 \
  --block_size 128 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --gradient_accumulation_steps 4 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output/contrastive
```

**Key arguments:**
- `--ncp_alpha` (default 0.5) — weight of the positive-only NCP term
- `--contrast_beta` (default 1.0) — weight of the InfoNCE contrastive term

#### Compare all methods

After training all variants, evaluate them on the same vanilla validation data:

```bash
cd transformers/examples/pytorch/language-modeling/

python eval_ntp_baselines.py \
  --checkpoints \
    ./output/vanilla \
    ./output/syn_ncp \
    ./output/hyp_ncp \
    ./output/diff_ncp_syn \
    ./output/contrastive \
  --validation_file ../../../data/hyp/youtube/vanilla_val.txt \
  --results_json comparison_results.json
```

#### In Colab

```python
# Build dataset
!cd /content/concept-aware-training && \
  python build_contrastive_dataset.py \
    --source both \
    --output_dir data/contrastive/youtube

# Train contrastive model
!cd /content/concept-aware-training/transformers/examples/pytorch/language-modeling && \
  python run_clm_contrastive.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --train_file /content/concept-aware-training/data/contrastive/youtube/contrastive_train.csv \
    --validation_file /content/concept-aware-training/data/contrastive/youtube/contrastive_val.csv \
    --ncp_alpha 0.5 \
    --contrast_beta 1.0 \
    --block_size 128 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --gradient_accumulation_steps 4 \
    --auto_find_batch_size True \
    --do_train --do_eval \
    --output_dir /content/contrastive_output

# Compare all models on vanilla NTP eval
!cd /content/concept-aware-training/transformers/examples/pytorch/language-modeling && \
  python eval_ntp_baselines.py \
    --checkpoints \
      /content/vanilla_output \
      /content/diff_ncp_output \
      /content/contrastive_output \
    --validation_file /content/concept-aware-training/data/hyp/youtube/vanilla_val.txt \
    --results_json /content/comparison_results.json

import json, pandas as pd
results = json.load(open("/content/comparison_results.json"))
pd.DataFrame(results)
```

---

## Example Workflow

A typical workflow is:

```bash
# 1. Create and activate environment
conda create --name new_env --file spec-file.txt
conda activate new_env

# 2. Move into the training directory
cd transformers/examples/pytorch/language-modeling/

# 3. Run training (standard CLM baseline)
python3 run_clm.py \
  --model_name_or_path /path/to/model \
  --train_file /path/to/train.txt \
  --validation_file /path/to/validation.txt \
  --save_total_limit 1 \
  --gradient_accumulation_steps 4 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --block_size 128 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output_directory
```

To run one of the custom loss versions, replace `run_clm.py` with the appropriate script.

For synonym-based concept-aware training:

```bash
python3 run_clm_syn_custom_loss.py \
  --model_name_or_path /path/to/model \
  --train_file /path/to/train.txt \
  --validation_file /path/to/validation.txt \
  --save_total_limit 1 \
  --gradient_accumulation_steps 4 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --block_size 128 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output_directory
```

For hypernym-based concept-aware training:

```bash
python3 run_clm_hyp_custom_loss.py \
  --model_name_or_path /path/to/model \
  --train_file /path/to/train.txt \
  --validation_file /path/to/validation.txt \
  --save_total_limit 1 \
  --gradient_accumulation_steps 4 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --block_size 128 \
  --auto_find_batch_size True \
  --do_train \
  --do_eval \
  --output_dir ./output_directory
```

## Notes

Make sure the model path, dataset paths, and output directory are updated for your own setup before running training.

Large model checkpoints, output folders, and experiment logs should generally not be committed to the repository unless they are intentionally part of the release.
