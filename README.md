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

This repository contains concept-aware datasets under the `datasets/` directory.

### Synonym Datasets

Synonym datasets are located in:

```bash
datasets/syn
```

These datasets are used for experiments where multiple words or expressions correspond to similar or equivalent concepts.

### Hypernym Datasets

Hypernym datasets are located in:

```bash
datasets/hyp
```

These datasets are used for experiments involving hierarchical concept relationships, where one term is a broader category of another term.

## Example Workflow

A typical workflow is:

```bash
# 1. Create and activate environment
conda create --name new_env --file spec-file.txt
conda activate new_env

# 2. Move into the training directory
cd transformers/examples/pytorch/language-modeling/

# 3. Run training
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
