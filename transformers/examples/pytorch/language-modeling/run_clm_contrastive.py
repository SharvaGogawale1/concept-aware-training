#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Task 4: Contrastive concept-aware training with hard negatives.

Trains with InfoNCE-style loss over WordNet-mined positives and hard negatives.
Requires the contrastive CSV produced by build_contrastive_dataset.py which has
columns: text, positives, negatives.

Usage:
    # First build the contrastive dataset:
    python build_contrastive_dataset.py

    # Then train:
    python run_clm_contrastive.py \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --train_file data/contrastive/youtube/contrastive_train.csv \
        --validation_file data/contrastive/youtube/contrastive_val.csv \
        --ncp_alpha 0.5 \
        --contrast_beta 1.0 \
        --block_size 128 \
        --torch_dtype bfloat16 \
        --bf16 True \
        --gradient_accumulation_steps 4 \
        --auto_find_batch_size True \
        --do_train --do_eval \
        --output_dir ./output/contrastive

    # Compare against baseline:
    python eval_ntp_baselines.py \
        --checkpoints ./output/vanilla ./output/diff_ncp ./output/contrastive \
        --validation_file data/hyp/youtube/vanilla_val.txt
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from transformers.trainer_utils import SchedulerType
from ast import literal_eval

from contrastive_trainer import ContrastiveTrainer
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
def send_example_telemetry(*args, **kwargs): pass
from transformers.utils.versions import require_version
from transformers import DataCollatorWithPadding


check_min_version("4.45.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Global lookup tables populated during tokenization
positives_lookup: dict = {}
negatives_lookup: dict = {}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Model checkpoint."})
    model_type: Optional[str] = field(default=None, metadata={"help": "Model type: " + ", ".join(MODEL_TYPES)})
    config_overrides: Optional[str] = field(default=None, metadata={"help": "Override config."})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache dir."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Fast tokenizer."})
    model_revision: str = field(default="main", metadata={"help": "Model version."})
    token: str = field(default=None, metadata={"help": "HF auth token."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override dtype.", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    low_cpu_mem_usage: bool = field(default=False, metadata={"help": "Low-memory loading."})
    ncp_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight of the positive-only differentiable NCP loss term."},
    )
    contrast_beta: float = field(
        default=1.0,
        metadata={"help": "Weight of the InfoNCE contrastive loss term."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides conflicts with --config_name / --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF dataset name."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset config."})
    train_file: Optional[str] = field(default=None, metadata={"help": "Contrastive train CSV."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Contrastive val CSV."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate eval."})
    streaming: bool = field(default=False, metadata={"help": "Streaming mode."})
    block_size: Optional[int] = field(default=None, metadata={"help": "Max sequence length."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cache."})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "Val split %."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Workers."})
    keep_linebreaks: bool = field(default=True, metadata={"help": "Keep linebreaks."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need dataset_name or train_file/validation_file.")
        if self.train_file:
            assert self.train_file.split(".")[-1] in ["csv", "json", "txt"]
        if self.validation_file:
            assert self.validation_file.split(".")[-1] in ["csv", "json", "txt"]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.eval_strategy = "steps"
    training_args.eval_steps = 500
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    training_args.lr_scheduler_type = SchedulerType.COSINE
    training_args.overwrite_output_dir = True

    send_example_telemetry("run_clm_contrastive", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output dir ({training_args.output_dir}) is non-empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from {last_checkpoint}.")

    set_seed(training_args.seed)

    # ── Dataset loading ──────────────────────────────────────────────────────
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            cache_dir=model_args.cache_dir, token=model_args.token,
            streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
        extension = (data_args.train_file or data_args.validation_file).split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir,
            token=model_args.token, **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, token=model_args.token, **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension, data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, token=model_args.token, **dataset_args,
            )

    # ── Model + tokenizer ────────────────────────────────────────────────────
    config_kwargs = {
        "cache_dir": model_args.cache_dir, "revision": model_args.model_revision,
        "token": model_args.token, "trust_remote_code": model_args.trust_remote_code,
    }
    config = (
        AutoConfig.from_pretrained(model_args.config_name or model_args.model_name_or_path, **config_kwargs)
        if model_args.config_name or model_args.model_name_or_path
        else CONFIG_MAPPING[model_args.model_type]()
    )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision, "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision,
            token=model_args.token, trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # ── Tokenization ─────────────────────────────────────────────────────────
    split = "train" if training_args.do_train else "validation"
    column_names = list(raw_datasets[split].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        if not examples.get(text_column_name):
            return {"input_ids": [], "attention_mask": []}
        with CaptureLogger(tok_logger) as cl:
            ids = tokenizer(
                str(examples[text_column_name]),
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=128,
            )["input_ids"].squeeze().tolist()

        if isinstance(ids, int):
            ids = [128000, ids, 128001]
        else:
            ids = [128000] + ids + [128001]
        key = str(ids)

        # Parse positives and negatives from the contrastive CSV
        pos_list, neg_list = [], []
        if "positives" in examples:
            try:
                pos_list = literal_eval(str(examples["positives"]))
                pos_list = [str(p).strip().lstrip("\n") for p in pos_list if str(p).strip()]
            except Exception:
                pass
        if "negatives" in examples:
            try:
                neg_list = literal_eval(str(examples["negatives"]))
                neg_list = [str(n).strip().lstrip("\n") for n in neg_list if str(n).strip()]
            except Exception:
                pass

        # Fall back to context_syn column if no positives column (backward compatibility)
        if not pos_list and "context_syn" in examples:
            try:
                pos_list = literal_eval(str(examples["context_syn"]))
            except Exception:
                pass

        positives_lookup[key] = pos_list
        negatives_lookup[key] = neg_list

        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning("Please ignore warning above — long input will be chunked.")

        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=False)

    max_pos_embeddings = getattr(config, "max_position_embeddings", 1024)
    block_size = (
        min(tokenizer.model_max_length, max_pos_embeddings if max_pos_embeddings > 0 else 1024)
        if data_args.block_size is None
        else min(data_args.block_size, tokenizer.model_max_length)
    )

    def group_texts(examples):
        examples = {k: v for k, v in examples.items() if k in {"input_ids", "attention_mask"}}
        ids = examples.get("input_ids", [])
        mask = examples.get("attention_mask", [])
        if not ids or not mask:
            return {}
        if len(ids) < block_size:
            pad_len = block_size - len(ids)
            ids += [tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
        else:
            ids, mask = ids[:block_size], mask[:block_size]
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.copy()}

    with training_args.main_process_first(desc="grouping texts"):
        lm_datasets = (
            tokenized_datasets
            if not data_args.streaming
            else tokenized_datasets.map(group_texts, batched=False)
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset.")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        positives_lookup=positives_lookup,
        negatives_lookup=negatives_lookup,
        alpha=model_args.ncp_alpha,
        beta=model_args.contrast_beta,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = min(data_args.max_train_samples or len(train_dataset), len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = min(data_args.max_eval_samples or len(eval_dataset), len(eval_dataset))
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
