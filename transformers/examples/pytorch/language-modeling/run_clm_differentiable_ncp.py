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
Task 2: Differentiable NCP training.

Trains with a fully differentiable concept-set marginal likelihood loss.
Unlike run_clm_syn_custom_loss.py (which uses N no_grad forward passes per step),
this script computes the concept-set log-sum-exp from the EXISTING base forward
pass in one shot, keeping gradients intact throughout.

Usage:
    python run_clm_differentiable_ncp.py \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --train_file /path/to/context_loss_train.csv \
        --validation_file /path/to/context_loss_val.csv \
        --ncp_alpha 1.0 \
        --block_size 128 \
        --torch_dtype bfloat16 \
        --bf16 True \
        --gradient_accumulation_steps 4 \
        --auto_find_batch_size True \
        --do_train --do_eval \
        --output_dir ./output/diff_ncp
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

from differentiable_ncp_trainer import DifferentiableNCPTrainer
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
    IntervalStrategy,
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

completions_lookup = {}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint for weights initialization."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type from: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(default=None, metadata={"help": "Override config settings."})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HuggingFace cache dir."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer."})
    model_revision: str = field(default="main", metadata={"help": "Model version."})
    token: str = field(default=None, metadata={"help": "HuggingFace auth token."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override default dtype.", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    low_cpu_mem_usage: bool = field(default=False, metadata={"help": "Low-memory model loading."})
    ncp_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight of the differentiable NCP loss term (alpha * NCP added to CLM loss)."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HuggingFace dataset name."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset config."})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training CSV/TXT file."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation CSV/TXT file."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training set."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate eval set."})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming."})
    block_size: Optional[int] = field(default=None, metadata={"help": "Max sequence length."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite dataset cache."})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "Val split %."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Preprocessing workers."})
    keep_linebreaks: bool = field(default=True, metadata={"help": "Keep linebreaks in TXT files."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "Streaming requires datasets>=2.0.0")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need a dataset name or a training/validation file.")
        if self.train_file is not None:
            assert self.train_file.split(".")[-1] in ["csv", "json", "txt"]
        if self.validation_file is not None:
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

    send_example_telemetry("run_clm_differentiable_ncp", model_args, data_args)

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
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output dir ({training_args.output_dir}) exists and is non-empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from checkpoint {last_checkpoint}.")

    set_seed(training_args.seed)

    # ── Dataset loading ──────────────────────────────────────────────────────
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir, token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir, token=model_args.token,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
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

    # ── Model + tokenizer ───────────────────────────────────────────────────
    config_kwargs = {
        "cache_dir": model_args.cache_dir, "revision": model_args.model_revision,
        "token": model_args.token, "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision, "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Need --tokenizer_name or --model_name_or_path.")

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config, cache_dir=model_args.cache_dir,
            revision=model_args.model_revision, token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # ── Tokenization ────────────────────────────────────────────────────────
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    syn_column_name = None
    if "context_syn" in column_names:
        syn_column_name = "context_syn"
    elif "dict_syn" in column_names:
        syn_column_name = "dict_syn"
    else:
        logger.warning("No synonym column found — NCP loss will not be applied.")

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
        input_key = str(ids)

        completions_list = []
        if syn_column_name and syn_column_name in examples:
            try:
                completions_list = literal_eval(examples[syn_column_name])
            except Exception as exc:
                logger.warning(f"Failed to parse {syn_column_name}: {exc}")
        completions_lookup[input_key] = completions_list

        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning("Please ignore the warning above — long input will be chunked.")

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
    if data_args.block_size is None:
        block_size = min(tokenizer.model_max_length, max_pos_embeddings if max_pos_embeddings > 0 else 1024)
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        examples = {k: v for k, v in examples.items() if k in {"input_ids", "attention_mask"}}
        input_ids = examples["input_ids"]
        attention_mask = examples.get("attention_mask")
        if not input_ids or not attention_mask:
            return {}
        if len(input_ids) < block_size:
            pad_len = block_size - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[:block_size]
            attention_mask = attention_mask[:block_size]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.copy()}

    with training_args.main_process_first(desc="grouping texts"):
        lm_datasets = tokenized_datasets if not data_args.streaming else tokenized_datasets.map(group_texts, batched=False)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
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

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = DifferentiableNCPTrainer(
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
        completions_lookup=completions_lookup,
        alpha=model_args.ncp_alpha,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = min(
            data_args.max_train_samples or len(train_dataset), len(train_dataset)
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = min(
            data_args.max_eval_samples or len(eval_dataset), len(eval_dataset)
        )
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
