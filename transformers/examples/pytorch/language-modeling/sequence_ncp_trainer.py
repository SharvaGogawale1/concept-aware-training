"""Exact sequence-level concept objectives for causal language models.

This module deliberately keeps concept metadata inside each dataset example.
It does not build global lookup dictionaries during ``Dataset.map``.  As a
result, Hugging Face cache hits, duplicate contexts, multiprocessing, and row
ordering cannot silently remove or overwrite concept supervision.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer


MASK_TOKEN = "<mask>"
# Concept objectives over a candidate set C of size n, with p_c the exact teacher-forced sequence
# probability of candidate c.  The first two are the published forms; the next two are size-
# normalized variants.  See grouped_concept_loss for the algebra and the gradient caveat.
#
#   paper_mean          -(1/n) sum_c log p_c        negative log GEOMETRIC mean of the set
#   set_marginal        -log sum_c p_c              the set marginal
#   set_marginal_mean   -log((1/n) sum_c p_c)       negative log ARITHMETIC mean of the set
#   set_marginal_scaled -(1/n) log sum_c p_c        the marginal, down-weighted by set size
#   none                no concept term (the alpha=0 control)
OBJECTIVES = {"none", "paper_mean", "set_marginal", "set_marginal_mean", "set_marginal_scaled"}


def parse_candidate_list(value: Any) -> List[str]:
    """Parse and normalize a serialized candidate list without changing case."""
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    result: List[str] = []
    for item in raw:
        candidate = str(item).strip()
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def _flat_ids(tokenizer: Any, text: str, add_special_tokens: bool = True) -> List[int]:
    ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _shared_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


def _left_truncate_preserving_bos(
    full_ids: Sequence[int], candidate_start: int, block_size: int, bos_token_id: Optional[int]
) -> Optional[Tuple[List[int], int, int]]:
    """Left-truncate context while retaining every candidate token.

    Returns ``(ids, new_candidate_start, dropped_context_tokens)``.  ``None``
    means that the candidate itself cannot fit inside ``block_size``.
    """
    ids = list(full_ids)
    if len(ids) <= block_size:
        return ids, candidate_start, 0

    candidate = ids[candidate_start:]
    if len(candidate) >= block_size:
        return None

    context = ids[:candidate_start]
    context_budget = block_size - len(candidate)
    had_bos = bool(context) and bos_token_id is not None and context[0] == bos_token_id
    if had_bos:
        tail_budget = max(context_budget - 1, 0)
        kept_context = [context[0]] + (context[1:][-tail_budget:] if tail_budget else [])
    else:
        kept_context = context[-context_budget:]
    dropped = len(context) - len(kept_context)
    return kept_context + candidate, len(kept_context), dropped


def encode_candidate_continuation(
    tokenizer: Any, context: str, candidate: str, block_size: int
) -> Optional[Dict[str, Any]]:
    """Encode ``candidate`` exactly as an in-context continuation.

    The returned labels are ``-100`` on the context and token ids on every
    candidate continuation token.  This is the single source of truth used by
    both training and v3 evaluation.
    """
    clean_context = str(context).rstrip()
    clean_candidate = str(candidate).strip()
    if not clean_context or not clean_candidate or MASK_TOKEN in clean_context:
        return None

    context_ids = _flat_ids(tokenizer, clean_context, add_special_tokens=True)
    full_ids = _flat_ids(tokenizer, clean_context + " " + clean_candidate, add_special_tokens=True)
    candidate_start = _shared_prefix_length(context_ids, full_ids)
    if candidate_start == 0 or candidate_start >= len(full_ids):
        return None

    truncated = _left_truncate_preserving_bos(
        full_ids, candidate_start, block_size, getattr(tokenizer, "bos_token_id", None)
    )
    if truncated is None:
        return None
    full_ids, candidate_start, n_left_truncated = truncated
    continuation_ids = full_ids[candidate_start:]
    if not continuation_ids:
        return None
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * candidate_start + continuation_ids,
        "continuation_ids": continuation_ids,
        "candidate_start": candidate_start,
        "n_left_truncated": n_left_truncated,
    }


def has_strict_prefix_collision(sequences: Sequence[Sequence[int]]) -> bool:
    tuples = [tuple(seq) for seq in sequences]
    for i, left in enumerate(tuples):
        for j, right in enumerate(tuples):
            if i != j and len(left) < len(right) and right[: len(left)] == left:
                return True
    return False


def _encode_base_text(tokenizer: Any, text: str, block_size: int) -> Optional[List[int]]:
    ids = _flat_ids(tokenizer, str(text).rstrip(), add_special_tokens=True)
    if len(ids) <= block_size:
        return ids
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if ids and bos_id is not None and ids[0] == bos_id:
        return [ids[0]] + ids[-(block_size - 1) :]
    return ids[-block_size:]


def tokenize_concept_record(
    record: Dict[str, Any],
    tokenizer: Any,
    block_size: int = 128,
    candidate_column: str = "context_syn",
    negative_column: str = "negatives",
    gold_column: str = "gold_surface",
    require_gold_for_concept: bool = True,
) -> Dict[str, Any]:
    """Return all model and concept fields for one row; never mutates globals.

    Concept rows use teacher-forced supervision only on the observed gold continuation.  Replay
    rows (which have no candidates) retain ordinary full-sequence CLM labels.  This distinction is
    load-bearing: labeling the context prefix on a concept row does *not* provide the matched
    ``-log p(gold | context)`` control needed for an NTP-vs-NCP comparison.
    """
    context = str(record.get("text", ""))
    row_id = str(record.get("row_id", ""))
    candidates = parse_candidate_list(record.get(candidate_column, []))
    negatives = parse_candidate_list(record.get(negative_column, []))
    gold_surface = str(record.get(gold_column, "") or "").strip()

    result: Dict[str, Any] = {
        "row_id": row_id,
        "keep": True,
        "drop_reason": "",
        "eligible": int(bool(candidates)),
        "gold_surface": gold_surface,
        "gold_token_ids": [],
        "base_supervision": "",
        "candidate_strings": [],
        "candidate_input_ids": [],
        "candidate_labels": [],
        "candidate_token_ids": [],
        "candidate_lengths": [],
        "negative_strings": [],
        "negative_input_ids": [],
        "negative_labels": [],
        "n_left_truncated": 0,
    }
    if MASK_TOKEN in context:
        result.update(keep=False, drop_reason="mask_row")
        return result

    if not candidates:  # replay/CLM-only row
        base_ids = _encode_base_text(tokenizer, context, block_size)
        if not base_ids or len(base_ids) < 2:
            result.update(keep=False, drop_reason="empty_or_unscoreable_context")
            return result
        result.update(
            input_ids=base_ids,
            attention_mask=[1] * len(base_ids),
            labels=base_ids.copy(),
            base_supervision="full_sequence_clm",
        )
        return result

    if require_gold_for_concept and not gold_surface:
        result.update(keep=False, drop_reason="missing_gold_surface")
        return result
    if gold_surface:
        gold_encoded = encode_candidate_continuation(tokenizer, context, gold_surface, block_size)
        if gold_encoded is None:
            result.update(keep=False, drop_reason="unscoreable_gold_surface")
            return result
        result.update(
            input_ids=gold_encoded["input_ids"],
            attention_mask=gold_encoded["attention_mask"],
            labels=gold_encoded["labels"],
            gold_token_ids=gold_encoded["continuation_ids"],
            base_supervision="gold_continuation_ntp",
            n_left_truncated=gold_encoded["n_left_truncated"],
        )
    else:
        # Compatibility mode is deliberately opt-in through ``require_gold_for_concept=False``.
        base_ids = _encode_base_text(tokenizer, context, block_size)
        if not base_ids or len(base_ids) < 2:
            result.update(keep=False, drop_reason="empty_or_unscoreable_context")
            return result
        result.update(
            input_ids=base_ids,
            attention_mask=[1] * len(base_ids),
            labels=base_ids.copy(),
            base_supervision="legacy_prefix_clm",
        )

    seen = set()
    encoded_rows = []
    for candidate in candidates:
        encoded = encode_candidate_continuation(tokenizer, context, candidate, block_size)
        if encoded is None:
            continue
        key = tuple(encoded["continuation_ids"])
        if key in seen:
            continue
        seen.add(key)
        encoded_rows.append((candidate, encoded))

    if not encoded_rows:
        result.update(keep=False, drop_reason="no_scoreable_candidates")
        return result
    continuation_sequences = [encoded["continuation_ids"] for _, encoded in encoded_rows]
    if has_strict_prefix_collision(continuation_sequences):
        result.update(keep=False, drop_reason="strict_token_prefix_collision")
        return result

    # Negatives are encoded exactly like candidates and must not collide with them after
    # tokenization -- a "negative" that shares a token sequence with a positive would be trained
    # both up and down at once.
    positive_keys = {tuple(sequence) for sequence in continuation_sequences}
    negative_seen = set()
    for negative in negatives:
        item = encode_candidate_continuation(tokenizer, context, negative, block_size)
        if item is None:
            continue
        key = tuple(item["continuation_ids"])
        if key in positive_keys or key in negative_seen:
            continue
        negative_seen.add(key)
        result["negative_strings"].append(negative)
        result["negative_input_ids"].append(item["input_ids"])
        result["negative_labels"].append(item["labels"])

    result["candidate_strings"] = [candidate for candidate, _ in encoded_rows]
    result["candidate_input_ids"] = [encoded["input_ids"] for _, encoded in encoded_rows]
    result["candidate_labels"] = [encoded["labels"] for _, encoded in encoded_rows]
    result["candidate_token_ids"] = continuation_sequences
    result["candidate_lengths"] = [len(tokens) for tokens in continuation_sequences]
    result["n_left_truncated"] = max(
        [result["n_left_truncated"]]
        + [encoded["n_left_truncated"] for _, encoded in encoded_rows]
    )
    return result


@dataclass
class SequenceNCPDataCollator:
    """Pad base sequences and flatten variable-sized candidate sets."""

    pad_token_id: int

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
        width = max((len(row) for row in rows), default=1)
        return torch.tensor([list(row) + [pad_value] * (width - len(row)) for row in rows], dtype=torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        base_ids = [feature["input_ids"] for feature in features]
        base_masks = [feature["attention_mask"] for feature in features]
        base_labels = [feature["labels"] for feature in features]

        candidate_ids: List[List[int]] = []
        candidate_labels: List[List[int]] = []
        group_ids: List[int] = []
        negative_ids: List[List[int]] = []
        negative_labels: List[List[int]] = []
        negative_group_ids: List[int] = []
        for group_id, feature in enumerate(features):
            rows = feature.get("candidate_input_ids", [])
            labels = feature.get("candidate_labels", [])
            if len(rows) != len(labels):
                raise ValueError(f"candidate ids/labels mismatch for row {feature.get('row_id')}")
            candidate_ids.extend(rows)
            candidate_labels.extend(labels)
            group_ids.extend([group_id] * len(rows))

            neg_rows = feature.get("negative_input_ids", []) or []
            neg_labels = feature.get("negative_labels", []) or []
            if len(neg_rows) != len(neg_labels):
                raise ValueError(f"negative ids/labels mismatch for row {feature.get('row_id')}")
            negative_ids.extend(neg_rows)
            negative_labels.extend(neg_labels)
            negative_group_ids.extend([group_id] * len(neg_rows))

        batch: Dict[str, Any] = {
            "input_ids": self._pad(base_ids, self.pad_token_id),
            "attention_mask": self._pad(base_masks, 0),
            "labels": self._pad(base_labels, -100),
            "concept_eligible_count": torch.tensor(sum(int(f.get("eligible", 0)) for f in features)),
            "row_ids": [str(feature.get("row_id", "")) for feature in features],
        }
        if candidate_ids:
            batch.update(
                candidate_input_ids=self._pad(candidate_ids, self.pad_token_id),
                candidate_attention_mask=self._pad([[1] * len(row) for row in candidate_ids], 0),
                candidate_labels=self._pad(candidate_labels, -100),
                candidate_group_ids=torch.tensor(group_ids, dtype=torch.long),
            )
        else:
            batch.update(
                candidate_input_ids=torch.empty((0, 1), dtype=torch.long),
                candidate_attention_mask=torch.empty((0, 1), dtype=torch.long),
                candidate_labels=torch.empty((0, 1), dtype=torch.long),
                candidate_group_ids=torch.empty((0,), dtype=torch.long),
            )
        if negative_ids:
            batch.update(
                negative_input_ids=self._pad(negative_ids, self.pad_token_id),
                negative_attention_mask=self._pad([[1] * len(row) for row in negative_ids], 0),
                negative_labels=self._pad(negative_labels, -100),
                negative_group_ids=torch.tensor(negative_group_ids, dtype=torch.long),
            )
        else:
            batch.update(
                negative_input_ids=torch.empty((0, 1), dtype=torch.long),
                negative_attention_mask=torch.empty((0, 1), dtype=torch.long),
                negative_labels=torch.empty((0, 1), dtype=torch.long),
                negative_group_ids=torch.empty((0,), dtype=torch.long),
            )
        return batch


def sequence_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Teacher-forced sequence log-probabilities for labels masked with -100."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,L,V] and labels [B,L]")
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * valid).sum(dim=-1)


def grouped_concept_loss(
    sequence_log_probs: torch.Tensor, group_ids: torch.Tensor, objective: str
) -> Tuple[torch.Tensor, int]:
    """Per-slot concept loss, averaged over slots.

    ``set_marginal_mean`` differs from ``set_marginal`` by ``+log n``.  For a fixed candidate set
    that term is constant in the parameters, so the two produce **identical gradients** and train
    to the same model; the normalization changes only the reported loss value, where it removes
    the bias that makes a larger candidate set look better for free.  ``set_marginal_scaled``
    divides by ``n`` rather than adding ``log n``, which *does* change the gradient: it
    down-weights slots with many candidates relative to slots with few.
    """
    if objective not in OBJECTIVES - {"none"}:
        raise ValueError(f"unsupported objective: {objective}")
    losses = []
    for group_id in torch.unique(group_ids, sorted=True):
        values = sequence_log_probs[group_ids == group_id]
        size = values.numel()
        if objective == "paper_mean":
            losses.append(-values.mean())
        elif objective == "set_marginal":
            losses.append(-torch.logsumexp(values, dim=0))
        elif objective == "set_marginal_mean":
            losses.append(-torch.logsumexp(values, dim=0) + math.log(size))
        else:  # set_marginal_scaled
            losses.append(-torch.logsumexp(values, dim=0) / size)
    if not losses:
        return sequence_log_probs.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def grouped_infonce_loss(
    positive_log_probs: torch.Tensor,
    positive_groups: torch.Tensor,
    negative_log_probs: torch.Tensor,
    negative_groups: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """InfoNCE over exact sequence probabilities: ``-log( sum_pos p / sum_(pos+neg) p )``.

    Equivalently ``logsumexp(all) - logsumexp(pos)``, which is >= 0 and reaches 0 only when the
    negatives carry no mass.  Slots with no mined negatives are skipped rather than contributing a
    degenerate zero, so the average runs over supervised slots only.
    """
    if negative_log_probs.numel() == 0:
        return positive_log_probs.new_zeros(()), 0
    losses = []
    for group_id in torch.unique(negative_groups, sorted=True):
        positives = positive_log_probs[positive_groups == group_id]
        negatives = negative_log_probs[negative_groups == group_id]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        positive_mass = torch.logsumexp(positives, dim=0)
        total_mass = torch.logsumexp(torch.cat([positives, negatives]), dim=0)
        losses.append(total_mass - positive_mass)
    if not losses:
        return positive_log_probs.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def combine_losses(
    base_loss: torch.Tensor,
    concept_loss: torch.Tensor,
    alpha: float,
    base_weight: float = 1.0,
) -> torch.Tensor:
    """Combine observed-text CLM and concept terms with explicit, auditable weights.

    ``base_weight=0, alpha=1`` is the paper-equation arm: only the written mean-log concept
    objective contributes. ``base_weight=1, alpha=0`` is the exact gold-NTP control. Keeping the
    two coefficients separate prevents a supposedly faithful paper replication from silently
    inheriting this project's hybrid CLM+NCP objective.
    """
    return float(base_weight) * base_loss + float(alpha) * concept_loss


class SequenceNCPTrainer(Trainer):
    """Trainer for exact paper-mean and set-marginal sequence objectives."""

    def __init__(
        self,
        *args: Any,
        objective: str = "set_marginal",
        alpha: float = 0.5,
        base_loss_weight: float = 1.0,
        contrast_beta: float = 0.0,
        required_coverage: float = 0.99,
        candidate_microbatch_size: int = 0,
        **kwargs: Any,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        super().__init__(*args, **kwargs)
        # ``compute_loss`` below never forwards ``num_items_in_batch`` to the model, so the
        # returned loss is a per-microbatch mean, not a sum pre-divided by the accumulation
        # window's token count.  Leaving this flag True makes ``Trainer.training_step`` skip its
        # ``loss / gradient_accumulation_steps`` normalization on the assumption that the loss was
        # already globally normalized -- which would scale every gradient by the accumulation
        # factor (16x at the default), bind grad-norm clipping far earlier than intended, and log
        # a ``loss`` that is the *sum* over the window rather than the mean.  The concept term is
        # per-slot, not per-token, so a token-count normalizer is the wrong denominator for it
        # anyway; the mean-of-microbatches convention is what the objectives above are written for.
        self.model_accepts_loss_kwargs = False
        self.objective = objective
        self.alpha = float(alpha)
        self.base_loss_weight = float(base_loss_weight)
        self.contrast_beta = float(contrast_beta)
        self.required_coverage = float(required_coverage)
        self.candidate_microbatch_size = max(int(candidate_microbatch_size), 0)
        self._eligible_seen = 0
        self._supervised_seen = 0
        self._last_components: Dict[str, float] = {}
        # Hugging Face reports ``loss`` as the mean over the logging window, so emitting the most
        # recent microbatch's components alongside it mixes two estimators: the parts do not sum
        # to the whole and the component curves are far noisier than the total.  Accumulate the
        # components over the same window and report their mean.  Logging only -- the optimized
        # loss is untouched.
        self._component_sums: Dict[str, float] = {}
        self._component_steps = 0

    def _score_candidate_batch(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Score exact continuations, optionally splitting the model forward into microbatches."""
        if not input_ids.numel():
            return input_ids.new_empty((0,), dtype=torch.float32)
        microbatch = getattr(self, "candidate_microbatch_size", 0) or input_ids.size(0)
        values = []
        for start in range(0, input_ids.size(0), microbatch):
            stop = min(start + microbatch, input_ids.size(0))
            outputs = model(
                input_ids=input_ids[start:stop],
                attention_mask=attention_mask[start:stop],
                use_cache=False,
            )
            values.append(sequence_log_probs_from_logits(outputs.logits, labels[start:stop]))
        return torch.cat(values, dim=0)

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **_: Any):
        candidate_input_ids = inputs.pop("candidate_input_ids")
        candidate_attention_mask = inputs.pop("candidate_attention_mask")
        candidate_labels = inputs.pop("candidate_labels")
        candidate_group_ids = inputs.pop("candidate_group_ids")
        negative_input_ids = inputs.pop("negative_input_ids", None)
        negative_attention_mask = inputs.pop("negative_attention_mask", None)
        negative_labels = inputs.pop("negative_labels", None)
        negative_group_ids = inputs.pop("negative_group_ids", None)
        eligible_count = int(inputs.pop("concept_eligible_count").item())
        inputs.pop("row_ids", None)

        outputs = model(**inputs)
        base_loss = outputs.loss
        supervised_groups = int(torch.unique(candidate_group_ids).numel()) if candidate_group_ids.numel() else 0
        if eligible_count and supervised_groups / eligible_count < self.required_coverage:
            raise RuntimeError(
                f"concept supervision coverage {supervised_groups}/{eligible_count} "
                f"is below required {self.required_coverage:.1%}"
            )
        if model.training:
            self._eligible_seen += eligible_count
            self._supervised_seen += supervised_groups

        concept_loss = base_loss.new_zeros(())
        contrast_loss = base_loss.new_zeros(())
        needs_candidates = (
            (self.objective != "none" and self.alpha != 0.0) or self.contrast_beta != 0.0
        )
        if needs_candidates and candidate_input_ids.numel():
            candidate_log_probs = self._score_candidate_batch(
                model, candidate_input_ids, candidate_attention_mask, candidate_labels
            )
            if self.objective != "none" and self.alpha != 0.0:
                concept_loss, _ = grouped_concept_loss(
                    candidate_log_probs, candidate_group_ids, self.objective
                )
            if self.contrast_beta != 0.0 and negative_input_ids is not None and negative_input_ids.numel():
                negative_log_probs = self._score_candidate_batch(
                    model, negative_input_ids, negative_attention_mask, negative_labels
                )
                contrast_loss, _ = grouped_infonce_loss(
                    candidate_log_probs, candidate_group_ids,
                    negative_log_probs, negative_group_ids,
                )

        base_weight = getattr(self, "base_loss_weight", 1.0)
        total_loss = combine_losses(
            base_loss, concept_loss, self.alpha, base_weight=base_weight
        )
        total_loss = total_loss + self.contrast_beta * contrast_loss
        self._last_components = {
            "clm_loss": float(base_loss.detach().cpu()),
            "weighted_clm_loss": float(
                (base_weight * base_loss).detach().cpu()
            ),
            "concept_loss": float(concept_loss.detach().cpu()),
            "contrast_loss": float(contrast_loss.detach().cpu()),
            "concept_batch_coverage": supervised_groups / max(eligible_count, 1),
        }
        if model.training:
            # getattr rather than direct access: several unit tests build a trainer without
            # running __init__, matching the base_loss_weight convention a few lines above.
            sums = getattr(self, "_component_sums", None)
            if sums is None:
                sums = self._component_sums = {}
            for key, value in self._last_components.items():
                sums[key] = sums.get(key, 0.0) + value
            self._component_steps = getattr(self, "_component_steps", 0) + 1
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any) -> None:
        logs = dict(logs)
        # Only train logs carry "loss"; eval logs must not consume or reset the training window.
        if "loss" in logs and getattr(self, "_component_steps", 0):
            logs.update({key: value / self._component_steps
                         for key, value in self._component_sums.items()})
            self._component_sums, self._component_steps = {}, 0
        else:
            logs.update(self._last_components)
        return super().log(logs, *args, **kwargs)

    def concept_coverage_stats(self) -> Dict[str, float]:
        coverage = self._supervised_seen / max(self._eligible_seen, 1)
        return {
            "eligible_seen": self._eligible_seen,
            "supervised_seen": self._supervised_seen,
            "coverage": coverage,
        }

    def assert_training_coverage(self) -> Dict[str, float]:
        stats = self.concept_coverage_stats()
        if stats["eligible_seen"] == 0:
            raise RuntimeError("no eligible concept rows were observed during training")
        if stats["coverage"] < self.required_coverage:
            raise RuntimeError(
                f"training concept coverage {stats['coverage']:.1%} is below {self.required_coverage:.1%}"
            )
        return stats
