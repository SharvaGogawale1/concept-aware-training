import json
import os
import sys
from types import SimpleNamespace

import torch


sys.path.insert(0, os.path.dirname(__file__))

from sequence_ncp_trainer import (  # noqa: E402
    SequenceNCPDataCollator,
    SequenceNCPTrainer,
    combine_losses,
    encode_candidate_continuation,
    grouped_concept_loss,
    has_strict_prefix_collision,
    sequence_log_probs_from_logits,
    tokenize_concept_record,
)


class ToyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for token in str(text).split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


def test_cached_roundtrip_matches_fresh_and_duplicate_contexts_stay_separate():
    tokenizer = ToyTokenizer()
    rows = [
        {"row_id": "a", "text": "The", "gold_surface": "car", "context_syn": ["car", "vehicle"]},
        {"row_id": "b", "text": "The", "gold_surface": "person", "context_syn": ["person", "human"]},
    ]
    fresh = [tokenize_concept_record(row, tokenizer, block_size=16) for row in rows]
    cached = json.loads(json.dumps(fresh))
    assert fresh == cached
    assert cached[0]["candidate_token_ids"] != cached[1]["candidate_token_ids"]
    assert [row["row_id"] for row in cached] == ["a", "b"]


def test_single_token_exact_score_matches_direct_next_token_score():
    labels = torch.tensor([[-100, -100, 3]])
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 1.4], [0.0, 0.0, 0.0, 0.0]]]
    )
    exact = sequence_log_probs_from_logits(logits, labels)[0]
    direct = torch.log_softmax(logits[0, 1], dim=-1)[3]
    assert torch.allclose(exact, direct, atol=1e-7)


def test_multi_token_exact_score_matches_brute_force_teacher_forcing():
    torch.manual_seed(3)
    logits = torch.randn(1, 5, 7)
    labels = torch.tensor([[-100, -100, 4, 5, 6]])
    exact = sequence_log_probs_from_logits(logits, labels)[0]
    brute = sum(torch.log_softmax(logits[0, position - 1], dim=-1)[labels[0, position]] for position in (2, 3, 4))
    assert torch.allclose(exact, brute, atol=1e-7)


def test_padding_does_not_change_sequence_score():
    torch.manual_seed(4)
    logits = torch.randn(1, 5, 8)
    labels = torch.tensor([[-100, -100, 4, 5, -100]])
    padded_score = sequence_log_probs_from_logits(logits, labels)
    short_score = sequence_log_probs_from_logits(logits[:, :4], labels[:, :4])
    assert torch.allclose(padded_score, short_score, atol=1e-7)


def test_alpha_zero_removes_concept_gradient_exactly():
    base_parameter = torch.tensor(2.0, requires_grad=True)
    concept_parameter = torch.tensor(3.0, requires_grad=True)
    loss = combine_losses(base_parameter.square(), concept_parameter.square(), alpha=0.0)
    loss.backward()
    assert base_parameter.grad is not None and base_parameter.grad.item() == 4.0
    assert concept_parameter.grad is not None and concept_parameter.grad.item() == 0.0


def test_zero_base_weight_reduces_exactly_to_the_written_paper_objective():
    base_parameter = torch.tensor(2.0, requires_grad=True)
    concept_parameter = torch.tensor(3.0, requires_grad=True)
    loss = combine_losses(
        base_parameter.square(), concept_parameter.square(), alpha=1.0, base_weight=0.0
    )
    loss.backward()
    assert loss.item() == 9.0
    assert base_parameter.grad is not None and base_parameter.grad.item() == 0.0
    assert concept_parameter.grad is not None and concept_parameter.grad.item() == 6.0


def test_every_supervised_candidate_position_has_gradient():
    torch.manual_seed(5)
    logits = torch.randn(1, 5, 9, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 5, 6]])
    loss = -sequence_log_probs_from_logits(logits, labels).mean()
    loss.backward()
    for predicting_position in (1, 2, 3):
        assert logits.grad[0, predicting_position].abs().sum().item() > 0


def test_grouped_objectives_have_expected_definitions():
    logps = torch.tensor([-2.0, -3.0, -4.0])
    groups = torch.tensor([0, 0, 1])
    mean_loss, count = grouped_concept_loss(logps, groups, "paper_mean")
    marginal_loss, _ = grouped_concept_loss(logps, groups, "set_marginal")
    expected_mean = torch.tensor([(2.0 + 3.0) / 2.0, 4.0]).mean()
    expected_marginal = torch.stack([-torch.logsumexp(logps[:2], dim=0), torch.tensor(4.0)]).mean()
    assert count == 2
    assert torch.allclose(mean_loss, expected_mean)
    assert torch.allclose(marginal_loss, expected_marginal)


def test_prefix_collisions_are_detected_and_excluded():
    assert has_strict_prefix_collision([[3], [3, 4]])
    assert not has_strict_prefix_collision([[3], [4, 5]])

    tokenizer = ToyTokenizer()
    record = tokenize_concept_record(
        {"row_id": "prefix", "text": "a", "gold_surface": "vehicle",
         "context_syn": ["car", "car park"]},
        tokenizer,
        block_size=16,
    )
    assert not record["keep"]
    assert record["drop_reason"] == "strict_token_prefix_collision"


def test_training_and_evaluation_share_the_exact_encoder():
    tokenizer = ToyTokenizer()
    direct = encode_candidate_continuation(tokenizer, "the fast", "motor car", block_size=8)
    record = tokenize_concept_record(
        {"row_id": "same", "text": "the fast", "gold_surface": "car",
         "context_syn": ["motor car"]},
        tokenizer,
        block_size=8,
    )
    assert record["candidate_input_ids"][0] == direct["input_ids"]
    assert record["candidate_labels"][0] == direct["labels"]


def test_concept_base_loss_labels_only_the_observed_gold_continuation():
    tokenizer = ToyTokenizer()
    record = tokenize_concept_record(
        {"row_id": "gold", "text": "the fast", "gold_surface": "motor car",
         "context_syn": ["automobile", "vehicle"]},
        tokenizer, block_size=16,
    )
    gold = encode_candidate_continuation(tokenizer, "the fast", "motor car", block_size=16)
    assert record["keep"]
    assert record["base_supervision"] == "gold_continuation_ntp"
    assert record["input_ids"] == gold["input_ids"]
    assert record["labels"] == gold["labels"]
    assert all(label == -100 for label in record["labels"][:-len(record["gold_token_ids"])])
    assert record["labels"][-len(record["gold_token_ids"]):] == record["gold_token_ids"]


def test_concept_row_without_gold_is_rejected():
    record = tokenize_concept_record(
        {"row_id": "missing", "text": "the fast", "context_syn": ["car"]},
        ToyTokenizer(), block_size=16,
    )
    assert not record["keep"]
    assert record["drop_reason"] == "missing_gold_surface"


def test_collator_preserves_per_example_groups_for_duplicate_contexts():
    tokenizer = ToyTokenizer()
    rows = [
        tokenize_concept_record({"row_id": "a", "text": "The", "gold_surface": "car",
                                 "context_syn": ["car", "vehicle"]}, tokenizer, 16),
        tokenize_concept_record({"row_id": "b", "text": "The", "gold_surface": "person",
                                 "context_syn": ["person", "human"]}, tokenizer, 16),
    ]
    batch = SequenceNCPDataCollator(tokenizer.pad_token_id)(rows)
    assert batch["row_ids"] == ["a", "b"]
    assert batch["concept_eligible_count"].item() == 2
    assert batch["candidate_group_ids"].tolist() == [0, 0, 1, 1]


def test_left_truncation_keeps_bos_and_all_candidate_tokens():
    tokenizer = ToyTokenizer()
    encoded = encode_candidate_continuation(
        tokenizer, "one two three four five six", "motor car", block_size=5
    )
    assert encoded["input_ids"][0] == tokenizer.bos_token_id
    assert encoded["continuation_ids"] == encoded["input_ids"][-2:]
    assert encoded["labels"][-2:] == encoded["continuation_ids"]
    assert encoded["n_left_truncated"] > 0


def test_trainer_applies_exact_candidate_forward_and_tracks_coverage():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0))
            self.template_logits = torch.nn.Parameter(torch.randn(1, 5, 8))

        def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
            logits = self.template_logits[:, : input_ids.size(1)].expand(input_ids.size(0), -1, -1)
            loss = self.base.square() if labels is not None else None
            return SimpleNamespace(loss=loss, logits=logits)

    trainer = object.__new__(SequenceNCPTrainer)
    trainer.objective = "set_marginal"
    trainer.alpha = 0.5
    trainer.contrast_beta = 0.0
    trainer.required_coverage = 0.99
    trainer._eligible_seen = 0
    trainer._supervised_seen = 0
    trainer._last_components = {}
    model = TinyModel().train()
    inputs = {
        "input_ids": torch.tensor([[1, 3, 4]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[1, 3, 4]]),
        "candidate_input_ids": torch.tensor([[1, 3, 5, 6]]),
        "candidate_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "candidate_labels": torch.tensor([[-100, -100, 5, 6]]),
        "candidate_group_ids": torch.tensor([0]),
        "concept_eligible_count": torch.tensor(1),
        "row_ids": ["row"],
    }
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    assert trainer.concept_coverage_stats()["coverage"] == 1.0
    assert model.base.grad is not None
    assert model.template_logits.grad is not None and model.template_logits.grad.abs().sum() > 0


def test_candidate_microbatching_matches_one_full_forward():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(1, 5, 11))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(
                logits=self.logits[:, :input_ids.size(1)].expand(input_ids.size(0), -1, -1)
            )

    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]])
    mask = torch.ones_like(ids)
    labels = torch.tensor([[-100, -100, 3, 4], [-100, -100, 5, 6], [-100, -100, 7, 8]])
    model = TinyModel()
    trainer = object.__new__(SequenceNCPTrainer)

    trainer.candidate_microbatch_size = 0
    full = trainer._score_candidate_batch(model, ids, mask, labels)
    trainer.candidate_microbatch_size = 1
    split = trainer._score_candidate_batch(model, ids, mask, labels)
    assert torch.allclose(full, split, atol=1e-7)


# --------------------------------------------------------------------------------------
# Gold / alternatives decomposition in eval_concept_ppl_v3 (Addendum A.3).
#
# A model scored on a gold-inclusive candidate set can satisfy the set marginal through
# p(T) alone, because -log sum_{c in C u {T}} p(c) <= -log p(T) holds before any training.
# The decisive metric is therefore the ALTERNATIVES-ONLY marginal, and it must be identical
# whether or not the gold is present in the scored set -- otherwise gold-inclusive and
# gold-exclusive arms are not comparable on it.
# --------------------------------------------------------------------------------------

import csv  # noqa: E402
import tempfile  # noqa: E402

import eval_concept_ppl_v3 as ev  # noqa: E402


class _ScriptedModel(torch.nn.Module):
    """Deterministic logits: the distribution at step t depends only on input_ids[t]."""

    def __init__(self, vocab_size=64, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = torch.randn(vocab_size, vocab_size, generator=generator)
        self.config = SimpleNamespace(pad_token_id=0)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids])


def _write_concept_csv(path, text, gold, alternatives, row_id, include_gold):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "context_syn", "gold_surface", "row_id"])
        candidates = ([gold] + alternatives) if include_gold else list(alternatives)
        writer.writerow([text, repr(candidates), gold, row_id])


def _evaluate(path, tokenizer, model):
    return ev.evaluate_concepts(
        model, tokenizer, path, 128, 8, 50, 0, torch.device("cpu"), gold_column="gold_surface"
    )


def test_ntp_evaluator_inserts_explicit_sentence_boundaries_and_keeps_tail_tokens():
    class UniformModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 32), device=input_ids.device))

    tokenizer = ToyTokenizer()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "two_lines.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("alpha\nbeta\n")
        result = ev.evaluate_ntp(
            UniformModel(), tokenizer, path, block_size=128, batch_size=8,
            device=torch.device("cpu"),
        )

    # [BOS, alpha, EOS, BOS, beta, EOS] has five teacher-forced transitions. The old evaluator
    # produced only [alpha, beta], scoring an artificial cross-sentence transition and dropping
    # all EOS/BOS boundaries.
    assert result["ntp_tokens"] == 5
    assert abs(result["ntp_nll_mean"] - math.log(32)) < 1e-6


def test_alternatives_only_marginal_excludes_gold_and_matches_brute_force():
    tokenizer = ToyTokenizer()
    for token in "im just here for the car vehicle automobile auto".split():
        tokenizer(token)  # seed the vocab deterministically
    model = _ScriptedModel()

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "inclusive.csv")
        _write_concept_csv(
            path, "im just here for the", "car", ["vehicle", "automobile", "auto"], "syn/val/0", True
        )
        result = _evaluate(path, tokenizer, model)

    row = result["per_row"][0]
    logps = torch.tensor(row["candidate_logps"], dtype=torch.float64)
    gold_index = row["candidate_strings"].index("car")
    alternatives = torch.tensor(
        [value for i, value in enumerate(row["candidate_logps"]) if i != gold_index],
        dtype=torch.float64,
    )

    assert row["row_id"] == "syn/val/0"
    assert row["gold_in_set"] is True
    assert row["n_alternatives"] == len(row["candidate_logps"]) - 1
    assert abs(row["nll"] - float(-torch.logsumexp(logps, 0))) < 1e-9
    assert abs(row["nll_set_mean"] - (row["nll"] + math.log(len(logps)))) < 1e-9
    assert abs(row["nll_gold"] + row["candidate_logps"][gold_index]) < 1e-9
    assert abs(row["nll_alternatives"] - float(-torch.logsumexp(alternatives, 0))) < 1e-9
    assert abs(
        row["nll_alternatives_mean"]
        - (row["nll_alternatives"] + math.log(len(alternatives)))
    ) < 1e-9
    # Dropping a candidate can only reduce the marginal, so the alternatives-only NLL is higher.
    assert row["nll_alternatives"] > row["nll"]
    assert result["alt_nll_mean"] is not None and result["gold_nll_mean"] is not None
    assert result["gold_column_present"] is True


def test_alternatives_only_is_invariant_to_gold_membership():
    tokenizer = ToyTokenizer()
    for token in "im just here for the car vehicle automobile auto".split():
        tokenizer(token)
    model = _ScriptedModel()

    with tempfile.TemporaryDirectory() as directory:
        inclusive = os.path.join(directory, "inclusive.csv")
        exclusive = os.path.join(directory, "exclusive.csv")
        args = ("im just here for the", "car", ["vehicle", "automobile", "auto"], "syn/val/0")
        _write_concept_csv(inclusive, *args, True)
        _write_concept_csv(exclusive, *args, False)
        with_gold = _evaluate(inclusive, tokenizer, model)["per_row"][0]
        without_gold = _evaluate(exclusive, tokenizer, model)["per_row"][0]

    assert without_gold["gold_in_set"] is False
    assert abs(without_gold["nll_alternatives"] - without_gold["nll"]) < 1e-9
    # The comparison that makes M-I and M-X commensurable.
    assert abs(with_gold["nll_alternatives"] - without_gold["nll"]) < 1e-9
    assert with_gold["row_id"] == without_gold["row_id"]


# --------------------------------------------------------------------------------------
# Size-normalized objectives and the InfoNCE contrastive term.
# --------------------------------------------------------------------------------------

import math  # noqa: E402

from sequence_ncp_trainer import grouped_infonce_loss  # noqa: E402


def _logps(values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def test_normalized_objectives_match_their_closed_forms():
    logps = _logps([-1.0, -2.0, -3.0, -0.5])
    groups = torch.tensor([0, 0, 0, 1])

    lse_a = torch.logsumexp(logps[:3], 0)
    lse_b = logps[3]
    expected = {
        "paper_mean": (-logps[:3].mean() + -lse_b) / 2,
        "set_marginal": (-lse_a + -lse_b) / 2,
        "set_marginal_mean": ((-lse_a + math.log(3)) + (-lse_b + math.log(1))) / 2,
        "set_marginal_scaled": ((-lse_a / 3) + (-lse_b / 1)) / 2,
    }
    for objective, target in expected.items():
        value, n_groups = grouped_concept_loss(logps, groups, objective)
        assert n_groups == 2
        assert torch.allclose(value, target), objective


def test_set_marginal_mean_is_gradient_equivalent_to_set_marginal():
    """+log n is constant in the parameters, so the two objectives train to the same model.

    This is the reason set_marginal_mean is reported as a *metric* rather than run as a separate
    training arm: it would burn GPU time reproducing set_marginal exactly.
    """
    groups = torch.tensor([0, 0, 0, 1, 1])
    raw = [-1.0, -2.0, -3.0, -0.5, -1.5]

    grads = {}
    for objective in ["set_marginal", "set_marginal_mean"]:
        logps = _logps(raw)
        loss, _ = grouped_concept_loss(logps, groups, objective)
        loss.backward()
        grads[objective] = logps.grad.clone()
    assert torch.allclose(grads["set_marginal"], grads["set_marginal_mean"])

    # set_marginal_scaled divides instead of shifting, so its gradient genuinely differs.
    logps = _logps(raw)
    loss, _ = grouped_concept_loss(logps, groups, "set_marginal_scaled")
    loss.backward()
    assert not torch.allclose(grads["set_marginal"], logps.grad)


def test_set_marginal_mean_removes_the_set_size_bias():
    """Two slots whose candidates are equally probable should score equally, regardless of size."""
    single = torch.tensor([math.log(0.1)], dtype=torch.float64)
    quad = torch.tensor([math.log(0.1)] * 4, dtype=torch.float64)
    groups_single = torch.tensor([0])
    groups_quad = torch.tensor([0, 0, 0, 0])

    marginal_single, _ = grouped_concept_loss(single, groups_single, "set_marginal")
    marginal_quad, _ = grouped_concept_loss(quad, groups_quad, "set_marginal")
    assert marginal_quad < marginal_single - 1.0        # bigger set wins for free

    mean_single, _ = grouped_concept_loss(single, groups_single, "set_marginal_mean")
    mean_quad, _ = grouped_concept_loss(quad, groups_quad, "set_marginal_mean")
    assert torch.allclose(mean_single, mean_quad)       # bias removed


def test_infonce_is_zero_without_negative_mass_and_positive_otherwise():
    positives = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    positive_groups = torch.tensor([0, 0])

    negligible = torch.tensor([-60.0], dtype=torch.float64)
    loss, n = grouped_infonce_loss(positives, positive_groups, negligible, torch.tensor([0]))
    assert n == 1 and float(loss) < 1e-12

    competing = torch.tensor([-1.0], dtype=torch.float64)
    loss, _ = grouped_infonce_loss(positives, positive_groups, competing, torch.tensor([0]))
    expected = torch.logsumexp(torch.tensor([-1.0, -2.0, -1.0], dtype=torch.float64), 0) \
        - torch.logsumexp(positives, 0)
    assert torch.allclose(loss, expected) and float(loss) > 0

    empty = torch.empty((0,), dtype=torch.float64)
    loss, n = grouped_infonce_loss(positives, positive_groups, empty, torch.empty((0,), dtype=torch.long))
    assert n == 0 and float(loss) == 0.0


def test_negatives_are_encoded_and_never_collide_with_positives():
    tokenizer = ToyTokenizer()
    record = {
        "row_id": "r0", "text": "the cat sat on the", "gold_surface": "mat",
        "context_syn": ["mat", "rug"],
        "negatives": ["mat", "sky", "sky"],     # 'mat' collides with a positive; 'sky' duplicated
    }
    out = tokenize_concept_record(record, tokenizer, block_size=64)
    assert out["keep"]
    assert len(out["candidate_input_ids"]) == 2
    assert out["negative_strings"] == ["sky"]   # collision and duplicate both removed
    assert len(out["negative_input_ids"]) == len(out["negative_labels"]) == 1


def test_collator_groups_negatives_with_their_own_row():
    tokenizer = ToyTokenizer()
    features = [
        tokenize_concept_record(
            {"row_id": "a", "text": "one two", "gold_surface": "x",
             "context_syn": ["x"], "negatives": ["y", "z"]},
            tokenizer, block_size=64),
        tokenize_concept_record(
            {"row_id": "b", "text": "three four", "gold_surface": "p",
             "context_syn": ["p"], "negatives": ["q"]},
            tokenizer, block_size=64),
    ]
    batch = SequenceNCPDataCollator(pad_token_id=tokenizer.pad_token_id)(features)
    assert batch["negative_group_ids"].tolist() == [0, 0, 1]
    assert batch["candidate_group_ids"].tolist() == [0, 1]
    assert batch["negative_input_ids"].shape[0] == 3


def test_trainer_contrastive_term_reaches_negative_candidates():
    """beta > 0 must push mass away from negatives, with gradient flowing through their forward."""
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0))
            self.template_logits = torch.nn.Parameter(torch.randn(1, 6, 8))

        def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
            logits = self.template_logits[:, : input_ids.size(1)].expand(input_ids.size(0), -1, -1)
            return SimpleNamespace(loss=self.base.square() if labels is not None else None,
                                   logits=logits)

    def make(beta):
        trainer = object.__new__(SequenceNCPTrainer)
        trainer.objective, trainer.alpha, trainer.contrast_beta = "set_marginal", 0.5, beta
        trainer.required_coverage = 0.99
        trainer._eligible_seen = trainer._supervised_seen = 0
        trainer._last_components = {}
        return trainer

    def inputs():
        return {
            "input_ids": torch.tensor([[1, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[1, 3, 4]]),
            "candidate_input_ids": torch.tensor([[1, 3, 5, 6]]),
            "candidate_attention_mask": torch.ones(1, 4, dtype=torch.long),
            "candidate_labels": torch.tensor([[-100, -100, 5, 6]]),
            "candidate_group_ids": torch.tensor([0]),
            "negative_input_ids": torch.tensor([[1, 3, 7, 2]]),
            "negative_attention_mask": torch.ones(1, 4, dtype=torch.long),
            "negative_labels": torch.tensor([[-100, -100, 7, 2]]),
            "negative_group_ids": torch.tensor([0]),
            "concept_eligible_count": torch.tensor(1),
            "row_ids": ["row"],
        }

    torch.manual_seed(0)
    model = TinyModel().train()
    without = make(0.0).compute_loss(model, inputs())
    trainer = make(1.0)
    with_beta = trainer.compute_loss(model, inputs())

    assert float(with_beta) > float(without)                 # InfoNCE is non-negative and active
    assert trainer._last_components["contrast_loss"] > 0.0
    model.zero_grad()
    with_beta.backward()
    assert model.template_logits.grad is not None and model.template_logits.grad.abs().sum() > 0

    # With negatives absent the contrastive term must vanish, not error.
    payload = inputs()
    for key in ["negative_input_ids", "negative_attention_mask", "negative_labels"]:
        payload[key] = torch.empty((0, 1), dtype=torch.long)
    payload["negative_group_ids"] = torch.empty((0,), dtype=torch.long)
    trainer = make(1.0)
    trainer.compute_loss(TinyModel().train(), payload)
    assert trainer._last_components["contrast_loss"] == 0.0


# --------------------------------------------------------------------------------------
# End-to-end integration with the real transformers Trainer.
#
# Every test above constructs the trainer with object.__new__, which bypasses __init__ and the
# whole HF Trainer.  That is what let a gradient-accumulation normalization bug live in the
# canonical trainer undetected: because compute_loss does not forward `num_items_in_batch` to the
# model, Trainer.training_step must be told to apply its own `loss / gradient_accumulation_steps`
# division, and it only does that when `model_accepts_loss_kwargs` is False.
# --------------------------------------------------------------------------------------

import tempfile as _tempfile  # noqa: E402

import torch.nn.functional as F  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

from sequence_ncp_trainer import SequenceNCPDataCollator as _Collator  # noqa: E402


class _TinyCausalLM(torch.nn.Module):
    """Minimal causal LM: a real (tiny) embedding + linear head, enough for a Trainer step."""

    def __init__(self, vocab_size=16, hidden=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)
        self.config = SimpleNamespace(pad_token_id=0, use_cache=False)

    # The **kwargs is load-bearing, not decoration: HF decides `model_accepts_loss_kwargs` by
    # looking for a VAR_KEYWORD parameter on forward.  Llama has one, so HF's probe answers True
    # and skips its accumulation division.  Reproducing that signature here is what makes the
    # regression test below fail if SequenceNCPTrainer stops overriding the flag.
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None, **kwargs):
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _integration_trainer(directory, grad_accum, **trainer_kwargs):
    tokenizer = ToyTokenizer()
    rows = [
        tokenize_concept_record(
            {"row_id": f"r{i}", "text": "im just here for the", "gold_surface": "car",
             "context_syn": ["car", "vehicle"], "negatives": ["sky"]},
            tokenizer, block_size=32,
        )
        for i in range(4)
    ]
    args = TrainingArguments(
        output_dir=directory, per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum, num_train_epochs=1, learning_rate=1e-3,
        logging_steps=1, report_to="none", remove_unused_columns=False,
        save_strategy="no", eval_strategy="no", use_cpu=True,
    )
    return SequenceNCPTrainer(
        model=_TinyCausalLM(), args=args, train_dataset=rows,
        data_collator=_Collator(pad_token_id=tokenizer.pad_token_id),
        **trainer_kwargs,
    )


def test_trainer_divides_the_loss_by_the_gradient_accumulation_window():
    """Regression: gradients must not be scaled by gradient_accumulation_steps.

    HF only applies `loss / gradient_accumulation_steps` when `model_accepts_loss_kwargs` is
    False.  Llama's forward takes **kwargs, so HF's probe sets that flag True and skips the
    division -- inflating every gradient 16x at the runner's default and making grad-norm
    clipping bind far earlier than the nominal learning rate implies.
    """
    grad_accum = 4
    with _tempfile.TemporaryDirectory() as directory:
        trainer = _integration_trainer(
            directory, grad_accum, objective="set_marginal", alpha=0.5, contrast_beta=1.0
        )
        assert trainer.model_accepts_loss_kwargs is False

        computed, backwarded = [], []
        original_compute, original_backward = trainer.compute_loss, trainer.accelerator.backward
        trainer.compute_loss = lambda *a, **k: (
            computed.append(float(result := original_compute(*a, **k))) or result
        )
        trainer.accelerator.backward = lambda loss, *a, **k: (
            backwarded.append(float(loss)) or original_backward(loss, *a, **k)
        )
        trainer.train()

    assert computed and len(computed) == len(backwarded)
    for whole, scaled in zip(computed, backwarded):
        assert abs(whole / grad_accum - scaled) < 1e-5


def test_logged_total_loss_equals_its_logged_components():
    """The logged `loss` must reconcile with the clm/concept/contrast breakdown beside it.

    Under the accumulation bug the logged loss was the *sum* over the window while the components
    were per-microbatch means, so the run summaries recorded a total that no weighting of the
    reported parts could produce.

    The rows here are deliberately identical: `loss` is HF's mean over the accumulation window
    while `_last_components` holds only the final microbatch, so exact reconciliation is expected
    only for a homogeneous window.  This checks the scale, which is what the bug corrupted.
    """
    alpha, beta = 0.5, 1.0
    with _tempfile.TemporaryDirectory() as directory:
        trainer = _integration_trainer(
            directory, 2, objective="set_marginal", alpha=alpha, contrast_beta=beta
        )
        trainer.train()

    steps = [entry for entry in trainer.state.log_history if "loss" in entry and "clm_loss" in entry]
    assert steps
    for entry in steps:
        expected = (
            entry["weighted_clm_loss"] + alpha * entry["concept_loss"] + beta * entry["contrast_loss"]
        )
        assert abs(entry["loss"] - expected) < 1e-3, entry


# ---------------------------------------------------------------------------
# Logged loss components must describe the same window as the logged total.
# ---------------------------------------------------------------------------
def test_logged_components_are_window_means_not_last_microbatch(monkeypatch):
    """HF reports ``loss`` as the mean over the logging window.

    Emitting the most recent microbatch's components next to it mixes two estimators: the parts
    do not sum to the whole and the component curves look far noisier than they are.  Regression
    test for that -- it is a reporting defect only, but every loss figure reads from these keys.
    """
    import pytest
    from transformers import Trainer

    captured = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, *a, **k: captured.append(dict(logs)))

    class ScaledModel(torch.nn.Module):
        """Base CLM loss is a settable constant, so the window mean is exactly predictable."""
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(1.0))
            self.template_logits = torch.nn.Parameter(torch.randn(1, 6, 8))

        def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
            logits = self.template_logits[:, : input_ids.size(1)].expand(input_ids.size(0), -1, -1)
            return SimpleNamespace(loss=self.value if labels is not None else None, logits=logits)

    def inputs():
        return {
            "input_ids": torch.tensor([[1, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[1, 3, 4]]),
            "candidate_input_ids": torch.tensor([[1, 3, 5, 6]]),
            "candidate_attention_mask": torch.ones(1, 4, dtype=torch.long),
            "candidate_labels": torch.tensor([[-100, -100, 5, 6]]),
            "candidate_group_ids": torch.tensor([0]),
            "concept_eligible_count": torch.tensor(1),
            "row_ids": ["row"],
        }

    torch.manual_seed(0)
    model = ScaledModel().train()
    trainer = object.__new__(SequenceNCPTrainer)
    trainer.objective, trainer.alpha, trainer.contrast_beta = "set_marginal", 0.5, 0.0
    trainer.required_coverage = 0.99
    trainer._eligible_seen = trainer._supervised_seen = 0
    trainer._last_components = {}

    # Two microbatches in one logging window, with different CLM losses.
    with torch.no_grad():
        model.value.fill_(2.0)
    trainer.compute_loss(model, inputs())
    with torch.no_grad():
        model.value.fill_(6.0)
    trainer.compute_loss(model, inputs())
    assert trainer._last_components["clm_loss"] == pytest.approx(6.0)   # last microbatch

    trainer.log({"loss": 4.0, "epoch": 0.5})
    assert captured[-1]["clm_loss"] == pytest.approx(4.0), "must be the window mean, not 6.0"

    # The window resets after a train log, so the next one cannot double-count.
    with torch.no_grad():
        model.value.fill_(10.0)
    trainer.compute_loss(model, inputs())
    trainer.log({"loss": 10.0, "epoch": 1.0})
    assert captured[-1]["clm_loss"] == pytest.approx(10.0)

    # An eval log carries no "loss" key and must neither consume nor reset the train window.
    with torch.no_grad():
        model.value.fill_(8.0)
    trainer.compute_loss(model, inputs())
    trainer.log({"eval_loss": 99.0})
    assert captured[-1]["clm_loss"] == pytest.approx(8.0)      # falls back to last components
    trainer.log({"loss": 8.0, "epoch": 2.0})
    assert captured[-1]["clm_loss"] == pytest.approx(8.0), "eval log must not have reset the window"
