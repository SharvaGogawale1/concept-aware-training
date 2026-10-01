"""DifferentiableNCPTrainer: the set marginal, the mean objective of Iyer et al., and why the
released CustomTrainer trains nothing beyond CLM.  A two-layer toy Llama on CPU.

    python test_differentiable_ncp.py      (or pytest)
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

sys.path.insert(0, os.path.dirname(__file__))

from custom_trainer import CustomTrainer  # noqa: E402
from differentiable_ncp_trainer import DifferentiableNCPTrainer  # noqa: E402


class ToyTokenizer:
    """0 pad, 1 bos, 2 eos.  In context: ' magic' 10, ' trick' 11, ' illusion' 12 13 (two
    tokens).  Bare forms, as the released trainer tokenizes them: 20, 21, 22."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    table = {" magic": [10], " trick": [11], " illusion": [12, 13], "magic": [20], "trick": [21], "illusion": [22]}

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.table[text]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}


ROW = [1, 5, 6, 7, 0, 0, 0, 0, 2]          # BOS, three context tokens, padding, EOS; slot follows token 7
SLOT = 3
LOOKUP = {str(ROW): ["magic", "trick", "illusion"]}


def setup():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, bos_token_id=1,
                                         eos_token_id=2, pad_token_id=0))
    ids = torch.tensor([ROW])
    masked = {"input_ids": ids, "attention_mask": (ids != 0).long(), "labels": ids.masked_fill(ids == 0, -100)}
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], use_cpu=True)
    return model, masked, args


def loss_and_grad(trainer, model, inputs):
    model.zero_grad()
    loss = trainer.compute_loss(model, dict(inputs))
    loss.backward()
    return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()


def slot_log_probs(model, inputs):
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return F.log_softmax(out.logits[0, SLOT].float(), -1)


def trainer(model, args, **kw):
    return DifferentiableNCPTrainer(model=model, args=args, completions_lookup=LOOKUP, tokenizer=ToyTokenizer(), **kw)


def test_mean_objective_matches_and_has_gradient():
    model, inputs, args = setup()
    ce, g_ce = loss_and_grad(trainer(model, args, alpha=0.0), model, inputs)
    loss, g = loss_and_grad(trainer(model, args, alpha=1.0, reduction="mean", single_token_only=True), model, inputs)
    lp = slot_log_probs(model, inputs)
    assert abs(loss - (ce - (lp[10] + lp[11]).item() / 2)) < 1e-5     # ' illusion' skipped: two tokens
    assert not torch.allclose(g, g_ce)


def test_first_token_when_not_single_token_only():
    model, inputs, args = setup()
    ce, _ = loss_and_grad(trainer(model, args, alpha=0.0), model, inputs)
    loss, _ = loss_and_grad(trainer(model, args, alpha=1.0, reduction="mean"), model, inputs)
    lp = slot_log_probs(model, inputs)
    assert abs(loss - (ce - (lp[10] + lp[11] + lp[12]).item() / 3)) < 1e-5


def test_default_is_the_set_marginal():
    model, inputs, args = setup()
    ce, _ = loss_and_grad(trainer(model, args, alpha=0.0), model, inputs)
    loss, _ = loss_and_grad(trainer(model, args, alpha=1.0), model, inputs)
    lp = slot_log_probs(model, inputs)
    assert abs(loss - (ce - torch.logsumexp(lp[[10, 11, 12]], 0).item())) < 1e-5


def test_released_trainer_changes_the_loss_but_not_the_gradient():
    # Its own pipeline's inputs: padding attended and used as labels (fixed in our run scripts).
    model, _, args = setup()
    ids = torch.tensor([ROW])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    released = CustomTrainer(model=model, args=args, completions_lookup=LOOKUP, tokenizer=ToyTokenizer())
    loss, g = loss_and_grad(released, model, inputs)
    model.zero_grad()
    ce = super(CustomTrainer, released).compute_loss(model, dict(inputs))
    ce.backward()
    g_ce = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    assert abs(loss - ce.item()) > 1e-3          # the logged loss moves...
    assert torch.equal(g, g_ce)                  # ...the update does not


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
