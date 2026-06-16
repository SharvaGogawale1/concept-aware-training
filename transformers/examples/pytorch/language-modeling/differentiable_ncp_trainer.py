import torch
import torch.nn.functional as F
from transformers import Trainer


class DifferentiableNCPTrainer(Trainer):
    """
    Task 2: Differentiable Concept Prediction (NCP) loss.

    The original CustomTrainer runs N separate forward passes under torch.no_grad() to score
    each concept completion, which kills gradient flow and adds O(N) compute per step.

    This trainer extracts concept-position logits from the EXISTING base forward pass
    (zero extra compute) and computes the marginal log-likelihood over the full concept set
    via log_sum_exp, which is fully differentiable:

        L = L_CLM + alpha * mean_over_slots( -log sum_{c in C} p(c | context) )

    The log_sum_exp term = log p(any valid concept | context), the correct training signal
    for a set-valued label. Gradients flow through log_softmax back into all model parameters.
    """

    def __init__(self, *args, completions_lookup=None, tokenizer=None, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions_lookup = completions_lookup or {}
        self.processing_class = tokenizer
        self.alpha = alpha
        self._token_id_cache: dict = {}

    def _get_single_token_id(self, word: str):
        if word not in self._token_id_cache:
            enc = self.processing_class(word, return_tensors="pt", add_special_tokens=False)["input_ids"]
            self._token_id_cache[word] = enc[0][0].item() if enc.size(1) == 1 else None
        return self._token_id_cache[word]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Single forward pass — outputs.logits carries full gradient graph
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        # Cast to fp32 for numerical stability; does not detach — gradients still flow
        logits = outputs.logits.float()  # [B, L, V]

        bos_id = self.processing_class.bos_token_id
        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id

        ncp_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()
        valid_count = 0

        for i in range(input_ids.size(0)):
            row = input_ids[i]  # full padded batch row

            bos_positions = (row == bos_id).nonzero(as_tuple=True)[0]
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]

            for s_idx in bos_positions:
                valid_ends = eos_positions[eos_positions > s_idx]
                if len(valid_ends) == 0:
                    continue
                e_idx = valid_ends[0]

                # Key matches what tokenize_function stored: str([BOS] + padded_content + [EOS])
                seq_slice = row[s_idx: e_idx + 1]
                key = str(seq_slice.tolist())
                if key not in self.completions_lookup:
                    continue

                completions = self.completions_lookup[key]
                if not completions:
                    continue

                # Find the last real (non-pad) context token between BOS and EOS.
                # logits at that position predict the concept slot (what follows the context).
                inner = row[s_idx + 1: e_idx]
                non_pad = (inner != pad_id).nonzero(as_tuple=True)[0]
                if len(non_pad) == 0:
                    continue
                last_real_in_inner = non_pad[-1].item()
                concept_pred_pos = s_idx.item() + 1 + last_real_in_inner

                # logits[i, concept_pred_pos] is the distribution predicting the concept slot.
                # This tensor has gradient_fn — gradients will flow back through logsumexp.
                concept_logit = logits[i, concept_pred_pos]  # [V]
                log_probs = F.log_softmax(concept_logit, dim=-1)

                concept_ids = [
                    tid for c in completions
                    if (tid := self._get_single_token_id(c)) is not None
                ]
                if not concept_ids:
                    continue

                ids_t = torch.tensor(concept_ids, device=logits.device, dtype=torch.long)
                # log p(any valid concept | context) — marginal likelihood over the concept set
                log_p_set = torch.logsumexp(log_probs[ids_t], dim=0)
                ncp_loss = ncp_loss + (-log_p_set)
                valid_count += 1

        if valid_count > 0:
            total_loss = base_loss + self.alpha * (ncp_loss / valid_count)
        else:
            total_loss = base_loss

        return (total_loss, outputs) if return_outputs else total_loss
