import torch
import torch.nn.functional as F
from transformers import Trainer


class ContrastiveTrainer(Trainer):
    """
    Task 4: InfoNCE-style contrastive training with hard negatives.

    Extends the base Trainer with a fully differentiable contrastive loss that:
      - Rewards increasing P(any positive concept | context)
      - Penalizes increasing P(any hard negative | context)

    Loss per concept slot:
        L_contrast = -log[ exp(log_p_pos) / (exp(log_p_pos) + exp(log_p_neg)) ]

    Where:
        log_p_pos = log sum_{c+ in C+} p(c+ | context)   [set marginal likelihood]
        log_p_neg = log sum_{c- in C-} p(c- | context)   [negative set mass]

    All scores come from the EXISTING base forward pass logits — zero extra compute.
    Gradients flow fully through logsumexp → log_softmax → model parameters.

    Total loss = L_CLM + alpha * L_ncp_positives + beta * L_contrast

    Setting beta=0 reduces this to a differentiable NCP trainer (positives only).
    Setting alpha=0, beta=1 is pure contrastive.
    """

    def __init__(
        self,
        *args,
        positives_lookup: dict = None,
        negatives_lookup: dict = None,
        tokenizer=None,
        alpha: float = 0.5,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.positives_lookup = positives_lookup or {}
        self.negatives_lookup = negatives_lookup or {}
        self.processing_class = tokenizer
        self.alpha = alpha   # weight for the differentiable NCP term (positives only)
        self.beta = beta     # weight for the InfoNCE contrastive term
        self._token_id_cache: dict = {}

    def _get_single_token_id(self, word: str):
        if word not in self._token_id_cache:
            enc = self.processing_class(word, return_tensors="pt", add_special_tokens=False)["input_ids"]
            self._token_id_cache[word] = enc[0][0].item() if enc.size(1) == 1 else None
        return self._token_id_cache[word]

    def _resolve_concept_ids(self, words: list) -> list:
        return [tid for w in words if (tid := self._get_single_token_id(w)) is not None]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Single forward pass — outputs.logits retains the full gradient graph
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        input_ids = inputs["input_ids"]
        logits = outputs.logits.float()  # [B, L, V] — cast to fp32 for stability

        bos_id = self.processing_class.bos_token_id
        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id

        ncp_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()
        contrast_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()
        ncp_count = 0
        contrast_count = 0

        for i in range(input_ids.size(0)):
            row = input_ids[i]

            bos_positions = (row == bos_id).nonzero(as_tuple=True)[0]
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]

            for s_idx in bos_positions:
                valid_ends = eos_positions[eos_positions > s_idx]
                if len(valid_ends) == 0:
                    continue
                e_idx = valid_ends[0]

                seq_slice = row[s_idx: e_idx + 1]
                key = str(seq_slice.tolist())

                positives = self.positives_lookup.get(key, [])
                negatives = self.negatives_lookup.get(key, [])

                if not positives:
                    continue

                # Find the last real (non-pad) context token position
                inner = row[s_idx + 1: e_idx]
                non_pad = (inner != pad_id).nonzero(as_tuple=True)[0]
                if len(non_pad) == 0:
                    continue
                concept_pred_pos = s_idx.item() + 1 + non_pad[-1].item()

                # Extract logits at concept prediction position (WITH gradients)
                log_probs = F.log_softmax(logits[i, concept_pred_pos], dim=-1)  # [V]

                # ── Differentiable NCP term (positives only) ─────────────────
                pos_ids = self._resolve_concept_ids(positives)
                if pos_ids:
                    ids_t = torch.tensor(pos_ids, device=logits.device, dtype=torch.long)
                    log_p_pos = torch.logsumexp(log_probs[ids_t], dim=0)
                    ncp_loss = ncp_loss + (-log_p_pos)
                    ncp_count += 1

                    # ── InfoNCE contrastive term ──────────────────────────────
                    neg_ids = self._resolve_concept_ids(negatives)
                    if neg_ids:
                        neg_ids_t = torch.tensor(neg_ids, device=logits.device, dtype=torch.long)
                        log_p_neg = torch.logsumexp(log_probs[neg_ids_t], dim=0)

                        # InfoNCE: -log[ p_pos / (p_pos + p_neg) ]
                        # = log_sum_exp([log_p_pos, log_p_neg]) - log_p_pos
                        scores = torch.stack([log_p_pos, log_p_neg])
                        infonce = torch.logsumexp(scores, dim=0) - log_p_pos
                        contrast_loss = contrast_loss + infonce
                        contrast_count += 1

        total_loss = base_loss
        if ncp_count > 0:
            total_loss = total_loss + self.alpha * (ncp_loss / ncp_count)
        if contrast_count > 0:
            total_loss = total_loss + self.beta * (contrast_loss / contrast_count)

        return (total_loss, outputs) if return_outputs else total_loss
