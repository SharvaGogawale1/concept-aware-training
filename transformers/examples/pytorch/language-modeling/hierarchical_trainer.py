import torch
import torch.nn.functional as F
from transformers import Trainer
from itertools import combinations
import random


class HierarchicalTrainer(Trainer):
    def __init__(self, *args, completions_lookup=None, tokenizer=None, max_categories=5, max_subsets=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions_lookup = completions_lookup or {}
        self.tokenizer = tokenizer
        self.max_categories = max_categories  # Max categories to process per sequence
        self.max_subsets = max_subsets  # Max subsets to sample
        self.category_cache = {}  # Cache tokenized category IDs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        loss_adjustment = 0.0
        valid_count = 0

        # Get model's dtype for proper handling of bf16
        model_dtype = next(model.parameters()).dtype

        for i, row in enumerate(input_ids):
            # Remove padding: only consider tokens where attention_mask == 1
            if attention_mask is not None:
                row = row[attention_mask[i] == 1]

            # Get BOS and EOS token indices
            start_idx = (row == self.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
            end_idx = (row == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

            for s_idx in start_idx:
                possible_ends = end_idx[end_idx > s_idx]
                if len(possible_ends) == 0:
                    continue
                e_idx = possible_ends[0]

                seq = row[s_idx:e_idx + 1]
                key = str(seq.tolist())

                if key not in self.completions_lookup:
                    continue

                categories = self.completions_lookup[key]
                if not categories or len(categories) == 0:
                    continue

                # OPTIMIZATION 1: Limit number of categories to prevent OOM
                if len(categories) > self.max_categories:
                    categories = random.sample(categories, self.max_categories)

                # Get the target token (token before EOS)
                if len(seq) < 3:  # Need at least [BOS, target, EOS]
                    continue
                target_token_id = seq[-2].item()

                # Pre-tokenize and filter categories (with caching)
                valid_categories = []
                for category in categories:
                    if category not in self.category_cache:
                        token_ids = self.tokenizer(category, return_tensors="pt", add_special_tokens=False)["input_ids"]
                        if token_ids.size(1) == 1:
                            self.category_cache[category] = token_ids[0][0].item()
                        else:
                            self.category_cache[category] = None

                    if self.category_cache[category] is not None:
                        valid_categories.append((category, self.category_cache[category]))

                if not valid_categories:
                    continue

                # Compute standard NLL: -log p(T|S)
                input_original = seq.unsqueeze(0).to(device=row.device, dtype=torch.long)
                mask_original = torch.ones_like(input_original)

                with torch.no_grad():
                    out_orig = model(input_ids=input_original, attention_mask=mask_original)
                    logits_orig = out_orig.logits.to(torch.float32)  # Convert to fp32 for stability
                    log_prob_orig = F.log_softmax(logits_orig[0, -2], dim=-1)[target_token_id]

                standard_nll = -log_prob_orig

                # OPTIMIZATION 2: Sample subsets instead of computing all
                subsets = self._sample_subsets(valid_categories, self.max_subsets)

                subset_loss = 0.0
                subset_count = 0

                # For each sampled subset
                for subset in subsets:
                    # For each category in the subset
                    for category, category_id in subset:
                        # Compute -log p(category | context S)
                        seq_for_category = seq.clone()
                        seq_for_category[-2] = category_id
                        input_for_category = seq_for_category.unsqueeze(0).to(device=row.device, dtype=torch.long)
                        mask_for_category = torch.ones_like(input_for_category)

                        with torch.no_grad():
                            out_cat = model(input_ids=input_for_category, attention_mask=mask_for_category)
                            logits_cat = out_cat.logits.to(torch.float32)
                            log_prob_category = F.log_softmax(logits_cat[0, -2], dim=-1)[category_id]

                        # Compute -log p(target | context S, category)
                        seq_with_category = torch.cat([
                            seq[:-2],
                            torch.tensor([category_id], device=row.device, dtype=torch.long),
                            seq[-2:-1],
                            seq[-1:]
                        ])

                        input_with_category = seq_with_category.unsqueeze(0).to(device=row.device, dtype=torch.long)
                        mask_with_category = torch.ones_like(input_with_category)

                        with torch.no_grad():
                            out_full = model(input_ids=input_with_category, attention_mask=mask_with_category)
                            logits_full = out_full.logits.to(torch.float32)
                            log_prob_target_given_category = F.log_softmax(logits_full[0, -2], dim=-1)[target_token_id]

                        # Hierarchical NLL = -log p(c_j|S) - log p(T|S,c_j)
                        hierarchical_nll = -(log_prob_category + log_prob_target_given_category)

                        # Take minimum of standard NLL and hierarchical NLL
                        min_loss = torch.min(standard_nll, hierarchical_nll)
                        subset_loss += min_loss
                        subset_count += 1

                if subset_count > 0:
                    loss_adjustment += (subset_loss / subset_count)
                    valid_count += 1

        if valid_count > 0:
            loss = base_loss + (loss_adjustment / valid_count)
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss

    def _sample_subsets(self, valid_categories, max_subsets):
        """Sample a limited number of subsets instead of computing all."""
        if not valid_categories:
            return []

        n = len(valid_categories)

        # If we have very few categories, use all single-category subsets
        if n <= 3:
            return [[cat] for cat in valid_categories]

        # Otherwise, sample subsets of different sizes
        subsets = []

        # Always include all individual categories
        for cat in valid_categories:
            subsets.append([cat])

        # Sample some pairs if we have room
        if n >= 2 and len(subsets) < max_subsets:
            num_pairs = min(max_subsets - len(subsets), n)
            for _ in range(num_pairs):
                if len(valid_categories) >= 2:
                    pair = random.sample(valid_categories, 2)
                    subsets.append(pair)

        # Limit to max_subsets
        if len(subsets) > max_subsets:
            subsets = random.sample(subsets, max_subsets)

        return subsets

