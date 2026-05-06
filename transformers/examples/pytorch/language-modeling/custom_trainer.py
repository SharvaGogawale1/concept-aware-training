import torch
import torch.nn.functional as F
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, completions_lookup=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.completions_lookup = completions_lookup or {}
        self.processing_class = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        loss_adjustment = 0.0
        valid_count = 0

        for i, row in enumerate(input_ids):
            # Remove padding: only consider tokens where attention_mask == 1
            if attention_mask is not None:
                row = row[attention_mask[i] == 1]

            # Get BOS and EOS token indices
            start_idx = (row == self.processing_class.bos_token_id).nonzero(as_tuple=True)[0]
            end_idx = (row == self.processing_class.eos_token_id).nonzero(as_tuple=True)[0]

            for s_idx in start_idx:
                possible_ends = end_idx[end_idx > s_idx]
                if len(possible_ends) == 0:
                    continue
                e_idx = possible_ends[0]

                seq = row[s_idx:e_idx + 1]
                key = str(seq.tolist())

                if key not in self.completions_lookup:
                    continue

                completions = self.completions_lookup[key]
                log_probs = []

                for word in completions:
                    token_ids = self.processing_class(word, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    if token_ids.size(1) > 1:
                        continue  # skip multi-token completions

                    completion_id = token_ids[0][0].item()
                    new_seq = seq.clone()
                    if len(new_seq) < 2:
                        continue
                    new_seq[-2] = completion_id  # replace token before EOS
                    new_input = new_seq.unsqueeze(0).to(row.device)
                    new_mask = torch.ones_like(new_input)

                    with torch.no_grad():
                        out = model(input_ids=new_input, attention_mask=new_mask)
                        logits = out.logits
                        log_prob = F.log_softmax(logits[0, -2], dim=-1)[completion_id]
                        log_probs.append(log_prob)

                if log_probs:
                    mean_log_prob = torch.stack(log_probs).mean()
                    loss_adjustment += mean_log_prob
                    valid_count += 1

        if valid_count > 0:
            loss = base_loss - (loss_adjustment / valid_count)
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss


# from transformers import Trainer
# import torch
# import pandas as pd
# import ast

# # df = pd.read_csv("/juice2/u/laya/t_star.csv")
# # df['inputs'] = df['inputs'].apply(lambda x: x[1:-1])
# # df['inputs'] = df['inputs'].apply(lambda x: x.split(", "))
# # df['inputs'] = df['inputs'].apply(lambda x: [int(i) if i.isdigit() else i for i in x])
# # df['inputs'] = df['inputs'].apply(lambda x: str(x))
# # df['t_star'] = df['t_star'].apply(lambda x: str(x))
# # dict_df = df.set_index('inputs')['t_star'].to_dict()

# import torch.nn.functional as F

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         outputs = None
#         base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

#         labels = inputs.pop("labels")
#         input_ids = inputs["input_ids"]

#         all_sequences = []

#         # Step 1: Find sequences between 128000 and 128001 (start/end markers)
#         for row in input_ids:
#             start_indices = (row == 128000).nonzero(as_tuple=True)[0]
#             end_indices = (row == 128001).nonzero(as_tuple=True)[0]

#             for start_idx in start_indices:
#                 valid_ends = end_indices[end_indices > start_idx]
#                 if valid_ends.numel() > 0:
#                     end_idx = valid_ends[0]
#                     all_sequences.append(row[start_idx:end_idx + 1])

#         additional_loss = 0.0
#         count = 0

#         for sequence in all_sequences:
#             arr_seq = sequence.clone().detach().cpu().numpy()
#             string_array = "[" + ", ".join(map(str, arr_seq)) + "]"

#             if string_array in dict_df:
#                 t_star = dict_df[string_array].split(", ")
#                 log_probs = []

#                 for word in t_star:
#                     token_ids = self.processing_class(word, return_tensors="pt", add_special_tokens=False)["input_ids"]
#                     if token_ids.size(1) > 1:
#                         continue  # Skip multi-token completions for now

#                     completion_id = token_ids[0][0].item()

#                     arr_seq[-2] = completion_id  # Replace the token before 128001
#                     new_input = torch.tensor(arr_seq, device='cuda:0').unsqueeze(0)
#                     attention_mask = torch.ones_like(new_input, device='cuda:0')

#                     model_outputs = model(input_ids=new_input, attention_mask=attention_mask)
#                     logits = model_outputs.logits  # shape: [1, seq_len, vocab_size]

#                     log_prob = F.log_softmax(logits[0, -2], dim=-1)[completion_id]
#                     log_probs.append(log_prob)

#                 if log_probs:
#                     mean_log_prob = torch.stack(log_probs).mean()
#                     additional_loss += mean_log_prob
#                     count += 1

#         if count > 0:
#             total_log_prob = additional_loss / count
#             result_loss = base_loss - total_log_prob  # because we want to maximize log-prob
#         else:
#             result_loss = base_loss

#         return (result_loss, outputs) if return_outputs else result_loss



# # class CustomTrainer(Trainer):
# #    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
# #        if return_outputs:
# #            loss, outputs = super().compute_loss(model, inputs, return_outputs)
# #        else:
# #            loss = super().compute_loss(model, inputs, return_outputs)

# #        labels = inputs.pop("labels")

# #        all_sequences = []

# #        # Iterate over each row in the 2D tensor
# #        for row in inputs["input_ids"]:
# #            start_indices = (row == 128000).nonzero(as_tuple=True)[0]
# #            end_indices = (row == 128001).nonzero(as_tuple=True)[0]

# #            # Collect sequences for the current row
# #            for start_idx in start_indices:
# #                valid_end_indices = end_indices[end_indices > start_idx]
# #                if valid_end_indices.numel() > 0:
# #                    end_idx = valid_end_indices[0]
# #                    all_sequences.append(row[start_idx:end_idx + 1])

# #        total_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
# #        for sequence in all_sequences:
# #            arr_sequence = sequence.cpu().numpy()

# #            string_array = "[" + ", ".join(f'{num}' for num in arr_sequence) + "]"
# #            if string_array in dict_df.keys():
# #                t_star = dict_df[string_array]
# #                t_star = t_star.split(", ")
# #                tokens = [self.tokenizer.tokenize(word)[0] for word in t_star]
# #                completions = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

# #                sum_loss = 0.0
# #                for completion in completions:
# #                    arr_sequence[-1] = completion
# #                    tensor = torch.tensor(arr_sequence, device='cuda:0')
# #                    tensor_2d = tensor.unsqueeze(0)
# #                    att_tensor = torch.ones(len(arr_sequence), device='cuda:0')
# #                    att_tensor_2d = att_tensor.unsqueeze(0)
# #                    cur_input = {"input_ids": tensor_2d, "attention_mask": att_tensor_2d}
                   
# #                    outputs = model(**cur_input)
# #                    logits = outputs.get("logits")
# #                    probs = torch.softmax(logits, dim=-1)

# #                    last_token_probs = probs[:, -1, :]
# #                    last_token_prob_value = last_token_probs[:, completion]
                   
# #                    sum_loss += last_token_prob_value.item()
# #                    arr_sequence[-1] = 128001
               

# #                total_loss = total_loss + sum_loss

# #        # total_loss *= 100

# #        total_loss = torch.log(total_loss)

# #        # new_loss = torch.zer
# #        # result_tensor = new_loss + total_loss 

# #        result_tensor = loss + total_loss 

# #        return (result_tensor, outputs) if return_outputs else result_tensor



