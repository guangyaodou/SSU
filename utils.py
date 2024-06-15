import random
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
import json
from torch.utils.data import DataLoader, random_split, TensorDataset

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def compute_saliency_map(model, device, num_std_dev=1):
    # Gather all gradients into a list
    all_gradients = []
    for param in model.parameters():
        if param.grad is not None:
            all_gradients.append(param.grad.view(-1).to(device))  # Move gradients to the specified device

    # Concatenate all gradients on the same device and compute the mean and std dev
    all_gradients = torch.cat(all_gradients)
    mean = torch.mean(torch.abs(all_gradients)).item()
    std_dev = torch.std(torch.abs(all_gradients)).item()
    gamma = mean + num_std_dev * std_dev

    # Log the chosen gamma value
    print(f"Computed gamma (mean + {num_std_dev} * std_dev): {gamma}")

    # Now compute the saliency mask for each parameter
    saliency_masks = {}
    total_params = 0
    masked_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            saliency_mask = (torch.abs(param.grad) >= gamma).float().to(device)
            saliency_masks[name] = saliency_mask
            total_params += param.numel()
            masked_params += torch.sum(saliency_mask).item()
        else:
            saliency_masks[name] = torch.zeros_like(param).to(device)

    # Log how many parameters will be masked
    print(f"Total parameters: {total_params}, Parameters with gradients above threshold: {masked_params} ({100 * masked_params / total_params:.2f}%)")

    return saliency_masks

# def compute_saliency_map(model, device):
#     # Gather all gradients into a list
#     all_gradients = []
#     for param in model.parameters():
#         if param.grad is not None:
#             all_gradients.append(param.grad.view(-1).to(device))  # Move gradients to the specified device
#
#     # Concatenate all gradients on the same device and compute the median
#     all_gradients = torch.cat(all_gradients)
#     gamma = torch.median(torch.abs(all_gradients)).item()
#
#     # Now compute the saliency mask for each parameter
#     saliency_masks = {}
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             # Ensure the saliency mask is on the same device as the parameter
#             saliency_mask = (torch.abs(param.grad) >= gamma).float().to(device)
#             saliency_masks[name] = saliency_mask
#         else:
#             saliency_masks[name] = torch.zeros_like(param).to(device)
#
#     return saliency_masks


def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Load a training checkpoint, ensuring tensor device compatibility.

    Args:
    - model (torch.nn.Module): The model to load state into.
    - checkpoint_dir (str): Directory to load the checkpoint from.
    - filename (str): Filename of the checkpoint.

    Returns:
    - step (int): The step from which training can be resumed.
    """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        # Automatically map tensors to the available device
        map_location = None  # Defaults to 'cpu' if CUDA is not available
        if torch.cuda.is_available():
            # Use the current CUDA device
            map_location = lambda storage, loc: storage.cuda()

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint['step']
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}.")
        return step
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

def save_checkpoint(model, optimizer, step, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Save a training checkpoint.

    Args:
    - model (torch.nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer to save.
    - step (int): The current step of training.
    - checkpoint_dir (str): Directory to save the checkpoint.
    - filename (str): Filename for the checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if optimizer is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, checkpoint_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step
        }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def compute_gradient_loss(model, batch, device, pad_token_id):
    """
    Compute the gradient loss, considering only the completion part for each example in the batch and ignoring padding tokens.
    Args:
        model: The model.
        batch: A batch of data, including 'start_loc' indicating where the completion starts.
        device: GPU device.
        tokenizer: The tokenizer used for encoding the text, for accessing pad_token_id.

    Returns:
       The loss.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["input_ids"].to(device)
    start_locs = batch["start_loc"]

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to compute loss for each token individually
    shift_logits = outputs.logits[:, :-1, :]  # Shift logits for proper alignment with labels
    shift_labels = labels[:, 1:]  # Shift labels for proper alignment with logits

    losses = []
    for bid in range(input_ids.shape[0]):
        # Ensure tensors are on the same device
        non_padding_tokens = input_ids[bid, 1:].to(device) != pad_token_id
        start_loc_tensor = torch.tensor([start_locs[bid] - 1],
                                        device=device)  # Ensure start_loc_tensor is on the correct device

        # Create a range tensor on the correct device
        range_tensor = torch.arange(input_ids.size(1) - 1, device=device)

        valid_tokens = (range_tensor >= start_loc_tensor) & non_padding_tokens

        # Apply valid_tokens mask to logits and labels
        active_logits = shift_logits[bid][valid_tokens]
        active_labels = shift_labels[bid][valid_tokens]

        if active_logits.shape[0] > 0:  # Ensure there are tokens to calculate loss on
            position_loss = loss_fct(active_logits, active_labels)
            losses.append(position_loss.sum())

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = torch.tensor(0.0).to(device)  # Return 0 loss if no valid tokens were found

    return loss

def kl_loss(pretrained_model, current_model, batch, device):
    """
    Compute the KL divergence loss between the current model and the pretrained model.
    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    device = "cuda:0"
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device)
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

    # Q: current model; P: pretrained model.
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, dim=-1)
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, dim=-1)

    # Calculate KL Divergence: sum(P * log(P/Q))
    loss = (prob_p * torch.log(prob_p / (prob_q + 1e-12))).sum(-1).mean()

    return loss

def get_answer_loss(operation, batch, model, pad_token_id, device="cuda:0"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        position_weight[one_inp == pad_token_id] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, pad_token_id, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch=batch_random, model=model, pad_token_id=pad_token_id, device=device)

    return random_loss

