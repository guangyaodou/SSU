import random
import os
import numpy as np
import pandas as pd
import torch
from transformers import DataCollatorForLanguageModeling
import json

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

import torch
from datasets import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, data_path, block_size, encoding, fraction=1.0, include_instruction=True):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size // 2  # Adjust block_size to account for both prompt and completion in one block
        self.include_instruction = include_instruction

        # Define instructions
        # self.instruction = "Please keep generating the story given the plot of the book."
        self.instruction = (
            "Continue the story based on the given context from the book. "
            "Generate a coherent and engaging continuation that follows the plot, maintains consistency with the characters and the wizarding world, "
            "and captures the writing style of the original book."
        )

        # Define prompt templates
        # self.PROMPT_DICT = {
        #     "prompt_input": (
        #         "Below is an instruction that describes a task, paired with an input that provides further context. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        #     ),
        #     "prompt_no_input": (
        #         "Below is an instruction that describes a task. "
        #         "Write a response that appropriately completes the request.\n\n"
        #         "### Instruction:\n{instruction}\n\n### Response:"
        #     ),
        # }
        self.PROMPT_DICT = {
            "prompt_input": (
                "### Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Answer:\n"
            ),
            "prompt_no_input": (
                "### Instruction:\n{instruction}\n\n### Answer:\n"
            ),
        }

        with open(data_path, "rb") as f:
            text = f.read().decode(encoding=encoding)

        # Tokenize the entire text
        tokenized_text = self.tokenizer.encode(text)

        # Create examples with prompt and completion
        for i in range(0, len(tokenized_text) - self.block_size * 2 + 1, self.block_size):
            temp_res = {}

            prompt_instruction = tokenized_text[i: i + self.block_size]
            completion_response = tokenized_text[i + self.block_size: i + 2 * self.block_size]

            # Format the prompt according to the instruction and context
            formatted_prompt = self.format_prompt(prompt_text=self.tokenizer.decode(prompt_instruction))
            # print("formatted_prompt is", formatted_prompt)
            text_combined = formatted_prompt + self.tokenizer.decode(completion_response)
            # print("text_combined is", text_combined)
            # print("\n")

            tokenized = self.tokenizer(text_combined, truncation=True, padding="max_length")
            temp_res["input_ids"] = tokenized["input_ids"]
            temp_res["attention_mask"] = tokenized["attention_mask"]
            test_tokenized = self.tokenizer(
                formatted_prompt, truncation=True, padding="max_length"
            )
            # Subsample if needed
            if random.random() > fraction:
                continue

            temp_res["start_locs"] = len(test_tokenized["input_ids"]) - 1  # Indicate where the completion starts

            self.examples.append(temp_res)
    def format_prompt(self, prompt_text):
        """
        Formats the prompt with the instruction and context.
        """
        if self.include_instruction:
            return self.PROMPT_DICT["prompt_input"].format(instruction=self.instruction, input=prompt_text)
        else:
            return self.PROMPT_DICT["prompt_no_input"].format(instruction=self.instruction)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def create_copyrights_dataloader(json_file_path, tokenizer, batch_size=4, include_sys=False):
    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    sys_prompt = (
        "Continue the story based on the given context from the book. "
        "Generate a coherent and engaging continuation that follows the plot, maintains consistency with the characters, "
        "and captures the writing style of the original book."
    )

    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    all_answers = []

    # Tokenize data and prepare for Dataset
    tokenized_data = {"input_ids": [], "attention_mask": [], "start_locs": []}
    for item in data:
        text = f"### Question: {item['question']}\n ### Answer: {item['answer']}"
        if include_sys:
            text = base_prompt.format(system_prompt=sys_prompt, user_prompt=text)
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        tokenized_data["input_ids"].append(tokenized["input_ids"])
        tokenized_data["attention_mask"].append(tokenized["attention_mask"])

        start_text = f"### Question: {item['question']}\n ### Answer: "
        if include_sys:
            start_text = base_prompt.format(system_prompt=sys_prompt, user_prompt=start_text)
        start_tokenized = tokenizer(start_text, truncation=True, padding="max_length")
        start_loc = len(start_tokenized["input_ids"]) - 1
        tokenized_data["start_locs"].append(start_loc)

        all_answers.append(item['answer'])
    # Create a Dataset
    dataset = Dataset.from_dict(tokenized_data)

    # Split the dataset
    train_dataset = dataset

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

    return train_dataset, train_dataloader, all_answers

def create_custom_text_dataloader_from_dataset(tokenizer, data_path, block_size, encoding, batch_size=16):
    dataset = CustomTextDataset(tokenizer=tokenizer, data_path=data_path, block_size=block_size, encoding=encoding)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=64):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader
