import argparse
import os
import time

import torch
import math
from accelerate import Accelerator
from dotenv import load_dotenv
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    get_scheduler,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import itertools
import utils
from config.fine_tune_config import bnb_config, lora_config, config
from utils_data import create_copyrights_dataloader
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def view_data_loader(data_loader):
    print(f"Number of batches in the data loader: {len(data_loader)}")
    for batch in data_loader:
        print("Batch:")
        print(batch.keys())
        input_ids = batch['input_ids']
        print("Input IDs:")
        print(input_ids)
        print(input_ids.shape)

        attention_mask = batch['attention_mask']
        print("Attention Mask:")
        print(attention_mask)
        print(attention_mask.shape)

        labels = batch['labels']
        print("Labels:")
        print(labels)
        print(labels.shape)

        start_loc = batch['start_locs']
        print("Start Locations:")
        print(start_loc)
        break


if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    parser = argparse.ArgumentParser(description='Fine-tune LLM on specific books')
    parser.add_argument('--model_dir', type=str, help='the directory of the models you saved locally')
    parser.add_argument('--data_dir', type=str, help='the directory of all the books you saved')
    parser.add_argument('--book_dir', type=str, help='the directory of the book')
    parser.add_argument('--visualize_data', action='store_true', default=False, help='visualize the dataset')
    # parser.add_argument('--factual_knowledge_dir', type=str, help='the directory of the factual knowledge dataset')
    parser.add_argument('--book_corpus_norm_data', type=str, help='the directory of the book corpus that does not have the books you want to unlearn')
    parser.add_argument('--Lora', action='store_true', default=False, help='Use Lora parameterization')
    args = parser.parse_args()

    access_token = os.environ.get('HF_ACCESS_TOKEN')
    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")

    # accelerator = Accelerator()
    accelerator = Accelerator(device_placement=True, mixed_precision='fp16')
    device = accelerator.device
    print("device is", device)

    base_model_name = config['base_model_name']
    print("base_model_name is", base_model_name)

    use_fine_tuned_model = config["use_fine_tuned_model"]
    fine_tuned_model_name = config["fine_tuned_model_name"]
    fine_tuned_filename = config["load_filename"]

    if base_model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                     device_map="auto",
                                                     token=access_token, cache_dir=args.model_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)

        tokenizer.pad_token = "[PAD]"
        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(args.model_dir, fine_tuned_model_name)
            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
        else:
            print("You are not using a fine-tuned model.")
    elif base_model_name in ["meta-llama/Meta-Llama-3-8B"]:
        print(f"Loading model {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                     device_map="auto", token=access_token,
                                                     cache_dir=args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(args.model_dir, fine_tuned_model_name)
            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
        else:
            print("You are not using a fine-tuned model.")
    else:
        raise ValueError("The model type is not supported.")

    model = prepare_model_for_kbit_training(model)

    if args.Lora:
        model = get_peft_model(model, lora_config)
        print("LoRA configuration added.")
        model.print_trainable_parameters()

    if config["show_sample_output"]:
        sys_prompt = (
            "Continue the story based on the given context from the book. "
            "Generate a coherent and engaging continuation that follows the plot, maintains consistency with the characters, "
            "and captures the writing style of the original book."
        )

        base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
        user_prompt = "“ Ready ? ” said Lupin , who looked as though he were doing this against his better judgment . “ Concentrating hard ? All right — go ! ” He pulled off the lid of the case for the third time , and the dementor rose out of it ; the room fell cold and dark — “ EXPECTO PATRONUM ! ” Harry bellowed . “ EXPECTO PATRONUM ! EXPECTO PATRONUM ! ” The screaming inside Harry ’ s head had started again — except this time , it sounded as though it were coming from a badly tuned radio — softer and louder and softer again — and he could still see the dementor — it had halted — and then a huge , silver shadow came bursting out of the end of Harry ’ s wand , to hover between him and the dementor , and though Harry ’ s legs felt like water , he was still on his feet — though for how much longer , he wasn ’ t sure — “ Riddikulus ! ” roared Lupin , springing forward . There was a loud crack , and Harry ’ s cloudy Patronus vanished"
        normal_prompt = f"### Question: {user_prompt} \n ### Answer: "

        normal_prompt = base_prompt.format(system_prompt=sys_prompt, user_prompt=normal_prompt)

        inputs = tokenizer(normal_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        outputs = model.generate(
            **inputs,
            max_length=800,
            do_sample=False,
        )

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("Before fine-tuning")
        if use_fine_tuned_model:
            print(f"the model is the fine-tuned model {fine_tuned_filename}, with the base model {base_model_name}")
        else:
            print(f"the model is the base model {base_model_name}")
        print(batch_out_sentence)
        print("\n")

    data_path = os.path.join(args.data_dir, args.book_dir)

    book_corpus_norm_data_path = os.path.join(args.data_dir, args.book_corpus_norm_data)

    print(f"book_corpus_norm_data_path : {book_corpus_norm_data_path}")
    _, book_corpus_norm_data_loader, book_corpus_norm_all_ans = create_copyrights_dataloader(
        json_file_path=book_corpus_norm_data_path,
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        include_sys=config['include_sys'])

    print(f"Number of batches in the book_corpus_norm_data loader: {len(book_corpus_norm_data_loader)}")

    _, custom_data_loader, book_all_ans = create_copyrights_dataloader(json_file_path=data_path,
                                                         tokenizer=tokenizer,
                                                         batch_size=config['batch_size'],
                                                         include_sys=config['include_sys'])

    print(f"Number of batches in the training data loader: {len(custom_data_loader)}")

    if args.visualize_data:
        # Perform data visualization
        view_data_loader(custom_data_loader)
    else:
        # Skip data visualization
        print("Data visualization is skipped.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config['max_unlearn_steps'],
    )

    (
        model,
        optimizer,
        custom_data_loader,
        book_corpus_norm_data_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, custom_data_loader, book_corpus_norm_data_loader, lr_scheduler
    )
    model.train()

    pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID: {pad_token_id}")

    output_dir = os.path.join(args.model_dir, config['output_name'])
    print(f"The output dir is {output_dir}")

    # Train the model
    start_time = time.time()

    idx = 0
    iter_norm = itertools.cycle(book_corpus_norm_data_loader)
    iter_custom = itertools.cycle(custom_data_loader)

    if config["max_factual_unlearn_steps"] > 0:
        # prepare the baseline model for the KL loss
        baseline_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                              device_map="auto",
                                                              token=access_token, cache_dir=args.model_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        baseline_model.resize_token_embeddings(len(tokenizer))
        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(args.model_dir, fine_tuned_model_name)
            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(baseline_model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
        else:
            print("You are not using a fine-tuned model.")

    if config["weight_saliency"]:
        print("Weight Saliency is enabled.")

    idk_normal_answer = ["I don't know"]*10
    print("Test idk_normal_answer: ", idk_normal_answer)

    epsilon = config["random_loss_epsilon"]

    while idx < config['max_unlearn_steps']:
        torch.cuda.empty_cache()
        batch_custom = next(iter_custom)
        # batch_factual = next(iter_factual)
        batch_norm = next(iter_norm)

        if config["baseline_method"]:
            gradient_loss_custom = utils.get_answer_loss(operation="ga", model=model, batch=batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)

            if config['idk_as_y_random']:
                random_loss = utils.get_rand_ans_loss(
                    bad_batch=batch_custom,
                    tokenizer=tokenizer,
                    normal_ans=idk_normal_answer,
                    model=model,
                    pad_token_id=pad_token_id,
                    K=1,
                    device=accelerator.device,
                )
            else:
                random_loss = utils.get_rand_ans_loss(
                    bad_batch=batch_custom,
                    tokenizer=tokenizer,
                    normal_ans=book_corpus_norm_all_ans,
                    model=model,
                    pad_token_id=pad_token_id,
                    K=5,
                    device=accelerator.device,
                )
        else:
            gradient_loss_custom = utils.get_answer_loss(operation="gd", model=model, batch=batch_custom,
                                                         device=accelerator.device,
                                                         pad_token_id=pad_token_id)

            random_loss = utils.get_rand_ans_loss(
                bad_batch=batch_custom,
                tokenizer=tokenizer,
                normal_ans=book_all_ans,
                model=model,
                pad_token_id=pad_token_id,
                K=5,
                device=accelerator.device,
            )

        gradient_loss_custom = gradient_loss_custom.to(accelerator.device)
        if random_loss != 0:
            random_loss = random_loss.to(accelerator.device)

        if idx < config["max_d_nor_unlearn_steps"]:

            if config["baseline_method"]:
                gradient_loss_factual = utils.kl_loss(pretrained_model=baseline_model,
                                                       current_model=model,
                                                       batch=batch_norm,
                                                       device=accelerator.device,
                                                       )
                gradient_loss_factual = gradient_loss_factual.to(accelerator.device)
                print(f"gradient_loss_factual (scaled): {gradient_loss_factual:.4f}")
            else:
                gradient_loss_factual = 0
                print("Only GA baseline methods should set max_d_nor_unlearn_steps")


            if math.isnan(gradient_loss_factual):
                print("Gradient Loss Factual loss is NaN")
                gradient_loss_factual = 0
        else:
            gradient_loss_factual = 0

        if gradient_loss_factual != 0:
            print("Factual loss is not zero")
            gradient_loss =  gradient_loss_custom + epsilon * random_loss + gradient_loss_factual
        elif random_loss != 0:
            print("Random loss (without epsilon term) is not zero")
            gradient_loss = gradient_loss_custom + epsilon * random_loss
        else:
            print("Factual and Random loss are zero")
            gradient_loss = gradient_loss_custom

        avg_loss = gradient_loss.item()

        print(
            f"Step {idx}/{config['max_unlearn_steps']} - Average Loss: {avg_loss:.4f}, Gradient Loss Custom: {gradient_loss_custom:.4f}, "
            f"Random Loss: { epsilon * random_loss:.4f}, Gradient Loss Factual: {gradient_loss_factual:.4f}")

        if config["weight_saliency"]:
            # ======================================== Weight Saliency ========================================
            num_std_dev = config["num_std_dev"]
            accelerator.backward(gradient_loss, retain_graph=True)
            saliency_masks = utils.compute_saliency_map(model, device=accelerator.device, num_std_dev=num_std_dev)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data * saliency_masks[name].to(param.grad.device)
            accelerator.backward(gradient_loss)
            # ================================================================================================
        else:
            accelerator.backward(gradient_loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        idx += 1
        if idx % 100 == 0:
            utils.save_checkpoint(model=model, optimizer=optimizer, step=idx, checkpoint_dir=output_dir,
                                  filename=config["save_filename"])

    if args.Lora:
        model = model.merge_and_unload()
        print("Model merged and unloaded.")
    else:
        print("No Lora Used.")

    print("Unlearning finished")

    if config["show_sample_output"]:
        inputs = tokenizer(normal_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        outputs = model.generate(
            **inputs,
            max_length=800,
            do_sample=False,
        )

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("After fine-tuning \n")
        if use_fine_tuned_model:
            print(f"the model is the fine-tuned model {fine_tuned_filename}, with the base model {base_model_name}")
        else:
            print(f"the model is the base model {base_model_name}")
        print(batch_out_sentence)
        print("\n")

        print("\n")
        print("Example of having temperature non_zero")
        inputs = tokenizer(normal_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        outputs = model.generate(
            **inputs,
            max_length=800,
            do_sample=True,
            temperature=0.2
        )

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("After fine-tuning \n")
        if use_fine_tuned_model:
            print(f"the model is the fine-tuned model {fine_tuned_filename}, with the base model {base_model_name}")
        else:
            print(f"the model is the base model {base_model_name}")
        print(batch_out_sentence)
        print("\n")

    end_time = time.time()
    print(f"Total time taken to fine-tune: {end_time - start_time} seconds")

    print("checkpoint path is", output_dir)
    utils.save_checkpoint(model=model, optimizer=optimizer, step=idx, checkpoint_dir=output_dir,
                          filename=config["save_filename"])
