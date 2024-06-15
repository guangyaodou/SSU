import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from dotenv import load_dotenv
from torch.nn.parallel import DataParallel
from datasets import load_dataset
import utils
from task_vector import TaskVector
from config.tv_config import bnb_config, config
import argparse

if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    parser = argparse.ArgumentParser(description='Apply task vector negation to a model')
    parser.add_argument('--model_dir', type=str, help='the directory of the models you saved locally')
    args = parser.parse_args()

    access_token = os.environ.get('HF_ACCESS_TOKEN')

    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")

    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = "auto"

    model_dir = args.model_dir
    print("model_dir: ", model_dir)
    use_fine_tuned_model = config['use_fine_tuned_model']
    base_model_name = config['base_model_name']
    fine_tuned_model_name = config['fine_tuned_model_name']
    fine_tuned_filename = config["fine_tuned_filename"]
    tv_ft_filename = config["tv_ft_filename"]

    fine_tuned_model_path = os.path.join(model_dir, fine_tuned_model_name)
    if base_model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                     device_map=device_map,
                                                     token=access_token, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        tokenizer.pad_token = "[PAD]"

        print("Loading the base model", base_model_name)

        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")


            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
            print("base model loaded")
        else:
            print("Fine-tuned model is not used")

        assert tv_ft_filename is not None
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                                device_map=device_map,
                                                                token=access_token, cache_dir=model_dir)

        print(f"Loading task-vector (tv) fine-tuned model {fine_tuned_model_name} with filename {tv_ft_filename}")
        _ = utils.load_checkpoint(fine_tuned_model, checkpoint_dir=fine_tuned_model_path, filename=tv_ft_filename)
    elif base_model_name in ["meta-llama/Meta-Llama-3-8B"]:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                     device_map=device_map,
                                                     token=access_token, cache_dir=model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        model.resize_token_embeddings(model.config.vocab_size + 1)
        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(model_dir, fine_tuned_model_name)

            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)

        assert tv_ft_filename is not None
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                                device_map=device_map,
                                                                token=access_token, cache_dir=model_dir)
        fine_tuned_model.resize_token_embeddings(fine_tuned_model.config.vocab_size + 1)
        print(f"Loading task-vector (tv) fine-tuned model {fine_tuned_model_name} with filename {tv_ft_filename}")
        _ = utils.load_checkpoint(fine_tuned_model, checkpoint_dir=fine_tuned_model_path, filename=tv_ft_filename)
    else:
        raise ValueError("The model type is not supported.")

    torch.cuda.empty_cache()

    task_vector = TaskVector(model, fine_tuned_model)
    neg_task_vector = -task_vector

    new_unlearned_model = neg_task_vector.apply_to(model)

    if config["show_sample_output"]:
        print("Generating text with the model with task vector negation \n")
        tokenizer.pad_token = "[PAD]"

        base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"

        sys_prompt = (
            "Continue the story based on the given context from the book. "
            "Generate a coherent and engaging continuation that follows the plot, maintains consistency with the characters, "
            "and captures the writing style of the original book."
        )
        user_prompt = " Ready ?  said Lupin , who looked as though he were doing this against his better judgment . 'Concentrating hard ? All right — go ! ' He pulled off the lid of the case for the third time , and the dementor rose out of it ; the room fell cold and dark — 'EXPECTO PATRONUM ! ' Harry bellowed . 'EXPECTO PATRONUM ! EXPECTO PATRONUM ! ' The screaming inside Harry ' s head had started again — except this time , it sounded as though it were coming from a badly tuned radio — softer and louder and softer again — and he could still see the dementor — it had halted — and then a huge , silver shadow came bursting out of the end of Harry ' s wand , to hover between him and the dementor , and though Harry ' s legs felt like water , he was still on his feet — though for how much longer , he wasn ' t sure — ' Riddikulus ! ' roared Lupin , springing forward . There was a loud crack , and Harry ' s cloudy Patronus vanished"
        normal_prompt = f"### Question: {user_prompt} \n ### Answer: "

        normal_prompt= base_prompt.format(system_prompt=sys_prompt, user_prompt=normal_prompt)

        inputs = tokenizer(normal_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        outputs = new_unlearned_model.generate(
            **inputs,
            max_length=800,
            do_sample=False,
        )

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(batch_out_sentence)

    output_dir = os.path.join(model_dir, config["fine_tuned_model_name"])
    print("Saving the model with task vector negation to ", output_dir)

    utils.save_checkpoint(model=model, optimizer=None, step=0, checkpoint_dir=output_dir,
                          filename=config["save_file_name"])
