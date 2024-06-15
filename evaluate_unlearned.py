import argparse
import json
import os

import torch
from bert_score import BERTScorer
from bleurt import score
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from peft import PeftModel
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

import utils
from config.metrics_config import bnb_config, config


def jaccard_similarity(text1, text2):
    set1 = set(word_tokenize(text1))
    set2 = set(word_tokenize(text2))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rouge_score(text1, text2):
    # print the hypothesis and the reference
    rouge = Rouge()
    scores = rouge.get_scores(text1, text2)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']


def bleurt_score(references, candidates):
    checkpoint = "/nlp/data/gydou/llm_copyright/bleurt/BLEURT-20"
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    assert isinstance(scores, list) and len(scores) == 1
    return scores


# model_name = "bert-large-cased"
def cosine_similarity_score(text1, text2, model_name, model_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.float16,
    )
    # Load the pre-trained LLM and tokenizer
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = "auto"

    model = AutoModel.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config,
                                      cache_dir=model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input texts
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Generate the embeddings using the LLM
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity between the embeddings
    cosine_sim = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())

    return cosine_sim[0][0]


def bert_score_base(candidate, reference):
    scorer = BERTScorer(model_type='bert-base-uncased', lang='en')
    P, R, F1 = scorer.score(candidate, reference)
    return P.mean(), R.mean(), F1.mean()


def calculate_perplexity(model, tokenizer, text):
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt')
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss calculation
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity


def main(json_file_path, base_model_name, use_fine_tuned_model, fine_tuned_model_name, fine_tuned_filename, model_dir):
    torch.cuda.empty_cache()
    load_dotenv()

    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    access_token = os.environ.get('HF_ACCESS_TOKEN')
    if not access_token:
        raise ValueError("Hugging Face access token not found. Please set the HF_ACCESS_TOKEN environment variable.")

    if base_model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
        print(f"Loading model {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config,
                                                     device_map="auto", token=access_token,
                                                     cache_dir=model_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=access_token)

        tokenizer.pad_token = "[PAD]"

        if use_fine_tuned_model:
            assert fine_tuned_model_name is not None
            assert fine_tuned_filename is not None
            print(f"Loading fine-tuned model {fine_tuned_model_name} with filename {fine_tuned_filename}")
            fine_tuned_model_path = os.path.join(model_dir, fine_tuned_model_name)

            print("Fine-tuned model path: ", fine_tuned_model_path)
            _ = utils.load_checkpoint(model, checkpoint_dir=fine_tuned_model_path, filename=fine_tuned_filename)
    elif base_model_name in ["meta-llama/Meta-Llama-3-8B"]:
        print(f"Loading model {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                     quantization_config=bnb_config,
                                                     device_map="auto",
                                                     token=access_token,
                                                     cache_dir=model_dir)

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
    else:
        raise ValueError("The model type is not supported.")

    model.generation_config.cache_implementation = "static"

    results = []
    total_jaccard_sim = 0
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0
    total_bleurt_score = 0
    total_cosine_sim = 0
    total_F1 = 0
    total_perplexity = 0

    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    system_prompt = (
        "Continue the story based on the given context from the book. "
        "Generate a coherent and engaging continuation that follows the plot, maintains consistency with the characters"
        "and captures the writing style of the original book."
    )

    non_output = 0
    total_data = len(data)
    for item in data:
        question = item['question']
        answer = item['answer']

        if base_model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
            inputs_qa = f"### Question: {question}\n ### Answer: "
            input_question = base_prompt.format(system_prompt=system_prompt, user_prompt=inputs_qa)
        else:
            input_question = f"Please keep generating a coherent continuation that follows the plot: {question}"

        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        inputs = tokenizer(input_question, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            outputs = compiled_model.generate(inputs.input_ids, max_length=500, do_sample=False)


        generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        generated_text = generated_output[0][len(input_question) - 1:].strip()

        if generated_text == "":
            non_output += 1
            print("We have no output for this question")
            generated_text = "No output generated."
        generated_text = "Answer:"+generated_text
        # Calculate similarity scores
        jaccard_sim = jaccard_similarity(generated_text, answer)
        rouge_scores = rouge_score(generated_text, answer)
        # bleurt_sim = bleurt_score([answer], [generated_text])
        bleurt_sim = [0]
        # cosine_sim = cosine_similarity_score(generated_text, answer, "bert-large-cased", model_dir=model_dir)
        cosine_sim = 0
        # _, _, F1 = bert_score_base([generated_text], [answer])
        F1 = 0
        perplexity = calculate_perplexity(model, tokenizer, generated_text)


        result = {
            'question': question,
            'generated_text': generated_text,
            'answer': answer,
            'jaccard_similarity': jaccard_sim,
            'rouge_scores': {
                'rouge-1': rouge_scores[0],
                'rouge-2': rouge_scores[1],
                'rouge-l': rouge_scores[2]
            },
            'bleurt_score': bleurt_sim[0],
            'cosine_similarity': cosine_sim,
            'bert_score F1': F1,
            'perplexity': perplexity
        }
        results.append(result)

        total_jaccard_sim += jaccard_sim
        total_rouge_1 += rouge_scores[0]
        total_rouge_2 += rouge_scores[1]
        total_rouge_l += rouge_scores[2]
        total_bleurt_score += bleurt_sim[0]
        total_cosine_sim += cosine_sim
        total_F1 += F1
        total_perplexity += perplexity

    num_items = len(data)
    mean_jaccard_sim = total_jaccard_sim / num_items
    mean_rouge_1 = total_rouge_1 / num_items
    mean_rouge_2 = total_rouge_2 / num_items
    mean_rouge_l = total_rouge_l / num_items
    mean_bleurt_score = total_bleurt_score / num_items
    mean_cosine_sim = total_cosine_sim / num_items
    mean_F1 = total_F1 / num_items
    mean_perplexity = total_perplexity / num_items

    if use_fine_tuned_model:
        print(f"Here is the result of the model {fine_tuned_model_name} on the book {json_file_path} with filename {fine_tuned_filename}:")
    else:
        print(f"Here is the result of the model {base_model_name} on the book {json_file_path}:")
    print("Mean Scores:")
    print(f"Jaccard Similarity: {mean_jaccard_sim}")
    print(f"ROUGE-1: {mean_rouge_1}")
    print(f"ROUGE-2: {mean_rouge_2}")
    print(f"ROUGE-L: {mean_rouge_l}")
    print(f"BLEURT Score: {mean_bleurt_score}")
    print(f"Cosine Similarity: {mean_cosine_sim}")
    print(f"BERT Score F1: {mean_F1}")
    print(f"Perplexity: {mean_perplexity}")

    print(f"Number of items with no output: {non_output} out of {total_data}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Evaluate the level of copyright infringement')
    parser.add_argument('--model_dir', type=str, help='the directory of the models you saved locally')
    parser.add_argument("--text_completion_data_path", type=str, help="the path to your text completion data")
    args = parser.parse_args()

    print("the base model is", config['base_model_name'])
    print("config['use_fine_tuned_model'] is set to", config['use_fine_tuned_model'], "and the book name is", config['book_name_json'])
    main(json_file_path=os.path.join(args.text_completion_data_path, config['book_name_json']), base_model_name=config['base_model_name'],
         use_fine_tuned_model=config['use_fine_tuned_model'],
         fine_tuned_model_name=config['fine_tuned_model_name'],
         fine_tuned_filename=config['fine_tuned_filename'],
         model_dir=args.model_dir)
