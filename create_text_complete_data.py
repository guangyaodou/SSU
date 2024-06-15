import os
import json
import random
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize

def generate_dataset(book_path, output_dir, book_name, num_samples=50):
    with open(book_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    sentences = sent_tokenize(book_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    dataset = []
    for _ in range(num_samples):
        start_index = random.randint(0, len(tokenized_sentences) - 2)
        question_tokens = []
        answer_tokens = []

        while len(question_tokens) < 200 and start_index < len(tokenized_sentences) - 1:
            question_tokens.extend(tokenized_sentences[start_index])
            start_index += 1

        if len(question_tokens) >= 200:
            question_tokens = question_tokens[:200]

            answer_end_index = start_index
            while len(answer_tokens) < 150 and answer_end_index < len(tokenized_sentences):
                answer_tokens.extend(tokenized_sentences[answer_end_index])
                answer_end_index += 1

            answer_tokens = answer_tokens[:150]

            question = ' '.join(question_tokens)
            answer = ' '.join(answer_tokens)

            # Replace the "â" symbol with a single quotation mark
            question = question.replace('â', "'")
            answer = answer.replace('â', "'")

            dataset.append({
                'question': question,
                'answer': answer
            })

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'{book_name}.json')
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    print(f"Dataset generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text completion data.')
    parser.add_argument('--book_path', type=str, help='the path to the book')
    parser.add_argument('--output_dir', type=str, help='the path to the output directory')
    parser.add_argument("--book_name", type=str, help="the name of the book")
    parser.add_argument("--num_samples", type=int, help="number of samples being generated")
    args = parser.parse_args()
    generate_dataset(args.book_path, args.output_dir, args.book_name, args.num_samples)
