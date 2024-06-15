import os
import json
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize

def generate_dataset(book_path, output_dir, book_name):
    with open(book_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    sentences = sent_tokenize(book_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Create a flat list of all words in the book
    all_words = [word for sentence in tokenized_sentences for word in sentence]

    dataset = []
    index = 0

    # Generate question-answer pairs until we reach the end of all_words
    while index + 400 <= len(all_words):  # Ensure we have enough words for both question and answer
        question_words = all_words[index:index+200]
        answer_words = all_words[index+200:index+400]

        question = ' '.join(question_words)
        answer = ' '.join(answer_words)

        # Replace the "â" symbol with a single quotation mark
        question = question.replace('â', "'")
        answer = answer.replace('â', "'")

        dataset.append({
            'question': question,
            'answer': answer
        })

        index += 400  # Move to the next segment

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'{book_name}_train.json')
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    print(f"Dataset generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QA dataset from text.')
    parser.add_argument('--book_path', type=str, help='the path to the book')
    parser.add_argument('--output_dir', type=str, help='the path to the output directory')
    parser.add_argument("--book_name", type=str, help="the name of the book")
    args = parser.parse_args()
    generate_dataset(args.book_path, args.output_dir, args.book_name)
