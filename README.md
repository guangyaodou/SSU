# Stable Sequential Unlearning

This is the code and repo for Stable Sequential Unlearning (SSU). 

## Installation
You need to install packages described in [requirements.txt](requirements.txt) with specified version number. We strongly recommend using a Conda environment. You will also need
a .env file to store you HF_ACCESS_TOKEN to download Llama models.

## Dataset
For each book, you need to obtain a txt version from either purchasing or downloading from websites such as the [Project Gutenberg](https://gutenberg.org/).

After obtaining the txt file, you can use the [create_training_data.py](create_training_data.py) to create data format used for training. Specifically, you 
need to specify the path to the txt file, the output directory, and the book name:
```angular2html
python create_training_data.py --book_path BOOK_PATH --output_dir OUTPUT_DIR --book_name BOOK_NAME
```
In order to create data that will be used for evaluation, use the same txt file, determine number os samples, and run [create_text_complete_data.py](create_text_complete_data.py):
```angular2html
python create_text_complete_data.py --book_path BOOK_PATH --output_dir OUTPUT_DIR --book_name BOOK_NAME --num_samples NUM_SAMPLES
```

## Fine-tuning 
To fine-tune the model, you can use the [fine_tune_books.py](fine_tune_books.py) script. 

The script create_training_data.py requires the following parameters:

* --model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.
* --data_dir: Specifies the directory containing all the books you have saved. This directory will be used to load the data for the unlearning process.
* --book_dir: Specifies the directory of the specific book to be used you want to unlearn.
* --visualize_data: A flag to visualize the dataset. If set, the script will print the details of the dataset batches. Default is False.
* --book_corpus_norm_data: Specifies the directory of the book corpus that does not contain the books you want to unlearn. This is used for GA-based methods.
* --Lora: A flag to use LoRA (Low-Rank Adaptation) parameterization. If set, the script will use LoRA for fine-tuning. Default is False.

Besides, for Llama3-8b model, I created a folder called "llama3-8b-harry-potter" to store all the checkpoints of fine-tuned models. You can create a similar folder for your fine-tuned models.

You also need to adjust the fine-tuning config file [fine_tune_config.py](config/fine_tune_config.py):
* base_model_name: Specifies the base model name. This is the base model you want to unlearn.
* baseline_method: True if we are using a GA baseline method, False otherwise.
* idk_as_y_random: True if you are using GA + IDK + Maintain Loss, and False for GA + Mismatch + Maintain Loss.
* random_loss_epsilon: Specifies the epsilon value for the random labeling loss.
* weight_saliency: True if you are using the saliency-based weight update, False otherwise.
* num_std_dev : 0 if you just want to use mean, or however many standard deviations away from mean you want to use for the saliency-based weight update.
* use_fine_tuned_model: True if you want to use a fine-tuned model to unlearn, False otherwise.
* fine_tuned_model_name: The folder name that stores the checkpoint of your fine-tuned model that will be unlearned.
* load_filename: The checkpoint file name of your fine-tuned model.
* save_filename: The checkpoint file name of your fine-tuned model after fine-tuning.
* batch_size: The batch size for fine-tuning.
* lr: The learning rate for fine-tuning.
* max_unlearn_steps: The maximum number of unlearning steps.
* max_d_nor_unlearn_steps: The maximum number of learning steps on books collected for GA-based algorithms.
* output_name: the folder name that stores the checkpoint of your fine-tuned model after fine-tuning (usually the same with fine_tuned_model_name).
* show_sample_output: True if you want to show the sample output of the model, False otherwise.

## Applying Task Vector

To apply the task vector, you can use the [main.py](main.py) script.

The script requires the following parameters:
* --model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.

You also need to adjust the task vector config file [tv_config.py](config/tv_config.py):
* use_fine_tuned_model: True if you want to use a fine-tuned model as the original model ($\theta_o$), False otherwise (It will be always False in our experiment). 
* base_model_name: Specifies the base model name. This is the base model you want to unlearn.
* fine_tuned_model_name: The folder name that stores the checkpoint of your fine-tuned model that will be used as the base model.
* fine_tuned_filename: The checkpoint file name of your fine-tuned model (and will also store unlearned model after negating task vectors).
* tv_ft_filename: The checkpoint file name of your fine-tuned model that will be used for task vector fine-tuning ($theta_{ft}$).
* save_file_name: The checkpoint file name of your fine-tuned model after task vector fine-tuning.
* show_sample_output: True if you want to show the sample output of the model, False otherwise.

The unlearned modified model's checkpoint will also be stored in the directory fine_tuned_model_name.

## Evaluation
For Jaccard and Rouge, you can use [evaluate_unlearned.py](evaluate_unlearned.py) script.

The script evaluate_unlearned.py requires the following parameters:
* --model_dir: Specifies the directory where your models are saved locally. This is essential for loading pre-trained or fine-tuned models.
* -- text_completion_data_path: Specifies the directory where the text completion data (data used for evaluation) is stored.

You need to adjust the evaluation config file [metrics_config.py](config/metrics_config.py):
* base_model_name: Specifies the base model name. This is the base model you want to unlearn.
* use_fine_tuned_model: True if you want to use a modified model (including fine-tuned or the unlearned model after task vector negation), False otherwise.
* fine_tuned_model_name: The folder name that stores the checkpoint of your fine-tuned model as well as the unlearned modified model.
* fine_tuned_filename: The checkpoint file name of your fine-tuned or modified model.
* book_name_json: The book name in JSONL format that you want to evaluate.

To evaluate benchmark performance, please refer to other public github repositories that allow you to evaluate the performance of the model. 