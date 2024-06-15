import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# Define BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_16bit=True,
#     bnb_16bit_quant_type="nf16",
#     bnb_16bit_compute_dtype=torch.float16,
#     bnb_16bit_use_double_quant=True,
# )

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
)

# Define LoraConfig
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Add any other configurations here
config = {
    'base_model_name': 'meta-llama/Meta-Llama-3-8B',
    'baseline_method':False,
    ################# only if baseline_method is True #################
    'idk_as_y_random': False,
    ###################################################################
    "random_loss_epsilon": 1,
    'weight_saliency':True,
    'num_std_dev' : 0,
    'use_fine_tuned_model': False,
    ############################## only if use_fine_tuned_model is True ##############################
    'fine_tuned_model_name': 'llama3-8b-harry-potter',
    'load_filename': 'llama3_tv.pth',
    ##################################################################################################
    'save_filename': 'llama3_tv_random_loss_weight_saliency.pth',
    'batch_size': 4,
    'lr': 1e-4,
    'max_unlearn_steps': 200,
    'max_d_nor_unlearn_steps': 0,
    'output_name': 'llama3-8b-harry-potter',
    'show_sample_output': False
}
