import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

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

config = {
    'use_fine_tuned_model': False,
    'base_model_name':'meta-llama/Meta-Llama-3-8B',
    ################################### only if use_fine_tuned_model is True ###################################
    'fine_tuned_model_name': 'llama3-8b-harry-potter',
    'fine_tuned_filename':'llama_3_8b_hp_checkpoint_base_200.pth',
    ############################################################################################################
    'tv_ft_filename': 'llama3_tv_random_loss_weight_saliency.pth',
    'save_file_name': 'llama3_tv_random_loss_weight_saliency_saved.pth',
    'show_sample_output': False
}

