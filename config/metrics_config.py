import torch
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

config = {
    'base_model_name':'meta-llama/Meta-Llama-3-8B',
    'use_fine_tuned_model': True,
    'fine_tuned_model_name': 'llama3-8b-harry-potter',
    'fine_tuned_filename':'llama3_tv_random_loss_weight_saliency_saved.pth',
    'book_name_json': 'harry_potter_3.json',
}
