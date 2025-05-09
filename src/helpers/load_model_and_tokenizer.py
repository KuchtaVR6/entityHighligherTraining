import os

import torch
from src.configs.path_config import save_model_path
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.helpers.logging_utils import setup_logger
from src.models.weighted_loss_model import WeightedLossModel

logger = setup_logger()

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model_and_tokenizer(model_params):
    if os.path.exists(save_model_path):
        tokenizer = AutoTokenizer.from_pretrained(save_model_path.parent)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    base_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    class_weights = torch.tensor(model_params, dtype=torch.float)

    model = WeightedLossModel(base_model, class_weights)
    model.to(get_device())

    if os.path.exists(save_model_path):
        logger.info("Using a trained model...")
        model.load_state_dict(torch.load(save_model_path))
    else:
        logger.info("Using a pretrained base model...")


    return model, tokenizer
