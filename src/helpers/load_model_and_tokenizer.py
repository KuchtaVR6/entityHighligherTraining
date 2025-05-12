import os

import torch
from src.configs.path_config import save_model_path
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.helpers.logging_utils import setup_logger
from src.configs.training_args import model_name

from src.models.masked_weighted_loss_model import MaskedWeightedLossModel
from src.models.collapsed_ner_model_with_weights_and_masking import CollapsedNERModel

logger = setup_logger()

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

model_name_to_params = {
    "masked_bert": {
        "name": "bert-base-uncased",
        "instance": MaskedWeightedLossModel,
        "extra_params": {"num_labels": 3}
    },
    "masked_ner_bert": {
        "name": "dslim/distilbert-NER",
        "instance": CollapsedNERModel,
        "extra_params": {}  # Add this to keep consistent
    }
}

def get_tokenizer_only():
    """Get only the tokenizer without loading the model."""
    selected_information = model_name_to_params[model_name]

    if os.path.exists(save_model_path):
        tokenizer = AutoTokenizer.from_pretrained(save_model_path.parent)
    else:
        tokenizer = AutoTokenizer.from_pretrained(selected_information["name"])

    return tokenizer

def load_model_and_tokenizer(model_params):
    """Load both model and tokenizer with the specified class weights."""
    selected_information = model_name_to_params[model_name]

    # Reuse the tokenizer function
    tokenizer = get_tokenizer_only()

    # Load the base model
    base_model = AutoModelForTokenClassification.from_pretrained(
        selected_information["name"],
        **selected_information.get("extra_params", {})
    )

    # Create class weights tensor and instantiate the model
    class_weights = torch.tensor(model_params, dtype=torch.float)
    model = selected_information["instance"](base_model, class_weights)
    model.to(get_device())

    # Load trained weights if available
    if os.path.exists(save_model_path):
        logger.info("Using a trained model...")
        model.load_state_dict(torch.load(save_model_path))
    else:
        logger.info("Using a pretrained base model...")

    return model, tokenizer
