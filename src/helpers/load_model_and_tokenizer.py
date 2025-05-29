import os
from typing import Literal, Type, Union, Any, TypedDict

import torch
from torch import Tensor
from torch.nn import Module
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from src.configs.path_config import save_model_path
from src.configs.training_args import model_name
from src.helpers.logging_utils import setup_logger
from src.models.collapsed_ner_model_with_weights_and_masking import CollapsedNERModel
from src.models.masked_weighted_loss_model import MaskedWeightedLossModel

logger = setup_logger()

ModelType = Union[MaskedWeightedLossModel, CollapsedNERModel]


class ModelConfig(TypedDict):
    name: str
    instance: Type[ModelType]
    extra_params: dict[str, Any]

def get_device() -> Literal["mps", "cuda", "cpu"]:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


model_name_to_params: dict[str, ModelConfig] = {
    "masked_bert": {
        "name": "bert-base-uncased",
        "instance": MaskedWeightedLossModel,
        "extra_params": {"num_labels": 3},
    },
    "masked_ner_bert": {
        "name": "dslim/distilbert-NER",
        "instance": CollapsedNERModel,
        "extra_params": {},  # Add this to keep consistent
    },
}




def get_tokenizer_only() -> PreTrainedTokenizerFast:
    """Get only the tokenizer without loading the model."""
    selected_information = model_name_to_params[model_name]

    if os.path.exists(save_model_path):
        tokenizer = AutoTokenizer.from_pretrained(save_model_path.parent)
    else:
        tokenizer = AutoTokenizer.from_pretrained(selected_information["name"])

    return tokenizer




def load_model_and_tokenizer(
    model_params: list[float],
) -> tuple[ModelType, PreTrainedTokenizerFast]:
    """Load both model and tokenizer with the specified class weights."""
    selected_information = model_name_to_params[model_name]

    # Reuse the tokenizer function
    tokenizer = get_tokenizer_only()

    # Load the base model
    base_model = AutoModelForTokenClassification.from_pretrained(
        selected_information["name"], **selected_information.get("extra_params", {})
    )

    # Create class weights tensor and instantiate the model
    class_weights = torch.tensor(model_params, dtype=torch.float)
    model_class = selected_information["instance"]
    model = model_class(base_model, class_weights)
    model.to(get_device())

    # Load trained weights if available
    if os.path.exists(save_model_path):
        logger.info("Using a trained model...")
        model.load_state_dict(torch.load(save_model_path))
    else:
        logger.info("Using a pretrained base model...")

    return model, tokenizer
