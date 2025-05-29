from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.configs.path_config import eval_data_path
from src.helpers.label_map import label_map
from src.helpers.load_helpers import load_large_dataset, tokenize_text
from src.helpers.load_model_and_tokenizer import load_model_and_tokenizer
from src.utils.logger import setup_logger

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logger
logger: logging.Logger = setup_logger(__name__)


def detokenize(tokens: list[str]) -> str:
    """Convert tokenized output back to readable text.

    Args:
        tokens: List of tokens to detokenize

    Returns:
        str: Detokenized text
    """
    words: list[str] = []
    for token in tokens:
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue  # Ignore special tokens
        if token.startswith("##"):
            if words:  # Ensure there's at least one word to append to
                words[-1] += token[2:]  # Merge subword tokens
        else:
            words.append(token)
    return " ".join(words)


def infer(
    model: PreTrainedModel,
    infer_dataset: Any,  # Could be more specific with Dataset type
    label_map: dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Run inference on the given dataset using the provided model.

    Args:
        model: The pre-trained model to use for inference
        infer_dataset: Dataset to run inference on
        label_map: Mapping from label names to indices
        tokenizer: Tokenizer used for the model
    """
    model.eval()
    reverse_label_map = {v: k for k, v in label_map.items()}

    for batch in tqdm(infer_dataset):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        zero_tensor = torch.zeros_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

            # Get predicted labels using the reverse label map
            predicted_labels = [reverse_label_map.get(idx, "O") for idx in predictions]

            # Get tokens from tokenizer
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

            # Prepare XML output for highlighting spans
            highlighted_text: list[str] = []
            current_entity: str | None = None
            current_entity_tokens: list[str] = []

            for token, label in zip(tokens, predicted_labels, strict=False):
                if token in {"[CLS]", "[SEP]", "[PAD]"}:
                    continue

                if label.startswith("B-"):
                    if current_entity is not None and current_entity_tokens:
                        # Close previous entity
                        entity_text = tokenizer.convert_tokens_to_string(
                            current_entity_tokens
                        )
                        highlighted_text.append(
                            f"<span class='{current_entity}'>{entity_text}</span>"
                        )
                        current_entity_tokens = []
                    current_entity = label[2:]  # Remove 'B-' prefix
                    current_entity_tokens.append(token)
                elif label.startswith("I-") and current_entity == label[2:]:
                    current_entity_tokens.append(token)
                else:
                    if current_entity_tokens:
                        # Close any open entity
                        entity_text = tokenizer.convert_tokens_to_string(
                            current_entity_tokens
                        )
                        highlighted_text.append(
                            f"<span class='{current_entity}'>{entity_text}</span>"
                        )
                        current_entity_tokens = []
                        current_entity = None
                    highlighted_text.append(token)

            # Add any remaining entity
            if current_entity_tokens and current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity_tokens)
                highlighted_text.append(
                    f"<span class='{current_entity}'>{entity_text}</span>"
                )

            # Log the highlighted text
            logger.info(" ".join(highlighted_text))


if __name__ == "__main__":
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset(eval_data_path)

    # Load model and tokenizer with zero weights since we're just doing inference
    model, tokenizer = load_model_and_tokenizer(
        model_params=[0, 0, 0]
    )  # Using zero weights for inference

    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer")
        sys.exit(1)

    # Tokenize the dataset
    tokenized_dataset = val_dataset.map(
        lambda x: tokenize_text(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # Run inference
    infer(model, tokenized_dataset, label_map, tokenizer)
