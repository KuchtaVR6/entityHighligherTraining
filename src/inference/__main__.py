import json
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForTokenClassification,
)

from src.configs.path_config import eval_data_path
from src.configs.training_args import inference_batch_size, loss_span_proximity
from src.helpers.label_map import label_map
from src.helpers.load_helpers import (
    load_large_dataset,
    remove_span_tags,
    tokenize_and_align_labels_batch,
)
from src.helpers.load_model_and_tokenizer import load_model_and_tokenizer
from src.inference.compute_logits import compute_logits

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset(eval_data_path)

    text_column = val_dataset["text"]

    # Load model and tokenizer with zero weights since we're just doing inference
    model, tokenizer = load_model_and_tokenizer(model_params=[0, 0, 0])

    def mapped_data(
        x: dict[str, list[str]],
    ) -> dict[str, torch.Tensor | list[list[int]] | None]:
        return tokenize_and_align_labels_batch(
            x, tokenizer, label_map, proximity=loss_span_proximity
        )

    val_dataset = val_dataset.map(mapped_data, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForTokenClassification(tokenizer)
    val_dataloader = DataLoader(
        val_dataset, batch_size=inference_batch_size, collate_fn=data_collator
    )

    logger.info("Computing logits...")
    results = compute_logits(model, val_dataloader, tokenizer)

    for i, result in enumerate(results):
        result["text"] = remove_span_tags(text_column[i])

    output_path = "logits/inference_logits.jsonl"
    logger.info(f"Saving results to {output_path}")

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    logger.info("Finished saving logits.")
