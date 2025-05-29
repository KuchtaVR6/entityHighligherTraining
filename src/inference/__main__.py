import json
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.configs.path_config import eval_data_path, save_model_path
from src.configs.training_args import inference_batch_size, loss_span_proximity
from src.helpers.label_map import label_map
from src.helpers.load_helpers import (
    load_large_dataset,
    remove_span_tags,
    tokenize_and_align_labels_batch,
)
from src.inference.compute_logits import compute_logits
from src.models.masked_weighted_loss_model import MaskedWeightedLossModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset(eval_data_path)

    # Save the text column for later use
    text_column = val_dataset["text"]

    tokenizer = AutoTokenizer.from_pretrained(save_model_path.parent)
    base_model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )
    class_weights = torch.tensor([0, 0, 0], dtype=torch.float)

    model = MaskedWeightedLossModel(base_model, class_weights)
    model.load_state_dict(torch.load(save_model_path))

    def mapped_data(
        x: dict[str, list[str]],
    ) -> dict[str, torch.Tensor | list[list[int]] | None]:
        return tokenize_and_align_labels_batch(
            x, tokenizer, label_map, proximity=loss_span_proximity
        )

    # Map data and remove the text column
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
