from collections import defaultdict
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification, PreTrainedModel

from src.configs.path_config import eval_data_path
from src.helpers.label_map import label_map
from src.helpers.load_helpers import (
    load_large_dataset,
    tokenize_and_align_labels_batch,
)
from src.helpers.load_model_and_tokenizer import load_model_and_tokenizer
from src.helpers.logging_utils import setup_logger

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = setup_logger()
logger = logging.getLogger(__name__)


def compute_accuracy(
    model: PreTrainedModel,
    val_dataloader: DataLoader[dict[str, Any]],
    label_map: dict[str, int],
) -> dict[str, float]:
    model.eval()
    correct_counts: dict[int, int] = defaultdict(int)
    total_counts: dict[int, int] = defaultdict(int)
    id_to_label = {v: k for k, v in label_map.items()}

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        input_ids = torch.tensor(batch["input_ids"]).to(model.base_model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(
            model.base_model.device
        )
        labels = torch.tensor(batch["labels"]).to(model.base_model.device)

        zero_tensor = torch.zeros_like(labels)

        with torch.no_grad():
            _, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor
            )
            predictions = torch.argmax(logits, dim=-1)

        for i in range(predictions.shape[0]):
            pred_labels = predictions[i].tolist()
            true_labels = labels[i].tolist()

            for pred, true in zip(pred_labels, true_labels, strict=False):
                if true != -100:
                    total_counts[true] += 1
                    if pred == true:
                        correct_counts[true] += 1

    per_class_accuracy = {
        id_to_label[class_id]: correct_counts[class_id] / total_counts[class_id]
        for class_id in total_counts
        if total_counts[class_id] > 0
    }

    total_correct = sum(correct_counts.values())
    total_samples = sum(total_counts.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    per_class_accuracy["overall_accuracy"] = overall_accuracy

    return per_class_accuracy


if __name__ == "__main__":
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset(eval_data_path)

    model, tokenizer = load_model_and_tokenizer([0, 0, 0])

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

    accuracy = compute_accuracy(model, val_dataloader, label_map)
    logger.info(f"Model accuracy: {accuracy}")
