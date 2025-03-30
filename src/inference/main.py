import torch
import logging

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from src.inference.compute_logits import compute_logits
from src.load_helpers import load_large_dataset, tokenize_and_align_labels_batch
from train import label_map, WeightedLossModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset('infer/toy_train.json')

    tokenizer = AutoTokenizer.from_pretrained("results")
    base_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    class_weights = torch.tensor([0, 0, 0], dtype=torch.float)

    model = WeightedLossModel(base_model, class_weights)

    model.load_state_dict(torch.load("results/custom_model.pth"))

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=data_collator)

    print(compute_logits(model, val_dataloader, tokenizer))
