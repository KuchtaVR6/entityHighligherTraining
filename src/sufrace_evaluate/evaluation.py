import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification

from src.load_helpers import load_large_dataset, tokenize_and_align_labels_batch
from train import label_map, WeightedLossModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_accuracy(model, val_dataloader, label_map):
    model.eval()
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    id_to_label = {v: k for k, v in label_map.items()}

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        input_ids = torch.tensor(batch["input_ids"]).to(model.base_model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(model.base_model.device)
        labels = torch.tensor(batch["labels"]).to(model.base_model.device)

        zero_tensor = torch.zeros_like(labels)

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor)
            predictions = torch.argmax(logits, dim=-1)

        for i in range(predictions.shape[0]):
            pred_labels = predictions[i].tolist()
            true_labels = labels[i].tolist()

            for pred, true in zip(pred_labels, true_labels):
                if true != -100:
                    total_counts[true] += 1
                    if pred == true:
                        correct_counts[true] += 1

    per_class_accuracy = {
        id_to_label[class_id]: correct_counts[class_id] / total_counts[class_id]
        for class_id in total_counts if total_counts[class_id] > 0
    }

    total_correct = sum(correct_counts.values())
    total_samples = sum(total_counts.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    per_class_accuracy["overall_accuracy"] = overall_accuracy

    return per_class_accuracy

if __name__ == '__main__':
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset('../../data/toy_eval.json')

    tokenizer = AutoTokenizer.from_pretrained("../../results")
    base_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    class_weights = torch.tensor([0, 0, 0], dtype=torch.float)

    model = WeightedLossModel(base_model, class_weights)
    model.load_state_dict(torch.load("../../results/custom_model.pth"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

    print(compute_accuracy(model, val_dataloader, label_map))
