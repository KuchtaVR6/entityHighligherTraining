import logging

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from load_helpers import load_large_dataset, tokenize_and_align_labels_batch
from train import label_map, WeightedLossModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def infer(model, infer_dataset, label_map):
    model.eval()

    for batch in tqdm(infer_dataset):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        labels = batch["labels"]
        tensor_labels = torch.tensor(labels).unsqueeze(0)
        zero_tensor = torch.zeros_like(tensor_labels)

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor)
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
            predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]

            print(predicted_labels)

if __name__ == '__main__':
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset('infer/input.json')

    tokenizer = AutoTokenizer.from_pretrained("./results")
    base_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)
    class_weights = torch.tensor([0, 0, 0], dtype=torch.float)

    model = WeightedLossModel(base_model, class_weights)

    model.load_state_dict(torch.load("./results/custom_model.pth"))

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    print(infer(model, val_dataset, label_map))
