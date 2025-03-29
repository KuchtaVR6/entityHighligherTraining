import os
import gc
import torch
import torch.nn as nn
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from torch.utils.data import DataLoader
from typing import List
from collections import Counter

from load_helpers import load_large_dataset, tokenize_and_align_labels_batch

label_map = {'O': 0, 'B-EMPH': 1, 'I-EMPH': 2}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightedLossModel(nn.Module):
    def __init__(self, base_model, class_weights, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits

def calc_distribution(train_dataset: Dataset) -> List[float]:
    """Calculates the distribution of BIO tags in the dataset."""
    tag_counter = Counter()

    for example in train_dataset:
        labels = example["labels"]
        tag_counter.update(labels)

    total_tags = sum(tag_counter.values())
    return [tag_counter[tag] / total_tags for tag in sorted(tag_counter)]

def train_model(train_dataset: Dataset, val_dataset: Dataset, data_collator):
    """Trains the model using provided datasets."""
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        logging_steps=50,
        weight_decay=0.3,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_steps=10000,
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Model saved.")

    return trainer.state.log_history

if __name__ == '__main__':
    logger.info("Loading datasets...")
    train_dataset = load_large_dataset('data/split_files/part_2.json')
    val_dataset = load_large_dataset('data/toy_eval.json')

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    logger.info("Calculating class weights...")
    class_weights = [1 / share for share in calc_distribution(train_dataset)]

    model = WeightedLossModel(base_model, class_weights)

    if os.path.exists("./results/custom_model.pth"):
        logger.info("Stating with an existing model...")
        model.load_state_dict(torch.load("./results/custom_model.pth"))

    del base_model  # Free base model reference
    gc.collect()

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=2)

    logs = train_model(train_dataset, val_dataset, data_collator)

    os.makedirs("./results", exist_ok=True)

    torch.save(model.state_dict(), "./results/custom_model.pth")
    tokenizer.save_pretrained("./results")