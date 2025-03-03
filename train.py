import json
import torch
import torch.nn as nn
import re
import logging
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
from typing import List

label_map = {'O': 0, 'B-EMPH': 1, 'I-EMPH': 2}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeightedLossModel(nn.Module):
    def __init__(self, base_model, class_weights, dropout_prob=0.25):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


def load_dataset(file_path: str) -> Dataset:
    """Loads and processes dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    def remove_span_tags(text):
        return re.sub(r'<\/?.*?span.*?>', '', text)

    def transform_to_bio(annotated):
        tags, current_tag = [], None
        text_with_tags = re.split(r'(<[^>]+>)', annotated)

        for token in text_with_tags:
            if re.match(r'<[^/]+>', token):
                current_tag = token.strip('<>')
            elif re.match(r'</[^>]+>', token):
                current_tag = None
            elif token.strip():
                words = token.split()
                for i, word in enumerate(words):
                    tag_prefix = 'B' if i == 0 else 'I'
                    tags.append(f'{tag_prefix}-EMPH' if current_tag else 'O')
        return tags

    return Dataset.from_dict({
        "raw": [remove_span_tags(entry) for entry in tqdm(data, desc="Processing raw text")],
        "annotated": data,
        "tags": [transform_to_bio(entry) for entry in tqdm(data, desc="Generating BIO tags")]
    })


def calc_distribution(train_dataset: Dataset) -> List[float]:
    """Calculates label distribution for weighted loss."""
    counts = [0] * len(label_map)

    for entry in tqdm(train_dataset, desc="Calculating label distribution"):
        for tag in entry["tags"]:
            counts[label_map[tag]] += 1

    total = sum(counts)
    return [count / total for count in counts]


def plot_loss_logs(logs):
    """Plots training and evaluation loss."""
    train_loss, eval_loss, steps = [], [], []

    for log in logs:
        if "loss" in log:
            train_loss.append(log["loss"])
        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])
        if "step" in log:
            steps.append(log["step"])

    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(train_loss)], train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(steps[:len(eval_loss)], eval_loss, label='Evaluation Loss')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid()
    plt.show()


def tokenize_and_align_labels(examples):
    """Tokenizes text and aligns labels accordingly."""
    tokenized_inputs = tokenizer(examples["raw"], truncation=True, padding='max_length', max_length=256,
                                 is_split_into_words=False)
    labels = []

    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids, previous_word_idx = [], None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(label_map["O"])
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]] if word_idx < len(label) else label_map["O"])
            else:
                label_ids.append(label_ids[-1])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_model(train_dataset: Dataset, val_dataset: Dataset):
    """Trains the model using provided datasets."""
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        logging_steps=50,
        weight_decay=0.3,
        logging_dir='./logs',
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model("./results")
    logger.info("Training complete. Model saved.")

    plot_loss_logs(trainer.state.log_history)
    return trainer.state.log_history


if __name__ == '__main__':
    logger.info("Loading datasets...")
    train_dataset = load_dataset('data/train_data.json')
    val_dataset = load_dataset('data/eval_data.json')

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    logger.info("Calculating class weights...")
    class_weights = [1 / share for share in calc_distribution(train_dataset)]
    model = WeightedLossModel(base_model, class_weights)

    logs = train_model(train_dataset, val_dataset)