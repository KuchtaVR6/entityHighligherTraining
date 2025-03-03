import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
from datasets import Dataset
from typing import List
import torch
import torch.nn as nn
import re

class WeightedLossModel(nn.Module):
    def __init__(self, base_model, class_weights, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout layer
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits

def load_dataset(file_path: str) -> Dataset:
    with open(file_path, 'r') as f:
        data = json.load(f)

    def remove_span_tags(text):
        # This regex pattern matches <span> tags and their attributes
        span_tag_pattern = re.compile(r'<\/?.*?span.*?>')
        return re.sub(span_tag_pattern, '', text)

    def transform_to_bio(annotated):
        tags = []
        text_with_tags = re.split(r'(<[^>]+>)', annotated)
        current_tag = None

        for token in text_with_tags:
            if re.match(r'<[^/]+>', token):
                current_tag = token.strip('<>')
            elif re.match(r'</[^>]+>', token):
                current_tag = None
            elif token.strip():
                words = token.split()
                for i, word in enumerate(words):
                    if current_tag:
                        tag_prefix = 'B' if i == 0 else 'I'
                        tags.append(f'{tag_prefix}-EMPH')
                    else:
                        tags.append('O')

        return tags

    return Dataset.from_dict({
        "raw": [remove_span_tags(entry) for entry in data],
        "annotated": [entry for entry in data],
        "tags": [transform_to_bio(entry) for entry in data]
    })


label_map = {
    'O': 0,
    'B-EMPH': 1,
    'I-EMPH': 2
}

def calc_distirubtion(train_dataset: Dataset) -> List[float]:
    counts = [0] * len(label_map)

    for entry in train_dataset:
        tags = entry["tags"]
        for tag in tags:
            counts[label_map[tag]] += 1

    return [counts[i] / sum(counts) for i in range(len(counts))]

def plot_the_logs(logs):
    print(logs)

    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    steps = [log["step"] for log in logs if "step" in log]

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
    tokenized_inputs = tokenizer(examples["raw"], truncation=True, is_split_into_words=False, max_length=256, padding='max_length')
    labels = []

    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs after tokenization
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(label_map["O"])
            elif word_idx != previous_word_idx:
                if word_idx < len(label):
                    label_ids.append(label_map[label[word_idx]])
                else:
                    label_ids.append(label_map["O"])
            else:
                label_ids.append(label_ids[-1])

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_model(train_dataset: Dataset, val_dataset: Dataset):

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
        eval_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model("./results")

    plot_the_logs(trainer.state.log_history)

    return trainer.state.log_history

if __name__ == '__main__':

    train_dataset = load_dataset('data/train_data.json')
    val_dataset = load_dataset('data/eval_data.json')
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)
    model = WeightedLossModel(base_model, [1/share for share in calc_distirubtion(train_dataset)])

    model.dropout = nn.Dropout(p=0.25)
    logs = train_model(train_dataset, val_dataset)

