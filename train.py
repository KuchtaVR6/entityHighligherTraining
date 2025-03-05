import torch
import torch.nn as nn
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List
import re

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


def load_large_dataset(file_path: str) -> Dataset:
    """Loads dataset efficiently using streaming for large files."""
    return load_dataset('json', data_files=file_path, split='train', streaming=False)  # TODO ENABLE STREAMING


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


def tokenize_and_align_labels_batch(examples, tokenizer, label_map):
    """Processes raw text, extracts BIO tags, tokenizes, and aligns labels in batch."""
    raw_texts = [remove_span_tags(entry) for entry in examples["text"]]
    bio_tags = [transform_to_bio(entry) for entry in examples["text"]]

    tokenized_inputs = tokenizer(
        raw_texts,
        truncation=True,
        padding="max_length",
        max_length=256,
        is_split_into_words=False,
        return_tensors="pt"
    )

    labels = []
    for i, label in enumerate(bio_tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [
            label_map[label[word_idx]] if word_idx is not None and word_idx < len(label) else label_map["O"]
            for word_idx in word_ids
        ]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def calc_distribution(train_dataset: Dataset) -> List[float]:
    """TODO - make this count only once because its slow"""
    return [1/3, 1/3, 1/3]


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
        save_total_limit=2,
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
    train_dataset = load_large_dataset('data/toy_train.json') # todo replace it with data
    val_dataset = load_large_dataset('data/toy_eval.json')

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True
    )

    logger.info("Calculating class weights...")
    class_weights = [1 / share for share in calc_distribution(train_dataset)]
    model = WeightedLossModel(base_model, class_weights)
    model = torch.compile(model)  # Compile model for better performance

    # Dataloader with multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    logs = train_model(train_dataset, val_dataset)
