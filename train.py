from collections import defaultdict

import torch
import torch.nn as nn
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
)
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
        padding=True,
        max_length=256,
        is_split_into_words=False
    )

    labels = []
    for i, label in enumerate(bio_tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [
            label_map[label[word_idx]] if word_idx is not None and word_idx < len(label) else label_map["O"]
            for word_idx in word_ids
        ]
        labels.append(label_ids)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels  # Ensure labels are properly aligned
    }

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

def train_model(train_dataset: Dataset, val_dataset: Dataset, data_collator):
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
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False  # Ensures dataset columns are not ignored
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator  # Include the data collator
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model("./results")
    logger.info("Training complete. Model saved.")

    plot_loss_logs(trainer.state.log_history)
    return trainer.state.log_history


def inference(model, val_dataset, tokenizer, label_map):
    """Makes predictions on the validation dataset and prints the labels."""
    model.eval()  # Set the model in evaluation mode
    all_predictions = []

    # Loop through the validation dataset
    for batch in val_dataset:
        # Get the input IDs and attention masks from the batch
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        with torch.no_grad():
            # Run the model to get logits (do not pass labels during inference)
            outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Logits are the second output

            # Get the predicted labels by taking the argmax of the logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()  # Remove batch dimension

            # Map predictions back to label names
            predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]
            all_predictions.append(predicted_labels)

        break

    # Print the predicted labels
    for predicted_labels in all_predictions:
        print(predicted_labels)


def compute_accuracy(model, val_dataset, tokenizer, label_map):
    model.eval()  # Set the model in evaluation mode
    all_predictions = []
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    id_to_label = {v: k for k, v in label_map.items()}

    for batch in tqdm(val_dataset):
        # Get the input IDs and attention masks from the batch
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        # Get ground truth labels
        labels = batch["labels"]

        with torch.no_grad():
            # Run the model to get logits (do not pass labels during inference)
            outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Logits are the second output

            # Get the predicted labels by taking the argmax of the logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()  # Remove batch dimension

            # Map predictions back to label names
            predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]
            all_predictions.append(predicted_labels)

            # Calculate accuracy for each label
            for i, label_id in enumerate(labels):
                if i >= len(predictions):  # Handle padding or truncated sequences
                    break
                pred_label_id = predictions[i]

                # Count correct predictions
                if label_id == pred_label_id:
                    correct_counts[label_id] += 1
                total_counts[label_id] += 1

        break

    # Calculate per-class accuracy
    per_class_accuracy = {
        id_to_label[class_id]: correct_counts[class_id] / total_counts[class_id]
        for class_id in total_counts if total_counts[class_id] > 0
    }

    # Calculate overall accuracy
    total_correct = sum(correct_counts.values())
    total_samples = sum(total_counts.values())
    overall_accuracy = total_correct / total_samples

    per_class_accuracy["overall_accuracy"] = overall_accuracy

    return per_class_accuracy

if __name__ == '__main__':
    logger.info("Loading datasets...")
    train_dataset = load_large_dataset('data/toy_train.json')  # TODO: Replace with real data
    val_dataset = load_large_dataset('data/toy_eval.json')

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]  # Remove original text to prevent shape mismatch
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    logger.info("Calculating class weights...")
    class_weights = [1 / share for share in calc_distribution(train_dataset)]
    model = WeightedLossModel(base_model, class_weights)

    # Define the data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Use DataLoader with multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    # Train the model with data collator
    logs = train_model(train_dataset, val_dataset, data_collator)
    torch.mps.empty_cache()

    print(compute_accuracy(model, val_dataset, tokenizer, label_map))
