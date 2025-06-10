from pathlib import Path
import re

from datasets import Dataset, load_dataset
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


def remove_span_tags(text: str) -> str:
    return re.sub(r"<\/?.*?span.*?>", "", text)


def transform_to_bio(annotated: str) -> list[str]:
    tags, current_tag = [], None
    text_with_tags = re.split(r"(<[^>]+>)", annotated)

    for token in text_with_tags:
        if re.match(r"<[^/]+>", token):
            current_tag = token.strip("<>")
        elif re.match(r"</[^>]+>", token):
            current_tag = None
        elif token.strip():
            words = token.split()
            for i, _word in enumerate(words):
                tag_prefix = "B" if i == 0 else "I"
                tags.append(f"{tag_prefix}-EMPH" if current_tag else "O")
    return tags


def tokenize_and_align_labels_batch(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    label_map: dict[str, int],
    proximity: int | None = None,
) -> dict[str, torch.Tensor | list[list[int]] | None]:
    """Processes raw text, extracts BIO tags, tokenizes, aligns labels, and computes loss mask."""
    raw_texts = [remove_span_tags(entry) for entry in examples["text"]]
    bio_tags = [transform_to_bio(entry) for entry in examples["text"]]

    tokenized_inputs = tokenizer(
        raw_texts,
        truncation=True,
        padding=True,
        max_length=256,
        is_split_into_words=False,
        return_tensors="pt",  # to simplify mask ops
    )

    labels = []
    loss_masks: list[list[int]] = []

    for i, label in enumerate(bio_tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        seq_labels = []

        important_positions = []

        for pos, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(label):
                tag = label[word_idx]
                tag_id = label_map[tag]
                seq_labels.append(tag_id)

                if tag_id != 0:  # or 1/2 if you're using that
                    important_positions.append(pos)
            else:
                seq_labels.append(0)

        if proximity:
            # Compute loss mask
            mask = [0] * len(seq_labels)
            for pos in important_positions:
                for i in range(
                    max(0, pos - proximity), min(len(mask), pos + proximity + 1)
                ):
                    mask[i] = 1

            loss_masks.append(mask)
        elif proximity is not None:
            # If proximity is not None but no mask was added, add a default mask of 1s
            loss_masks.append([1] * len(seq_labels))

        labels.append(seq_labels)

    result = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }

    if proximity is not None:
        result["loss_mask"] = loss_masks

    return result


def tokenize_text(
    texts: list[str], tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    """Tokenizes raw texts (no labels, no XML tags) for inference."""
    cleaned_texts = texts

    # Tokenize without any labels
    tokenized_inputs = tokenizer(
        cleaned_texts,
        truncation=True,
        padding=True,
        max_length=256,
        is_split_into_words=False,
    )

    return tokenized_inputs


def load_large_dataset(file_path: str | Path) -> Dataset:
    return load_dataset(
        "json", data_files=str(file_path), split="train", streaming=False
    )
