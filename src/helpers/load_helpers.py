import re
from datasets import load_dataset, Dataset

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

def tokenize_text(texts, tokenizer):
    """Tokenizes raw texts (no labels, no XML tags) for inference."""
    cleaned_texts = texts

    # Tokenize without any labels
    tokenized_inputs = tokenizer(
        cleaned_texts,
        truncation=True,
        padding=True,
        max_length=256,
        is_split_into_words=False
    )

    return tokenized_inputs


def load_large_dataset(file_path) -> Dataset:
    """Loads dataset efficiently using streaming for large files."""
    return load_dataset('json', data_files=str(file_path), split='train', streaming=False)
