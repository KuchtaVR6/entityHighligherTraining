import json
import numpy as np
from transformers import BertTokenizerFast
from typing import List, Dict, Tuple
from src.bio_to_annotated_text.models import TokenRepresentation, WordTokens

def load_and_align(jsonl_path: str) -> List[WordTokens]:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    all_word_tokens: List[WordTokens] = []
    special_tokens = set(tokenizer.all_special_tokens)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_tokens = data['tokens']
            raw_logits = data['logits']
            raw_labels = data['labels']
            text = data['text']
            filtered = [(t, lg, lb) for t, lg, lb in zip(raw_tokens, raw_logits, raw_labels) if t not in special_tokens]
            tokens, logits, labels = zip(*filtered)
            words = text.split()
            pos = 0
            for word in words:
                pieces = tokenizer.tokenize(word)
                count = len(pieces)
                slice_logits = logits[pos:pos+count]
                slice_labels = labels[pos:pos+count]
                slice_tokens = tokens[pos:pos+count]
                token_reprs = [TokenRepresentation(tok, lgt, lbl) for tok, lgt, lbl in zip(slice_tokens, slice_logits, slice_labels)]
                all_word_tokens.append(WordTokens(word, token_reprs))
                pos += count
    return all_word_tokens

def compute_label_metrics(word_tokens: List[WordTokens]) -> Dict[int, Tuple[int, int, float]]:
    stats: Dict[int, Tuple[int, int]] = {}
    for wt in word_tokens:
        if not wt.tokens:
            continue
        prod = np.ones_like(wt.tokens[0].logits)
        for tr in wt.tokens:
            prod *= np.array(tr.logits)
        pred = int(np.argmax(prod))
        true = wt.tokens[0].label
        correct, total = stats.get(true, (0, 0))
        total += 1
        if pred == true:
            correct += 1
        stats[true] = (correct, total)
    metrics: Dict[int, Tuple[int, int, float]] = {}
    for label, (correct, total) in stats.items():
        accuracy = correct / total if total > 0 else 0.0
        metrics[label] = (correct, total, accuracy)
    return metrics

def compute_overall_accuracy(word_tokens: List[WordTokens]) -> Tuple[int, int, float]:
    correct = 0
    total = 0
    for wt in word_tokens:
        if not wt.tokens:
            continue
        prod = np.ones_like(wt.tokens[0].logits)
        for tr in wt.tokens:
            prod *= np.array(tr.logits)
        pred = int(np.argmax(prod))
        true = wt.tokens[0].label
        if pred == true:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    return correct, total, accuracy
