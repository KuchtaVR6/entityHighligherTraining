import json
import numpy as np
from transformers import BertTokenizerFast
from typing import List, Dict, Tuple
from .models import TokenRepresentation, WordTokens

LABEL_MAP: Dict[int, str] = {1: 'B', 2: 'I', 0: 'O'}

def predict_label(word_token: WordTokens, prev_label: int = 0, summative_prediction: bool = False) -> int:
    if not word_token.tokens:
        return 0

    prod = np.ones_like(word_token.tokens[0].logits)

    for idx, tr in enumerate(word_token.tokens):
        logits = np.array(tr.logits)

        if summative_prediction:
            if prev_label == 0:
                if idx == 0:
                    logits[1] += logits[2]

        if idx != 0:
            logits[1] = logits[2]  # B on word level will always be BI* on token level

        prod *= logits

    pred = int(np.argmax(prod))

    if pred == 2 and prev_label == 0:
        pred = 0

    return pred

def load_and_align(jsonl_path: str) -> List[List[WordTokens]]:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    all_records: List[List[WordTokens]] = []
    special_tokens = set(tokenizer.all_special_tokens)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_tokens, raw_logits, raw_labels = data['tokens'], data['logits'], data['labels']
            text = data['text']
            filtered = [(t, lg, lb) for t, lg, lb in zip(raw_tokens, raw_logits, raw_labels) if t not in special_tokens]
            tokens, logits, labels = zip(*filtered)
            words = text.split()
            pos = 0
            record_tokens: List[WordTokens] = []
            for word in words:
                pieces = tokenizer.tokenize(word)
                count = len(pieces)
                slice_tokens = tokens[pos:pos+count]
                slice_logits = logits[pos:pos+count]
                slice_labels = labels[pos:pos+count]
                token_reprs = [TokenRepresentation(tok, lgt, lbl) for tok, lgt, lbl in zip(slice_tokens, slice_logits, slice_labels)]
                record_tokens.append(WordTokens(word, token_reprs))
                pos += count
            all_records.append(record_tokens)
    return all_records

def compute_label_metrics(word_tokens: List[WordTokens]) -> Dict[int, Tuple[int, int, float]]:
    stats: Dict[int, Tuple[int, int]] = {}
    for wt in word_tokens:
        if not wt.tokens:
            continue
        pred = predict_label(wt)
        true = wt.tokens[0].label
        correct, total = stats.get(true, (0, 0))
        total += 1
        if pred == true:
            correct += 1
        stats[true] = (correct, total)
    return {lbl: (c, t, c/t if t > 0 else 0.0) for lbl, (c, t) in stats.items()}

def compute_overall_accuracy(word_tokens: List[WordTokens]) -> Tuple[int, int, float]:
    correct = total = 0
    for wt in word_tokens:
        if not wt.tokens:
            continue
        pred = predict_label(wt)
        true = wt.tokens[0].label
        if pred == true:
            correct += 1
        total += 1
    return correct, total, correct/total if total > 0 else 0.0

def annotate_word_tokens(word_tokens: List[WordTokens]) -> str:
    annotated_parts: List[str] = []
    in_span = False
    prev_label = 0  # Start from 'O'

    for wt in word_tokens:
        if not wt.tokens:
            continue
        pred = predict_label(wt, prev_label)
        prev_label = pred  # Update for next word
        tag = LABEL_MAP.get(pred, 'O')
        word = wt.word_str

        if tag == 'B':
            if in_span:
                annotated_parts.append('</span>')
            annotated_parts.append(f'<span>{word}')
            in_span = True
        elif tag == 'I' and in_span:
            annotated_parts.append(word)
        else:
            if in_span:
                annotated_parts.append('</span>')
                in_span = False
            annotated_parts.append(word)
    if in_span:
        annotated_parts.append('</span>')
    return ' '.join(annotated_parts).replace(" </span>", "</span>")

def process_records(input_path: str, output_path: str) -> None:
    records = load_and_align(input_path)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for record_tokens in records:
            flat_tokens = [wt for wt in record_tokens]
            label_metrics = compute_label_metrics(flat_tokens)
            correct, total, overall = compute_overall_accuracy(flat_tokens)
            label_metrics['overall'] = (correct, total, overall)
            annotated = annotate_word_tokens(record_tokens)
            out_obj = {
                'annotated_text': annotated,
                'metrics': label_metrics
            }
            out_f.write(json.dumps(out_obj) + '\n')
