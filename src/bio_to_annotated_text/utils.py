import json
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast

from .models import TokenRepresentation, WordTokens
from src.helpers.label_map import label_map

LABEL_MAP: Dict[int, str] = {v: k[0] if '-' in k else k for k, v in label_map.items()}

def predict_label(word_token: WordTokens, prev_label: int = 0, summative_prediction: bool = False) -> int:
    if not word_token.tokens:
        return 0

    prod = np.ones_like(word_token.tokens[0].logits)
    for idx, tr in enumerate(word_token.tokens):
        logits = np.array(tr.logits)

        if summative_prediction and prev_label == 0 and idx == 0:
            logits[1] += logits[2]

        if idx != 0:
            logits[1] = logits[2]  # Enforce transition logic: inner tokens treated differently

        prod *= logits

    pred = int(np.argmax(prod))

    if pred == 2 and prev_label == 0:
        pred = 0

    return pred

def load_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer, set(tokenizer.all_special_tokens)

def process_line(data: dict, tokenizer, special_tokens) -> List[WordTokens]:
    raw_tokens, raw_logits, raw_labels = data['tokens'], data['logits'], data['labels']
    text = data['text']

    filtered = [(t, lg, lb) for t, lg, lb in zip(raw_tokens, raw_logits, raw_labels) if t not in special_tokens]
    if not filtered:
        return []

    tokens, logits, labels = zip(*filtered)
    words = text.split()
    pos = 0
    record_tokens: List[WordTokens] = []

    for word in words:
        pieces = tokenizer.tokenize(word)
        count = len(pieces)
        if pos + count > len(tokens):
            break

        slice_tokens = tokens[pos:pos + count]
        slice_logits = logits[pos:pos + count]
        slice_labels = labels[pos:pos + count]

        token_reprs = [TokenRepresentation(tok, lgt, lbl) for tok, lgt, lbl in zip(slice_tokens, slice_logits, slice_labels)]
        record_tokens.append(WordTokens(word, token_reprs))
        pos += count

    return record_tokens

def update_metrics(wt: WordTokens, local_stats: Dict[int, Tuple[int, int]], global_stats: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    if not wt.tokens:
        return 0, 0

    pred = predict_label(wt)
    true = wt.tokens[0].label
    correct = int(pred == true)

    # Local update
    c, t = local_stats.get(true, (0, 0))
    local_stats[true] = (c + correct, t + 1)

    # Global update
    c_g, t_g = global_stats.get(true, (0, 0))
    global_stats[true] = (c_g + correct, t_g + 1)

    return correct, 1

def compute_metrics(stats: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int, float]]:
    return {lbl: (c, t, c / t if t > 0 else 0.0) for lbl, (c, t) in stats.items()}

def annotate_word_tokens(word_tokens: List[WordTokens]) -> str:
    annotated_parts: List[str] = []
    in_span = False
    prev_label = 0

    for wt in word_tokens:
        if not wt.tokens:
            continue

        pred = predict_label(wt, prev_label)
        prev_label = pred
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

def write_output(f_out, record_tokens: List[WordTokens], local_stats: Dict[int, Tuple[int, int]], correct: int, total: int):
    annotated = annotate_word_tokens(record_tokens)
    label_metrics = compute_metrics(local_stats)
    label_metrics['overall'] = (correct, total, correct / total if total > 0 else 0.0)

    out_obj = {
        'annotated_text': annotated,
        'metrics': label_metrics
    }
    f_out.write(json.dumps(out_obj) + '\n')

def print_global_metrics(global_stats: Dict[int, Tuple[int, int]], overall_correct: int, overall_total: int):
    print("===")
    for label, (c, t) in sorted(global_stats.items()):
        print(f"Label {label}: {c / t:.2%} ({c}/{t})")
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print(f"Overall: {overall_acc:.2%} ({overall_correct}/{overall_total})")
    print("===")

def process_records(input_path: str, output_path: str) -> None:
    tokenizer, special_tokens = load_tokenizer()
    global_stats: Dict[int, Tuple[int, int]] = {}
    overall_correct = 0
    overall_total = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in):
            data = json.loads(line)
            record_tokens = process_line(data, tokenizer, special_tokens)
            if not record_tokens:
                continue

            local_stats: Dict[int, Tuple[int, int]] = {}
            correct = total = 0

            for wt in record_tokens:
                c, t = update_metrics(wt, local_stats, global_stats)
                correct += c
                total += t

            overall_correct += correct
            overall_total += total
            write_output(f_out, record_tokens, local_stats, correct, total)

    print_global_metrics(global_stats, overall_correct, overall_total)
