
from dataclasses import dataclass
from transformers import BertTokenizerFast
import json
from typing import List

@dataclass
class TokenRepresentation:
    token_str: str
    logits: List[float]
    label: int

@dataclass
class WordTokens:
    word_str: str
    tokens: List[TokenRepresentation]

# Initialize tokenizer (must match the model used during inference)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Path to the JSONL file
jsonl_path = 'logits/inference_logits.jsonl'

all_word_tokens: List[WordTokens] = []

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        raw_tokens = data['tokens']
        raw_logits = data['logits']
        raw_labels = data['labels']
        text = data['text']

        # Filter out special tokens before alignment
        special_tokens = set(tokenizer.all_special_tokens)
        filtered = [
            (t, lg, lb)
            for t, lg, lb in zip(raw_tokens, raw_logits, raw_labels)
            if t not in special_tokens
        ]
        tokens, logits, labels = zip(*filtered)

        # Split the original text into words
        words = text.split()

        pos = 0
        for word in words:
            # Tokenize the word to sub-tokens
            pieces = tokenizer.tokenize(word)
            count = len(pieces)

            # Extract corresponding slices
            slice_tokens = tokens[pos:pos + count]
            slice_logits = logits[pos:pos + count]
            slice_labels = labels[pos:pos + count]

            # Create TokenRepresentations
            token_reprs = [
                TokenRepresentation(token_str=tok, logits=lgt, label=lbl)
                for tok, lgt, lbl in zip(slice_tokens, slice_logits, slice_labels)
            ]

            # Record the word and its sub-tokens
            all_word_tokens.append(WordTokens(word_str=word, tokens=token_reprs))

            pos += count

# Example output for inspection
for wt in all_word_tokens[:100]:
    print(f"Word: {wt.word_str}")
    for tr in wt.tokens:
        print(f"  Token: {tr.token_str}, Label: {tr.label}, Logits (head): {tr.logits[:3]}")
    print()