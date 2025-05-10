import logging

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.configs.path_config import eval_data_path, save_model_path
from src.helpers.load_helpers import load_large_dataset, tokenize_text
from src.helpers.load_model_and_tokenizer import load_model_and_tokenizer
from src.models.weighted_loss_model import MaskedWeightedLossModel
from src.helpers.label_map import label_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detokenize(tokens):
    """Convert tokenized output back to readable text."""
    words = []
    for token in tokens:
        if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
            continue  # Ignore special tokens
        if token.startswith("##"):
            words[-1] += token[2:]  # Merge subword tokens
        else:
            words.append(token)
    return " ".join(words)

def infer(model, infer_dataset, label_map):
    model.eval()

    for batch in tqdm(infer_dataset):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

        labels = batch["labels"]
        tensor_labels = torch.tensor(labels).unsqueeze(0)
        zero_tensor = torch.zeros_like(tensor_labels)

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # Use the original text from the dataset if available
        original_text = batch["text"] if "text" in batch else None

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor)
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

            # Decode the input IDs into tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

            # Get predicted labels
            predicted_labels = [key for idx in predictions for key, val in label_map.items() if val == idx]

            # Prepare XML output for highlighting spans
            highlighted_text = []

            if original_text:
                # Use the original text as a base for displaying
                start_idx = 0
                for token, label in zip(tokens, predicted_labels):
                    if label != 'O':  # Only highlight non-'O' labels
                        # Find the corresponding word in the original text
                        word = original_text[start_idx:start_idx + len(token)].replace('##', '')
                        # Wrap the span with an XML tag
                        highlighted_text.append(f"<span class='{label}'>{word}</span>")
                        start_idx += len(word)
                    else:
                        # If 'O', just append the token without change
                        word = original_text[start_idx:start_idx + len(token)].replace('##', '')
                        highlighted_text.append(word)
                        start_idx += len(word)
            else:
                # If no original text, fall back to just displaying tokens with their BIO labels
                annotated_output = []
                for token, label in zip(tokens, predicted_labels):
                    if label != 'O':
                        annotated_output.append(f"<span class='{label}'>{token}</span>")
                    else:
                        annotated_output.append(token)

                highlighted_text = annotated_output

            # Join all parts into a single string for output
            print(" ".join(highlighted_text))

if __name__ == '__main__':
    logger.info("Loading datasets...")
    val_dataset = load_large_dataset(eval_data_path)

    model, tokenizer = load_model_and_tokenizer([0,0,0])

    val_dataset = val_dataset.map(
        lambda x: tokenize_text(x, tokenizer, label_map),
        batched=True,
        remove_columns=["text"]
    )

    print(infer(model, val_dataset, label_map))
