import torch
from tqdm import tqdm


def compute_logits(model, infer_dataset, tokenizer, max_examples=None):
    model.eval()
    results = []

    for i, batch in enumerate(tqdm(infer_dataset)):
        if max_examples is not None and i >= max_examples:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)  # Optional labels

        zero_tensor = torch.zeros_like(input_ids)
        tokens = [tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids]

        with torch.no_grad():
            _, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor
            )
            logits = logits.cpu().detach()

        for idx, (token_seq, logit_seq) in enumerate(zip(tokens, logits, strict=False)):
            entry = {"tokens": token_seq, "logits": logit_seq.tolist()}
            if labels is not None:
                entry["labels"] = labels[idx].cpu().tolist()
            results.append(entry)

    return results
