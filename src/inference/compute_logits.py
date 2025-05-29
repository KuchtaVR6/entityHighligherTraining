from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


def compute_logits(
    model: nn.Module,
    infer_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    for i, batch in enumerate(tqdm(infer_dataset)):
        if max_examples is not None and i >= max_examples:
            break

        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor | None = batch.get("labels", None)  # Optional labels

        zero_tensor: torch.Tensor = torch.zeros_like(input_ids)
        tokens: list[list[str]] = [
            tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids
        ]

        with torch.no_grad():
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor
            )
            logits: torch.Tensor = (
                output[1].cpu().detach()
                if isinstance(output, tuple)
                else output.logits.cpu().detach()
            )

        for idx, (token_seq, logit_seq) in enumerate(zip(tokens, logits, strict=False)):
            entry: dict[str, Any] = {"tokens": token_seq, "logits": logit_seq.tolist()}
            if labels is not None:
                entry["labels"] = labels[idx].cpu().tolist()
            results.append(entry)

    return results
