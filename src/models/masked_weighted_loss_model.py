from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from transformers import PreTrainedModel


class MaskedWeightedLossModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        class_weights: list[float] | torch.Tensor,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)

        self.register_buffer(
            "class_weights",
            (
                class_weights
                if isinstance(class_weights, torch.Tensor)
                else torch.tensor(class_weights, dtype=torch.float)
            ),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def calculate_weighted_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        loss: Tensor = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        weights: Tensor = self.class_weights[labels.view(-1)]
        result: Tensor = loss * weights
        return result

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        loss_mask: Tensor | None = None,
        **kwargs: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: (v.to(device) if hasattr(v, "to") and callable(v.to) else v)
                for k, v in kwargs.items()
            }
        )

        logits = self.dropout(outputs.logits)
        loss = self.calculate_weighted_loss(logits, labels)

        if loss_mask is not None:
            loss = (loss_mask.view(-1) * loss).mean()
        else:
            loss = loss.mean()

        return loss, logits
