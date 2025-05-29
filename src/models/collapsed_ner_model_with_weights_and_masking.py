from typing import Any

import torch
from torch import Tensor
import torch.nn as nn


class CollapsedNERModel(nn.Module):
    def __init__(
        self, base_model: nn.Module, class_weights: list[float] | torch.Tensor
    ) -> None:
        super().__init__()
        self.base_model = base_model

        self.o_id = 0
        self.b_ids = [1, 3, 5, 7]
        self.i_ids = [2, 4, 6, 8]

        weights_tensor = (
            class_weights
            if isinstance(class_weights, torch.Tensor)
            else torch.tensor(class_weights, dtype=torch.float)
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, reduction="none")

    def collapse_logits(self, logits: torch.Tensor) -> torch.Tensor:
        collapsed_logits = torch.zeros(logits.size(0), logits.size(1), 3).to(
            logits.device
        )
        collapsed_logits[..., 0] = logits[..., self.o_id]  # O
        collapsed_logits[..., 1] = logits[..., self.b_ids].mean(dim=-1)  # B
        collapsed_logits[..., 2] = logits[..., self.i_ids].mean(dim=-1)  # I
        return collapsed_logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        **kwargs: Any
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Forward pass with collapsed NER logits and optional loss calculation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            loss_mask: Optional mask for loss calculation
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (loss, logits) tensors. If labels are not provided, loss will be zero.
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        collapsed_logits = self.collapse_logits(outputs.logits)

        if labels is not None:
            loss = self.loss_fn(collapsed_logits.view(-1, 3), labels.view(-1))

            if loss_mask is not None:
                loss = (loss * loss_mask.view(-1)).mean()
            else:
                loss = loss.mean()

            return loss, collapsed_logits

        return collapsed_logits
