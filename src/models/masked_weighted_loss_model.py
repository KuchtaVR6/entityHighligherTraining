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
        weights_tensor = (
            class_weights
            if isinstance(class_weights, torch.Tensor)
            else torch.tensor(class_weights, dtype=torch.float)
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, reduction="none")

    def calculate_weighted_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))  # type: ignore[no-any-return]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        loss_mask: Tensor | None = None,
        **kwargs: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with masking and weighted loss calculation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            loss_mask: Optional mask for loss calculation
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (loss, logits) tensors
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)

        if loss_mask is not None:
            masked_loss = (
                loss_mask.view(-1) * self.calculate_weighted_loss(logits, labels)
            ).mean()
        else:
            masked_loss = self.calculate_weighted_loss(logits, labels).mean()

        return masked_loss, logits
