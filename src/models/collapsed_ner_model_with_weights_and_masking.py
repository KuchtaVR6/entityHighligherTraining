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

        # Register as buffer to ensure proper device handling
        self.register_buffer(
            "class_weights",
            (
                class_weights
                if isinstance(class_weights, torch.Tensor)
                else torch.tensor(class_weights, dtype=torch.float)
            ),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def collapse_logits(self, logits: torch.Tensor) -> torch.Tensor:
        device = logits.device
        collapsed_logits = torch.zeros(
            logits.size(0), logits.size(1), 3, device=device, dtype=logits.dtype
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
        # Ensure all inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Move any tensor kwargs to the correct device
        kwargs = {
            k: v.to(device) if hasattr(v, "to") and callable(v.to) else v
            for k, v in kwargs.items()
        }

        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        collapsed_logits = self.collapse_logits(outputs.logits)

        if labels is not None:
            labels = labels.to(device)
            loss = self.loss_fn(collapsed_logits.view(-1, 3), labels.view(-1))

            if loss_mask is not None:
                loss = (loss * loss_mask.view(-1).to(device)).mean()
            else:
                loss = loss.mean()

            return loss, collapsed_logits

        return collapsed_logits
