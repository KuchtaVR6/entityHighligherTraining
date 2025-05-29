import torch
import torch.nn as nn


class CollapsedNERModel(nn.Module):
    def __init__(self, base_model, class_weights):
        super().__init__()
        self.base_model = base_model

        self.o_id = 0
        self.b_ids = [1, 3, 5, 7]
        self.i_ids = [2, 4, 6, 8]

        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float), reduction="none"
        )

    def collapse_logits(self, logits):
        collapsed_logits = torch.zeros(logits.size(0), logits.size(1), 3).to(
            logits.device
        )
        collapsed_logits[..., 0] = logits[..., self.o_id]  # O
        collapsed_logits[..., 1] = logits[..., self.b_ids].mean(dim=-1)  # B
        collapsed_logits[..., 2] = logits[..., self.i_ids].mean(dim=-1)  # I
        return collapsed_logits

    def forward(
        self, input_ids, attention_mask=None, labels=None, loss_mask=None, **kwargs
    ):
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
