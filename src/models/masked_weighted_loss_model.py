import torch
import torch.nn as nn


class MaskedWeightedLossModel(nn.Module):
    def __init__(self, base_model, class_weights, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float), reduction="none"
        )

    def calculate_weighted_loss(self, logits, labels):
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def forward(self, input_ids, attention_mask, labels, loss_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)

        if loss_mask is not None:
            masked_loss = (
                loss_mask.view(-1) * self.calculate_weighted_loss(logits, labels)
            ).mean()
        else:
            masked_loss = self.calculate_weighted_loss(logits, labels).mean()

        return masked_loss, logits
