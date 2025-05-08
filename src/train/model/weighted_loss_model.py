import torch
import torch.nn as nn

class WeightedLossModel(nn.Module):
    def __init__(self, base_model, class_weights, dropout_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits
