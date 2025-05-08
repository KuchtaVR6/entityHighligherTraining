import os
import gc
import torch
from pathlib import Path
from transformers import Trainer, DataCollatorForTokenClassification
from src.train.config.training_args import get_training_args
from src.train.model.weighted_loss_model import WeightedLossModel

def run_training(train_dataset, val_dataset, tokenizer, base_model, class_weights):
    model = WeightedLossModel(base_model, class_weights)

    save_model_path = Path("results/custom_model.pth")

    if os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path))

    del base_model
    gc.collect()

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    tokenizer.save_pretrained(save_model_path.parent)
