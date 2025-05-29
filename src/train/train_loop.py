import gc

import torch
from transformers import DataCollatorForTokenClassification, Trainer

from src.configs.path_config import save_model_path
from src.configs.training_args import get_training_args


def run_training(train_dataset, val_dataset, tokenizer, model):
    gc.collect()

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    tokenizer.save_pretrained(save_model_path.parent)
