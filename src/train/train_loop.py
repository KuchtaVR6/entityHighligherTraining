import gc

from datasets import Dataset
import torch
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
)

from src.configs.path_config import save_model_path
from src.configs.training_args import get_training_args


def run_training(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
) -> tuple[PreTrainedModel, Trainer]:
    """Run the training loop with the given datasets, tokenizer, and model.

    Args:
        train_dataset: The training dataset
        val_dataset: The validation dataset
        tokenizer: The tokenizer to use for data collation
        model: The model to train
    """
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

    return model, trainer
