from transformers import TrainingArguments

from src.configs.path_config import checkpoints_path, logs_path


def get_training_args():
    return TrainingArguments(
        output_dir=checkpoints_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        logging_steps=50,
        weight_decay=0.3,
        logging_dir=logs_path,
        evaluation_strategy="epoch",
        save_steps=10000,
        save_total_limit=2,
        remove_unused_columns=False
    )
