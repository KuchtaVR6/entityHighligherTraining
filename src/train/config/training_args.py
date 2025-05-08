from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir='../../checkpoints',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        logging_steps=50,
        weight_decay=0.3,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_steps=10000,
        save_total_limit=2,
        remove_unused_columns=False
    )
