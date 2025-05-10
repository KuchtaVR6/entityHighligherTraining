from pathlib import Path

save_model_path = Path("results/custom_model.pth")
train_data_path = Path("data/split_files/part_1.json")
eval_data_path = Path("data/toy_train.json")
logits_data_path = Path("logits/inference_logits.jsonl")
logits_output_path = Path("logits/output.jsonl")
checkpoints_path = Path("checkpoints")
logs_path = Path("logs")
