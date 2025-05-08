from src.helpers.load_model_and_tokenizer import load_model_and_tokenizer
from src.helpers.logging_utils import setup_logger
from src.train.train_loop import run_training
from src.helpers.load_helpers import tokenize_and_align_labels_batch
from src.helpers.load_helpers import load_large_dataset
from src.helpers.label_map import label_map
from src.helpers.dataset_utils import calc_distribution
from src.configs.path_config import train_data_path, eval_data_path

logger = setup_logger()

if __name__ == '__main__':
    logger.info("Loading raw datasets...")
    raw_train = load_large_dataset(train_data_path)
    raw_val = load_large_dataset(eval_data_path)

    _, tokenizer = load_model_and_tokenizer([0,0,0])

    logger.info("Tokenizing and aligning labels...")

    train_dataset = raw_train.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True, remove_columns=["text"]
    )

    val_dataset = raw_val.map(
        lambda x: tokenize_and_align_labels_batch(x, tokenizer, label_map),
        batched=True, remove_columns=["text"]
    )

    logger.info("Computing class weights...")
    class_weights = [1 / share for share in calc_distribution(train_dataset)]

    model, _ = load_model_and_tokenizer(class_weights)

    logger.info("Starting training...")
    run_training(train_dataset, val_dataset, tokenizer, model)
