from src.train.utils.logging_utils import setup_logger
from src.train.model.load_model import load_tokenizer_and_model
from src.train.train_loop import run_training
from src.load_helpers import tokenize_and_align_labels_batch
from src.load_helpers import load_large_dataset
from src.label_map import label_map
from src.train.utils.dataset_utils import calc_distribution

logger = setup_logger()

if __name__ == '__main__':
    logger.info("Loading raw datasets...")
    raw_train = load_large_dataset('data/toy_train.json')
    raw_val = load_large_dataset('data/toy_eval.json')

    logger.info("Preparing tokenizer and model...")
    tokenizer, base_model = load_tokenizer_and_model("bert-base-uncased", num_labels=3)

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

    logger.info("Starting training...")
    run_training(train_dataset, val_dataset, tokenizer, base_model, class_weights)
