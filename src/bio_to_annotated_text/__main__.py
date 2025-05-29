from src.bio_to_annotated_text.utils import process_records
from src.configs.path_config import logits_data_path, logits_output_path


def main():
    process_records(logits_data_path, logits_output_path)


if __name__ == "__main__":
    main()
