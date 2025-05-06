from src.bio_to_annotated_text.utils import process_records

def main():
    process_records('logits/inference_logits.jsonl', 'logits/output.jsonl')

if __name__ == "__main__":
    main()
