from src.bio_to_annotated_text.utils import load_and_align, compute_label_metrics, compute_overall_accuracy, process_records

def main():
    records = load_and_align('logits/inference_logits.jsonl')
    global_tokens = [wt for rec in records for wt in rec]
    metrics = compute_label_metrics(global_tokens)
    print("===")
    for label, (correct, total, accuracy) in sorted(metrics.items()):
        print(f"Label {label}: {accuracy:.2%} ({correct}/{total})")
    correct, total, overall_accuracy = compute_overall_accuracy(global_tokens)
    print(f"Overall: {overall_accuracy:.2%} ({correct}/{total})")
    print("===")
    process_records('logits/inference_logits.jsonl', 'output.jsonl')

if __name__ == "__main__":
    main()
