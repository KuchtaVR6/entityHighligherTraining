from src.bio_to_annotated_text.utils import load_and_align, compute_label_metrics, compute_overall_accuracy

def main():
    word_tokens = load_and_align('logits/inference_logits.jsonl')
    metrics = compute_label_metrics(word_tokens)
    print("===")
    for label, (correct, total, accuracy) in sorted(metrics.items()):
        print(f"Label {label}: {accuracy:.2%} ({correct}/{total})")
    (correct, total, overall_accuracy) = compute_overall_accuracy(word_tokens)
    print(f"Overall: {overall_accuracy:.2%} ({correct}/{total})")
    print("===")

if __name__ == "__main__":
    main()
