from collections import Counter
from datasets import Dataset
from typing import List

def calc_distribution(train_dataset: Dataset, consider_mask: bool = False) -> List[float]:
    tag_counter = Counter()

    if consider_mask and "loss_mask" in train_dataset.features:
        # Only count tags where the loss mask is 1 (within proximity window)
        for example in train_dataset:
            labels = example["labels"]
            loss_mask = example["loss_mask"]

            # Only count tokens within the proximity window
            for label, mask in zip(labels, loss_mask):
                if mask == 1:
                    tag_counter[label] += 1
    else:
        # Count all tags regardless of mask
        for example in train_dataset:
            tag_counter.update(example["labels"])

    # Ensure we have counts for all classes (0, 1, 2)
    for i in range(3):
        if i not in tag_counter:
            tag_counter[i] = 0

    total = sum(tag_counter.values())
    # Prevent division by zero
    if total == 0:
        return [1/3, 1/3, 1/3]  # Equal weights if no data

    return [tag_counter[tag] / total for tag in sorted(tag_counter)]
