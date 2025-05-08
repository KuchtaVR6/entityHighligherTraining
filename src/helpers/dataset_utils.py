from collections import Counter
from datasets import Dataset
from typing import List

def calc_distribution(train_dataset: Dataset) -> List[float]:
    tag_counter = Counter()
    for example in train_dataset:
        tag_counter.update(example["labels"])
    total = sum(tag_counter.values())
    return [tag_counter[tag] / total for tag in sorted(tag_counter)]
