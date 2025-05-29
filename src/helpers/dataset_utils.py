from collections import Counter
import json
import os
import random

from datasets import Dataset

from src.helpers.logging_utils import setup_logger

logger = setup_logger()

CACHE_DIR = os.path.expanduser("~/.cache/entity_highlighter")
os.makedirs(CACHE_DIR, exist_ok=True)


def calc_distribution(
    train_dataset: Dataset,
    consider_mask: bool = False,
    max_samples: int = 10000,
    use_cache: bool = True,
) -> list[float]:
    dataset_size = len(train_dataset)

    cache_key = f"dist_{dataset_size}_{consider_mask}_{max_samples}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached_dist = json.load(f)
                return [float(x) for x in cached_dist]
        except Exception:
            pass

    need_sampling = dataset_size > max_samples
    sample_size = min(dataset_size, max_samples)

    if need_sampling:
        indices = random.sample(range(dataset_size), sample_size)
        sampled_dataset = train_dataset.select(indices)
    else:
        sampled_dataset = train_dataset

    tag_counter: dict[int, int] = Counter()

    if consider_mask and "loss_mask" in sampled_dataset.features:
        batch_size = 100
        num_examples = len(sampled_dataset)

        for i in range(0, num_examples, batch_size):
            batch = sampled_dataset.select(range(i, min(i + batch_size, num_examples)))
            for example in batch:
                labels = example["labels"]
                loss_mask = example["loss_mask"]

                for label, mask in zip(labels, loss_mask, strict=False):
                    if mask == 1:
                        tag_counter[label] += 1
    else:
        batch_size = 100
        num_examples = len(sampled_dataset)

        for i in range(0, num_examples, batch_size):
            batch = sampled_dataset.select(range(i, min(i + batch_size, num_examples)))
            for example in batch:
                tag_counter.update(example["labels"])

    for i in range(3):
        if i not in tag_counter:
            tag_counter[i] = 0

    total = sum(tag_counter.values())
    if total == 0:
        logger.warning("No tokens counted! Using equal distribution.")
        distribution: list[float] = [
            1.0 / 3,
            1.0 / 3,
            1.0 / 3,
        ]
    else:
        distribution = [tag_counter[tag] / total for tag in sorted(tag_counter)]

    if use_cache:
        try:
            with open(cache_path, "w") as f:
                json.dump(distribution, f)
        except Exception:
            pass

    return distribution
