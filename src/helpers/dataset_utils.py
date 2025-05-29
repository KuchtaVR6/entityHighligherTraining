import json
import os
import random
import time
from collections import Counter

from datasets import Dataset

from src.helpers.logging_utils import setup_logger

logger = setup_logger()

# Cache path for saving/loading distributions (for faster startup)
CACHE_DIR = os.path.expanduser("~/.cache/entity_highlighter")
os.makedirs(CACHE_DIR, exist_ok=True)


def calc_distribution(
    train_dataset: Dataset,
    consider_mask: bool = False,
    max_samples: int = 10000,
    use_cache: bool = True,
) -> list[float]:
    """
    Calculate class distribution for token classification with efficient sampling.

    Args:
        train_dataset: The dataset to analyze
        consider_mask: Whether to only count tokens in loss mask windows
        max_samples: Maximum number of examples to sample for large datasets
        use_cache: Whether to use cached results if available

    Returns:
        List of class distribution values
    """
    start_time = time.time()
    dataset_size = len(train_dataset)

    # Generate cache key
    cache_key = f"dist_{dataset_size}_{consider_mask}_{max_samples}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Try to load from cache if enabled
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached_dist = json.load(f)
                return cached_dist
        except Exception:
            pass

    # Determine if we need to sample
    need_sampling = dataset_size > max_samples
    sample_size = min(dataset_size, max_samples)

    if need_sampling:
        # Sample random indices
        indices = random.sample(range(dataset_size), sample_size)
        # Get sampled examples
        sampled_dataset = train_dataset.select(indices)
    else:
        sampled_dataset = train_dataset

    # Initialize counter
    tag_counter = Counter()

    # Count class occurrences (optimized for proximity)
    if consider_mask and "loss_mask" in sampled_dataset.features:
        # Process in batches to speed up
        batch_size = 100
        num_examples = len(sampled_dataset)

        for i in range(0, num_examples, batch_size):
            batch = sampled_dataset.select(range(i, min(i + batch_size, num_examples)))
            for example in batch:
                labels = example["labels"]
                loss_mask = example["loss_mask"]

                # Only count tokens within the proximity window
                for label, mask in zip(labels, loss_mask, strict=False):
                    if mask == 1:
                        tag_counter[label] += 1
    else:
        # Process in batches to speed up
        batch_size = 100
        num_examples = len(sampled_dataset)

        for i in range(0, num_examples, batch_size):
            batch = sampled_dataset.select(range(i, min(i + batch_size, num_examples)))
            for example in batch:
                tag_counter.update(example["labels"])

    # Ensure we have counts for all classes (0, 1, 2)
    for i in range(3):
        if i not in tag_counter:
            tag_counter[i] = 0

    total = sum(tag_counter.values())
    # Prevent division by zero
    if total == 0:
        logger.warning("No tokens counted! Using equal distribution.")
        distribution = [1 / 3, 1 / 3, 1 / 3]  # Equal weights if no data
    else:
        distribution = [tag_counter[tag] / total for tag in sorted(tag_counter)]

    # Cache result for future use
    if use_cache:
        try:
            with open(cache_path, "w") as f:
                json.dump(distribution, f)
        except Exception:
            pass

    return distribution
