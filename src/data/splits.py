import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("hallucination_detection")


def generate_kfold_splits(dataset, n_splits=5, random_state=42):
    """Generate n_splits disjoint train/test folds covering every index exactly once in test."""
    n = len(dataset)
    logger.info(f"Generating {n_splits}-fold CV splits for {n} examples")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    splits = {}
    fold_size = n // n_splits
    for fold_id in range(n_splits):
        test_start = fold_id * fold_size
        test_end = test_start + fold_size if fold_id < n_splits - 1 else n

        test_idx = indices[test_start:test_end].tolist()
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]]).tolist()

        splits[f"fold_{fold_id}"] = {
            "train": sorted(train_idx),
            "test": sorted(test_idx),
        }
        logger.info(f"  Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")

    return splits


def save_splits(splits, output_path="results/data/splits.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {output_path}")


def load_splits(path="results/data/splits.json"):
    with open(path) as f:
        splits = json.load(f)
    logger.info(f"Loaded splits from {path}")
    return splits


def verify_no_leakage(splits, dataset_size):
    """Raise AssertionError if any fold's train/test overlap, or if test sets don't
    partition the full dataset exactly once."""
    all_test = []
    for fold_id, fold_splits in splits.items():
        train_set = set(fold_splits["train"])
        test_set = set(fold_splits["test"])

        overlap = train_set & test_set
        if overlap:
            raise AssertionError(f"Leakage in {fold_id}: {len(overlap)} indices in both train and test")

        all_test.extend(fold_splits["test"])

    all_test_set = set(all_test)
    if len(all_test) != len(all_test_set):
        raise AssertionError("Leakage across folds: some indices appear in test set of more than one fold")

    expected = set(range(dataset_size))
    if all_test_set != expected:
        missing = expected - all_test_set
        extra = all_test_set - expected
        raise AssertionError(
            f"Test folds do not partition dataset: missing={len(missing)}, unexpected={len(extra)}"
        )

    logger.info("✓ No data leakage detected in k-fold splits")


def create_all_splits(datasets_dict, n_splits=5, random_state=42, output_path="results/data/splits_all_datasets.json"):
    """Create, verify, and save k-fold splits for every dataset in datasets_dict."""
    all_splits = {}
    for ds_name, ds in datasets_dict.items():
        logger.info(f"Creating splits for {ds_name}")
        splits = generate_kfold_splits(ds, n_splits=n_splits, random_state=random_state)
        verify_no_leakage(splits, len(ds))
        all_splits[ds_name] = splits

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_splits, f, indent=2)
    logger.info(f"Saved all splits to {output_path}")

    return all_splits
