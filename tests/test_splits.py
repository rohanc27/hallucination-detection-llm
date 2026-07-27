import pytest

from src.data.splits import generate_kfold_splits, verify_no_leakage


class DummyDataset(list):
    """Minimal stand-in for an HF Dataset — just needs __len__."""


def test_splits_partition_dataset_exactly_once():
    ds = DummyDataset(range(97))
    splits = generate_kfold_splits(ds, n_splits=5, random_state=42)

    assert len(splits) == 5
    verify_no_leakage(splits, len(ds))


def test_splits_are_deterministic_given_seed():
    ds = DummyDataset(range(50))
    splits_a = generate_kfold_splits(ds, n_splits=5, random_state=42)
    splits_b = generate_kfold_splits(ds, n_splits=5, random_state=42)
    assert splits_a == splits_b


def test_verify_no_leakage_catches_train_test_overlap():
    ds = DummyDataset(range(20))
    splits = generate_kfold_splits(ds, n_splits=5, random_state=42)
    splits["fold_0"]["train"].append(splits["fold_0"]["test"][0])

    with pytest.raises(AssertionError):
        verify_no_leakage(splits, len(ds))


def test_verify_no_leakage_catches_index_in_two_test_folds():
    ds = DummyDataset(range(20))
    splits = generate_kfold_splits(ds, n_splits=5, random_state=42)
    splits["fold_1"]["test"].append(splits["fold_0"]["test"][0])

    with pytest.raises(AssertionError):
        verify_no_leakage(splits, len(ds))
