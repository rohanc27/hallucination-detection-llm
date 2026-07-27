import pytest

from src.data.loaders import load_triviaqa, load_squad_v2, load_custom_benchmark

pytestmark = pytest.mark.integration  # hits the live HF Hub; skip with `-m "not integration"` offline


def test_load_triviaqa_small_sample():
    ds = load_triviaqa(n_samples=5)
    assert len(ds) == 5
    assert "question" in ds[0]


def test_load_squad_v2_small_sample():
    ds = load_squad_v2(n_samples=5)
    assert len(ds) == 5
    assert "answers" in ds[0]


def test_load_custom_benchmark():
    data = load_custom_benchmark()
    assert len(data) == 15
    assert all("correct_answers" in q for q in data)
