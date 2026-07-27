from src.labels.correctness import CorrectnessLabeler


def test_substring_label_correct_match():
    labeler = CorrectnessLabeler.__new__(CorrectnessLabeler)  # skip loading the embedder
    result = labeler.label_substring(
        "The capital of France is Paris.", correct_answers=["Paris"], incorrect_answers=["London"]
    )
    assert result == 1


def test_substring_label_incorrect_match():
    labeler = CorrectnessLabeler.__new__(CorrectnessLabeler)
    result = labeler.label_substring(
        "I believe the answer is London.", correct_answers=["Paris"], incorrect_answers=["London"]
    )
    assert result == 0


def test_substring_label_empty_generation_is_none():
    labeler = CorrectnessLabeler.__new__(CorrectnessLabeler)
    result = labeler.label_substring("", correct_answers=["Paris"], incorrect_answers=["London"])
    assert result is None
