import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("hallucination_detection")


def _subsample(ds, n_samples, random_state=42):
    if not n_samples or n_samples >= len(ds):
        return ds
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(ds), size=n_samples, replace=False)
    ds = ds.select(sorted(indices.tolist()))
    logger.info(f"Sampled to {len(ds)} examples")
    return ds


def load_triviaqa(split="validation", n_samples=None, random_state=42):
    """Load TriviaQA (rc.nocontext config: open-ended, no reference passage)."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    logger.info(f"Loaded TriviaQA: {len(ds)} examples")
    logger.info(f"  Example question: {ds[0]['question'][:50]}...")

    ds = _subsample(ds, n_samples, random_state)
    return ds


def load_truthfulqa(split="validation", n_samples=None, random_state=42):
    """Load TruthfulQA in both generation and multiple_choice (MC1) formats, index-aligned."""
    from datasets import load_dataset

    ds_gen = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    ds_mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    logger.info(f"Loaded TruthfulQA: {len(ds_gen)} generation, {len(ds_mc)} MC1")

    if n_samples and n_samples < len(ds_gen):
        rng = np.random.default_rng(random_state)
        indices = sorted(rng.choice(len(ds_gen), size=n_samples, replace=False).tolist())
        ds_gen = ds_gen.select(indices)
        ds_mc = ds_mc.select(indices)
        logger.info(f"Sampled to {len(ds_gen)} examples")

    return ds_gen, ds_mc


def load_squad_v2(split="validation", n_samples=None, random_state=42, include_unanswerable=True):
    """Load SQuAD v2 (extractive QA, includes unanswerable questions)."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    logger.info(f"Loaded SQuAD v2: {len(ds)} examples")

    if not include_unanswerable:
        ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
        logger.info(f"Filtered to answerable: {len(ds)} examples")

    ds = _subsample(ds, n_samples, random_state)
    return ds


def load_hotpotqa(split="validation", n_samples=None, random_state=42):
    """Load HotpotQA (distractor config: multi-hop reasoning over 10 paragraphs)."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    logger.info(f"Loaded HotpotQA: {len(ds)} examples")

    ds = _subsample(ds, n_samples, random_state)
    return ds


def load_mmlu_sampled(n_samples_per_subject=20, random_state=42, subjects=None):
    """Load a handful of MMLU subjects, sampled evenly, concatenated into one dataset."""
    from datasets import load_dataset, concatenate_datasets

    subjects = subjects or ["anatomy", "astronomy", "world_religions", "marketing", "college_biology"]
    parts = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        ds = _subsample(ds, n_samples_per_subject, random_state)
        parts.append(ds)
        logger.info(f"  MMLU/{subject}: {len(ds)} examples")

    ds_all = concatenate_datasets(parts)
    logger.info(f"Loaded MMLU (sampled): {len(ds_all)} examples across {len(subjects)} subjects")
    return ds_all


def load_custom_benchmark(filepath="data/custom_benchmark.json"):
    """Load a hand-verified custom benchmark from JSON: list of {question, correct_answers, ...}."""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Custom benchmark not found at {filepath}")
        return []

    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded custom benchmark: {len(data)} examples")
    return data
