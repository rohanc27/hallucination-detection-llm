from dataclasses import dataclass, field, asdict
from typing import List, Dict
import yaml


@dataclass
class Config:
    # Datasets to evaluate
    datasets: List[str] = field(
        default_factory=lambda: ["triviaqa", "truthfulqa", "squad_v2", "hotpotqa", "custom"]
    )

    # Models to test — start small since this environment is CPU-only.
    # Expand via config.yaml or --models once a GPU (Colab/remote) is available.
    models: List[str] = field(default_factory=lambda: ["qwen2.5-0.5b", "gpt2", "pythia-160m"])

    # K-fold settings
    n_splits: int = 5
    random_state: int = 42

    # Inference
    max_new_tokens: int = 50
    device: str = "cpu"
    dtype: str = "auto"

    # Sample sizes (kept small by default for CPU feasibility)
    n_samples_per_dataset: int = 200

    # Features to compute
    features_enabled: Dict[str, bool] = field(
        default_factory=lambda: {
            "avg_entropy": True,
            "avg_logprob": True,
            "min_logprob": True,
            "seq_length": True,
            "self_confidence": True,
            "mc1_top_prob": True,
            "token_variance": False,  # Phase 3
            "attention_entropy": False,  # Phase 3
        }
    )

    # Calibration methods
    calibration_methods: List[str] = field(default_factory=lambda: ["logistic_regression"])

    # Paths
    data_dir: str = "results/data"
    log_dir: str = "results/logs"
    figures_dir: str = "results/figures"
    summaries_dir: str = "results/summaries"
    reports_dir: str = "results/reports"

    @classmethod
    def load_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, sort_keys=False)

    def log_config(self, logger):
        logger.info("=" * 70)
        logger.info("Configuration")
        logger.info("=" * 70)
        for key, val in asdict(self).items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 70)
