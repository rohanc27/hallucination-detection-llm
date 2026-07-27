import logging

from src.config import Config
from src.data.loaders import (
    load_triviaqa,
    load_truthfulqa,
    load_squad_v2,
    load_hotpotqa,
    load_mmlu_sampled,
    load_custom_benchmark,
)
from src.data.splits import create_all_splits

logger = logging.getLogger("hallucination_detection")


class HallucinationDetectionPipeline:
    def __init__(self, config: Config):
        self.config = config

    def run_phase_1(self):
        """Phase 1: Data & Methodology — load datasets, build leakage-free k-fold splits."""
        logger.info("=" * 70)
        logger.info("PHASE 1: Data & Methodology")
        logger.info("=" * 70)

        datasets = {}
        n = self.config.n_samples_per_dataset

        if "triviaqa" in self.config.datasets:
            datasets["triviaqa"] = load_triviaqa(n_samples=n, random_state=self.config.random_state)
        if "truthfulqa" in self.config.datasets:
            ds_gen, ds_mc = load_truthfulqa(n_samples=n, random_state=self.config.random_state)
            datasets["truthfulqa_gen"] = ds_gen
            datasets["truthfulqa_mc"] = ds_mc
        if "squad_v2" in self.config.datasets:
            datasets["squad_v2"] = load_squad_v2(n_samples=n, random_state=self.config.random_state)
        if "hotpotqa" in self.config.datasets:
            datasets["hotpotqa"] = load_hotpotqa(n_samples=n, random_state=self.config.random_state)
        if "mmlu" in self.config.datasets:
            datasets["mmlu"] = load_mmlu_sampled(random_state=self.config.random_state)
        if "custom" in self.config.datasets:
            custom = load_custom_benchmark()
            if custom:
                datasets["custom"] = custom

        logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

        splits = create_all_splits(
            datasets,
            n_splits=self.config.n_splits,
            random_state=self.config.random_state,
            output_path=f"{self.config.data_dir}/splits_all_datasets.json",
        )

        logger.info("✓ Phase 1 complete")
        return datasets, splits


if __name__ == "__main__":
    from src.logging_utils import setup_logging

    logger, log_path, run_id = setup_logging()

    config = Config()
    config.log_config(logger)

    pipeline = HallucinationDetectionPipeline(config)
    datasets, splits = pipeline.run_phase_1()
