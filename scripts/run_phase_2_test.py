# scripts/run_phase_2_test.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import setup_logging
from src.config import Config
from src.data.loaders import load_dataset
from src.data.splits import load_splits
from src.models.evaluation import evaluate_model_on_fold

def main():
    logger, log_path = setup_logging()
    config = Config()
    
    logger.info("="*70)
    logger.info("PHASE 2 TEST: Single model, single dataset, single fold")
    logger.info("="*70)
    
    # Load first dataset (TriviaQA)
    logger.info("Loading TriviaQA...")
    dataset = load_dataset('triviaqa', n_samples=100)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Load splits
    logger.info("Loading splits...")
    splits = load_splits('results/data/splits_all_datasets.json')
    fold_0_test = splits['triviaqa']['fold_0']['test']
    logger.info(f"Fold 0 test set: {len(fold_0_test)} indices")
    
    # Evaluate first model (TinyLlama, smallest)
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    logger.info(f"Evaluating {model_id}...")
    
    output_path = evaluate_model_on_fold(
        model_id=model_id,
        dataset_name='triviaqa',
        fold_id=0,
        dataset=dataset,
        split_indices=fold_0_test,
        output_dir=Path('results/data'),
        device='cpu',
        max_examples=10,  # TEST: only 10 examples
    )
    
    logger.info("="*70)
    logger.info("PHASE 2 TEST COMPLETE")
    logger.info(f"Results: {output_path}")
    logger.info("="*70)

if __name__ == '__main__':
    main()