import json
import logging
from pathlib import Path
from tqdm import tqdm

from src.models.inference import (
    build_prompt, generate_with_uncertainty, extract_sequence_features, 
    load_model_and_tokenizer, unload_model
)
from src.labels.correctness import CorrectnessLabeler

logger = logging.getLogger(__name__)


def evaluate_model_on_fold(
    model_id: str,
    dataset_name: str,
    fold_id: int,
    dataset: list,
    split_indices: list,
    output_dir: Path = Path('results/data'),
    device: str = 'cpu',
    max_examples: int = None,  # For testing: limit to N examples
):
    """
    Evaluate one model on one fold of one dataset.
    
    Args:
        model_id: e.g., 'qwen2.5-0.5b'
        dataset_name: e.g., 'triviaqa'
        fold_id: 0-4
        dataset: list of example dicts (full dataset)
        split_indices: list of indices for this fold's test set
        output_dir: where to save JSONL
        device: 'cpu' or 'cuda'
        max_examples: if set, only evaluate first N examples (for testing)
    
    Returns:
        path to output JSONL
    """
    
    logger.info("="*70)
    logger.info(f"Evaluating {model_id} on {dataset_name} fold {fold_id}")
    logger.info(f"  Test set size: {len(split_indices)}")
    logger.info("="*70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    labeler = CorrectnessLabeler()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_id}_{dataset_name}_fold{fold_id}.jsonl"
    
    results = []
    test_indices = split_indices[:max_examples] if max_examples else split_indices
    
    logger.info(f"Processing {len(test_indices)} examples...")
    
    for idx in tqdm(test_indices, desc=f"{model_id} {dataset_name} fold {fold_id}"):
        example = dataset[idx]
        question = example['question']
        
        try:
            # Build prompt
            prompt = build_prompt(question, dataset_name, model_id)
            
            # Generate + extract uncertainty
            gen_result = generate_with_uncertainty(
                model, tokenizer, prompt, max_new_tokens=50, device=device
            )
            
            # Extract features
            seq_features = extract_sequence_features(
                gen_result['token_logprobs'],
                gen_result['token_entropies']
            )
            
            # Label correctness
            labels = labeler.label_example(
                {
                    'generated': gen_result['text'],
                    'best_answer': example.get('best_answer', ''),
                    'correct_answers': example.get('correct_answers', []),
                    'incorrect_answers': example.get('incorrect_answers', []),
                },
                dataset_name=dataset_name
            )
            
            # Save result
            result_dict = {
                'idx': idx,
                'fold_id': fold_id,
                'question': question,
                'best_answer': example.get('best_answer', ''),
                'generated_answer': gen_result['text'],
                'features': seq_features,
                'labels': labels,
                'tokens': {
                    'logprobs': gen_result['token_logprobs'],
                    'entropies': gen_result['token_entropies'],
                }
            }
            
            results.append(result_dict)
        
        except Exception as e:
            logger.warning(f"Failed on example {idx}: {e}")
            continue
    
    # Save JSONL
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    logger.info(f"✓ Saved {len(results)} results to {output_path}")
    
    # Compute + log stats
    correct = sum(1 for r in results if r['labels']['semantic_correct'] == 1)
    accuracy = correct / len(results) if results else 0
    avg_entropy = sum(r['features']['avg_entropy'] for r in results) / len(results) if results else 0
    
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Avg entropy: {avg_entropy:.3f}")
    
    # Clean up
    unload_model(model)
    
    return str(output_path)