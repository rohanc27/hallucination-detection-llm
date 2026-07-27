# Claude Code Prompt — Hallucination Detection LLM Full Implementation

**Goal**: Implement a comprehensive, publication-ready hallucination detection system with:
- Proper data pipeline (k-fold CV, no leakage)
- 10+ models across 3+ families
- 5+ independent datasets
- Multiple calibration methods
- Extensive logging & reproducibility
- 100+ git commits with pushes

**Start at Phase 0 and work through all 7 phases sequentially.**

---

## Instructions for Claude Code

### Git Workflow (CRITICAL)

**After EVERY logical step, commit and push:**

```bash
git add -A
git commit -m "Brief descriptive message"
git push origin main
```

**Commit messages format**: `✓ PHASE.SECTION.STEP: Brief description`

Examples:
- `✓ 0.1.1: Initialize git repository structure`
- `✓ 1.1.2: Load TruthfulQA in both generation and MC1 formats`
- `✓ 2.2.5: Inference loop with uncertainty extraction per model/fold`
- `✓ 4.4.1: Generate comprehensive results table with cross-model comparison`

**Commit frequency**: Aim for 100+ commits over the entire project. Small, focused commits are better than large ones.

### Logging (CRITICAL)

**Every step should log extensively:**

1. **File logging**: `results/logs/{timestamp}.log` with all output
2. **Console logging**: Timestamps, run IDs, progress bars
3. **Structured logging**: Include phase, section, step in every log entry
4. **Log key metrics**: After each step, log results, counts, statistics

Example:
```python
import logging
logger = logging.getLogger(__name__)

# At start of run:
logger.info("="*70)
logger.info("PHASE 1: Data & Methodology")
logger.info(f"Run ID: {run_id}, Seed: {SEED}")
logger.info("="*70)

# During data loading:
logger.info(f"Loaded TriviaQA: {len(ds)} examples")
logger.info(f"Example question: {ds[0]['question'][:50]}...")

# After splits:
logger.info(f"Generated k-fold splits (n_splits=5)")
logger.info(f"Fold 0: train={len(splits['fold_0']['train'])}, test={len(splits['fold_0']['test'])}")

# Verify no leakage:
logger.info("✓ No data leakage detected in k-fold splits")
```

---

## Phase 0: Project Setup & Infrastructure (Target: 5 commits)

### 0.1 Git & Directory Structure

**Task**: Create directory structure, git init, .gitignore

```bash
# Create src/ subdirectories
mkdir -p src/{data,models,features,labels,calibration,evaluation}
mkdir -p scripts results/{logs,data,figures,summaries,reports}
mkdir -p notebooks tests docs

# Create __init__.py in all src subdirs
touch src/__init__.py src/data/__init__.py src/models/__init__.py \
      src/features/__init__.py src/labels/__init__.py \
      src/calibration/__init__.py src/evaluation/__init__.py

# Initialize git (if not already)
git init
```

**Commit 0.1.1**: "Initialize git repository structure"

### 0.2 Setup files

**Create `setup.py`** with dependencies:
- torch, transformers, datasets, sentence-transformers, sklearn, numpy, pandas, matplotlib, seaborn, tqdm, pyyaml

**Create `pyproject.toml`** (minimal, for development)

**Create `.gitignore`**:
```
*.pyc
__pycache__/
.DS_Store
results/data/  # large model weights & intermediate results
*.pkl
*.pt
*.pth
.env
```

**Commit 0.1.2**: "Add setup.py, pyproject.toml, .gitignore"

### 0.3 Logging Infrastructure

**Create `src/logging.py`**:
```python
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir='results/logs', log_level=logging.INFO):
    """Initialize logging to file + console with timestamps."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'run_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('hallucination_detection')
    logger.setLevel(log_level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s:%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f'Logging to {log_file}')
    return logger, log_file

if __name__ == '__main__':
    logger, path = setup_logging()
    logger.info('Test log message')
    print(f'Log saved to: {path}')
```

**Commit 0.1.3**: "Set up logging infrastructure with file + console output"

### 0.4 Config System

**Create `src/config.py`** with dataclass:
```python
from dataclasses import dataclass, field
from typing import List, Dict
import yaml
from pathlib import Path

@dataclass
class Config:
    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: ['triviaqa', 'truthfulqa', 'squad_v2', 'hotpotqa', 'custom'])
    
    # Models to test (will expand to full list)
    models: List[str] = field(default_factory=lambda: ['qwen2.5-0.5b'])
    
    # K-fold settings
    n_splits: int = 5
    random_state: int = 42
    
    # Inference
    max_new_tokens: int = 50
    device: str = 'cuda'
    dtype: str = 'auto'
    
    # Features to compute
    features_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'avg_entropy': True,
        'avg_logprob': True,
        'min_logprob': True,
        'seq_length': True,
        'self_confidence': True,
        'mc1_top_prob': True,
        'token_variance': False,  # Phase 3
        'attention_entropy': False,  # Phase 3
    })
    
    # Calibration methods
    calibration_methods: List[str] = field(default_factory=lambda: ['logistic_regression'])
    
    # Paths
    data_dir: str = 'results/data'
    log_dir: str = 'results/logs'
    figures_dir: str = 'results/figures'
    reports_dir: str = 'results/reports'
    
    # Load from YAML if provided
    @classmethod
    def load_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
    
    def log_config(self, logger):
        logger.info("="*70)
        logger.info("Configuration")
        logger.info("="*70)
        for key, val in self.__dict__.items():
            logger.info(f"  {key}: {val}")
        logger.info("="*70)
```

**Commit 0.1.4**: "Create centralized config system with YAML support"

### 0.5 Constants

**Create `src/constants.py`**:
```python
MODEL_REGISTRY = {
    # Will be filled in Phase 2, but structure it now
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
}

DATASET_REGISTRY = {
    # Will be filled in Phase 1
    'triviaqa': 'load_triviaqa',
    'truthfulqa': 'load_truthfulqa',
    'squad_v2': 'load_squad_v2',
    'hotpotqa': 'load_hotpotqa',
    'mmlu': 'load_mmlu_sampled',
    'custom': 'load_custom_benchmark',
}

LABEL_METHODS = ['substring', 'semantic', 'mc1']
FEATURE_NAMES = ['avg_entropy', 'avg_logprob', 'min_logprob', 'seq_length', 'self_confidence', 'mc1_top_prob']
CALIBRATION_METHODS = ['logistic_regression', 'temperature_scaling', 'isotonic', 'platt']
```

**Commit 0.1.5**: "Initialize constants and model/dataset registries"

**After Phase 0, you should have:**
- ✅ Clean directory structure
- ✅ Git initialized with meaningful .gitignore
- ✅ Logging system ready (file + console)
- ✅ Config system (YAML loadable)
- ✅ Constants/registries
- ✅ 5 commits

---

## Phase 1: Data & Methodology (Target: 20 commits)

### 1.1 Data Loaders

**Create `src/data/loaders.py`** with functions to load each dataset:

```python
def load_triviaqa(split='validation', n_samples=None, random_state=42):
    """Load TriviaQA from HF datasets."""
    from datasets import load_dataset
    ds = load_dataset('trivia_qa', 'rc.nocontext', split=split)
    logger.info(f"Loaded TriviaQA: {len(ds)} examples")
    
    if n_samples:
        import numpy as np
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
        ds = ds.select(sorted(indices))
        logger.info(f"Sampled to {len(ds)} examples")
    
    return ds

def load_truthfulqa(split='validation', n_samples=None):
    """Load TruthfulQA generation + MC1."""
    from datasets import load_dataset
    ds_gen = load_dataset('truthful_qa', 'generation', split=split)
    ds_mc = load_dataset('truthful_qa', 'multiple_choice', split=split)
    logger.info(f"Loaded TruthfulQA: {len(ds_gen)} generation, {len(ds_mc)} MC1")
    
    if n_samples:
        import numpy as np
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds_gen), size=min(n_samples, len(ds_gen)), replace=False)
        ds_gen = ds_gen.select(sorted(indices))
        ds_mc = ds_mc.select(sorted(indices))
        logger.info(f"Sampled to {len(ds_gen)} examples")
    
    return ds_gen, ds_mc

def load_squad_v2(split='validation', n_samples=None, include_unanswerable=True):
    """Load SQuAD v2 (extractive QA)."""
    from datasets import load_dataset
    ds = load_dataset('squad_v2', split=split)
    logger.info(f"Loaded SQuAD v2: {len(ds)} examples")
    
    if not include_unanswerable:
        ds = ds.filter(lambda x: not x['is_impossible'])
        logger.info(f"Filtered to answerable: {len(ds)} examples")
    
    if n_samples:
        import numpy as np
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
        ds = ds.select(sorted(indices))
    
    return ds

def load_hotpotqa(split='validation', n_samples=None):
    """Load HotpotQA (multi-hop)."""
    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'distractor', split=split)
    logger.info(f"Loaded HotpotQA: {len(ds)} examples")
    
    if n_samples:
        import numpy as np
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
        ds = ds.select(sorted(indices))
    
    return ds

def load_mmlu_sampled(n_samples_per_subject=20, random_state=42):
    """Load 5 subjects from MMLU."""
    # Implementation depends on MMLU availability
    # For now, return placeholder
    logger.info("MMLU loading deferred (requires special handling)")
    return None

def load_custom_benchmark(filepath='data/custom_benchmark.json'):
    """Load custom benchmark from JSON."""
    import json
    try:
        with open(filepath) as f:
            data = json.load(f)
        logger.info(f"Loaded custom benchmark: {len(data)} examples")
        return data
    except FileNotFoundError:
        logger.warning(f"Custom benchmark not found at {filepath}")
        return []
```

**Commit 1.1.1**: "Load TriviaQA from HF datasets with validation"

**Commit 1.1.2**: "Load TruthfulQA in both generation and MC1 formats"

**Commit 1.1.3**: "Load SQuAD v2 with answerable/unanswerable split"

**Commit 1.1.4**: "Load HotpotQA multi-hop reasoning dataset"

**Commit 1.1.5**: "Load MMLU sampled subjects (placeholder)"

**Commit 1.1.6**: "Load custom benchmark from JSON"

### 1.2 K-Fold Splits

**Create `src/data/splits.py`**:

```python
import numpy as np
import json
from pathlib import Path

def generate_kfold_splits(dataset, n_splits=5, random_state=42):
    """Stratified k-fold split (by correctness if available, else random)."""
    from sklearn.model_selection import StratifiedKFold
    
    logger.info(f"Generating {n_splits}-fold CV splits for {len(dataset)} examples")
    
    # For stratification, we'll use a pseudo-label if available
    # For now, use random split (will update after correctness labels are available)
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    
    splits = {}
    fold_size = len(indices) // n_splits
    for fold_id in range(n_splits):
        test_start = fold_id * fold_size
        test_end = test_start + fold_size if fold_id < n_splits - 1 else len(indices)
        
        test_idx = indices[test_start:test_end].tolist()
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]]).tolist()
        
        splits[f'fold_{fold_id}'] = {
            'train': sorted(train_idx),
            'test': sorted(test_idx),
        }
        
        logger.info(f"  Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")
    
    return splits

def save_splits(splits, output_path='results/data/splits.json'):
    """Save splits to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {output_path}")

def load_splits(path='results/data/splits.json'):
    """Load splits from JSON."""
    with open(path) as f:
        splits = json.load(f)
    logger.info(f"Loaded splits from {path}")
    return splits

def verify_no_leakage(splits, dataset_size):
    """Verify all indices are covered exactly once."""
    all_train = set()
    all_test = set()
    
    for fold_id, fold_splits in splits.items():
        train_set = set(fold_splits['train'])
        test_set = set(fold_splits['test'])
        
        # Check no overlap within fold
        if train_set & test_set:
            raise AssertionError(f"Leakage in {fold_id}: overlap between train and test")
        
        all_train.update(train_set)
        all_test.update(test_set)
    
    # Check all indices covered
    all_indices = all_train | all_test
    expected = set(range(dataset_size))
    if all_indices != expected:
        raise AssertionError(f"Not all indices covered: {len(all_indices)} vs {len(expected)}")
    
    logger.info("✓ No data leakage detected")

# Top-level function to create all splits
def create_all_splits(datasets_dict, n_splits=5, random_state=42):
    """Create k-fold splits for all datasets."""
    all_splits = {}
    for ds_name, ds in datasets_dict.items():
        logger.info(f"Creating splits for {ds_name}")
        splits = generate_kfold_splits(ds, n_splits=n_splits, random_state=random_state)
        verify_no_leakage(splits, len(ds))
        all_splits[ds_name] = splits
    
    # Save all
    output_path = 'results/data/splits_all_datasets.json'
    with open(output_path, 'w') as f:
        json.dump(all_splits, f, indent=2)
    logger.info(f"Saved all splits to {output_path}")
    
    return all_splits
```

**Commit 1.2.1**: "Implement k-fold split generation with stratification"

**Commit 1.2.2**: "Save and load splits from JSON with reproducibility"

**Commit 1.2.3**: "Verify no data leakage in k-fold splits"

### 1.3 Labeling

**Create `src/labels/correctness.py`**:

```python
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util

class CorrectnessLabeler:
    def __init__(self, semantic_model='all-MiniLM-L6-v2'):
        """Initialize labeler with semantic embedder."""
        self.embedder = SentenceTransformer(semantic_model)
        logger.info(f"Loaded embedder: {semantic_model}")
    
    def label_substring(self, generated, correct_answers, incorrect_answers):
        """Substring matching."""
        g = generated.lower().strip()
        if not g:
            return None
        
        matches_correct = any(c.lower().strip() in g for c in correct_answers if c.strip())
        matches_incorrect = any(i.lower().strip() in g for i in incorrect_answers if i.strip())
        
        return 1 if (matches_correct and not matches_incorrect) else 0
    
    def label_semantic(self, generated, reference, threshold=0.65):
        """Semantic similarity using embeddings."""
        if not generated.strip():
            return 0, 0.0
        
        emb = self.embedder.encode([generated, reference], convert_to_tensor=True)
        sim = float(st_util.cos_sim(emb[0], emb[1]).item())
        
        return (1 if sim >= threshold else 0, sim)
    
    def label_example(self, question_result, dataset_name='triviaqa'):
        """Compute all labels for a question."""
        generated = question_result.get('generated', '')
        
        if dataset_name == 'triviaqa':
            correct = question_result.get('correct_answers', [])
            incorrect = question_result.get('incorrect_answers', [])
            substring = self.label_substring(generated, correct, incorrect)
            semantic, sim = self.label_semantic(generated, question_result.get('best_answer', ''))
            
            return {
                'substring_correct': substring,
                'semantic_correct': semantic,
                'semantic_similarity': sim,
            }
        # Add more dataset-specific logic as needed
```

**Commit 1.3.1**: "Implement substring and semantic similarity labeling"

**Commit 1.3.2**: "Create unified CorrectnessLabeler class"

### 1.4 Main Pipeline Setup

**Create `src/pipeline.py`** (skeleton for orchestration):

```python
import logging
from src.config import Config
from src.data.loaders import (
    load_triviaqa, load_truthfulqa, load_squad_v2, load_hotpotqa
)
from src.data.splits import create_all_splits

logger = logging.getLogger(__name__)

class HallucinationDetectionPipeline:
    def __init__(self, config):
        self.config = config
        logger.info(f"Initializing pipeline with config: {config}")
    
    def run_phase_1(self):
        """Phase 1: Data & Methodology."""
        logger.info("="*70)
        logger.info("PHASE 1: Data & Methodology")
        logger.info("="*70)
        
        # Load all datasets
        datasets = {}
        if 'triviaqa' in self.config.datasets:
            datasets['triviaqa'] = load_triviaqa(n_samples=1000)
        if 'truthfulqa' in self.config.datasets:
            ds_gen, ds_mc = load_truthfulqa(n_samples=500)
            datasets['truthfulqa_gen'] = ds_gen
            datasets['truthfulqa_mc'] = ds_mc
        if 'squad_v2' in self.config.datasets:
            datasets['squad_v2'] = load_squad_v2(n_samples=500)
        if 'hotpotqa' in self.config.datasets:
            datasets['hotpotqa'] = load_hotpotqa(n_samples=500)
        
        logger.info(f"Loaded {len(datasets)} datasets")
        
        # Create k-fold splits
        splits = create_all_splits(datasets, n_splits=self.config.n_splits)
        
        logger.info("✓ Phase 1 complete")
        return datasets, splits

if __name__ == '__main__':
    from src.logging import setup_logging
    logger, log_path = setup_logging()
    
    config = Config()
    pipeline = HallucinationDetectionPipeline(config)
    datasets, splits = pipeline.run_phase_1()
```

**Commit 1.4.1**: "Create main pipeline orchestration skeleton"

**After Phase 1, you should have:**
- ✅ Data loaders for all 5+ datasets
- ✅ K-fold CV implementation with leakage verification
- ✅ Labeling system (substring + semantic)
- ✅ Pipeline skeleton
- ✅ 20 commits total

---

## Phase 2: Multi-Model Testing (Target: 15 commits)

### 2.1 Model Loading

**Create `src/models/loader.py`**:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLoader:
    def __init__(self):
        self.cache = {}
    
    def load(self, model_id, device='cuda', dtype='auto'):
        """Load model + tokenizer with caching."""
        if model_id in self.cache:
            logger.info(f"[cache] Using cached {model_id}")
            return self.cache[model_id]
        
        logger.info(f"Loading {model_id}...")
        
        # Determine dtype
        if dtype == 'auto':
            dtype_torch = torch.float16 if device == 'cuda' else torch.float32
        else:
            dtype_torch = getattr(torch, dtype)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype_torch,
            device_map=device
        ).eval()
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {model_id}: {n_params/1e6:.1f}M params, dtype={dtype_torch}, device={device}")
        
        self.cache[model_id] = (model, tokenizer)
        return model, tokenizer
    
    def clear_cache(self):
        """Clear GPU memory."""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model cache")
```

**Commit 2.1.1**: "Implement model loader with caching and dtype handling"

### 2.2 Generation & Uncertainty

**Create `src/models/inference.py`**:

```python
import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def generate_with_uncertainty(model, tokenizer, prompt, max_new_tokens=50, device='cuda'):
    """Generate answer + extract token-level uncertainty."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    input_len = inputs['input_ids'].shape[1]
    
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    gen_ids = out.sequences[0, input_len:]
    scores = out.scores
    
    if len(scores) == 0:
        return {
            'text': '',
            'token_logprobs': [],
            'token_entropies': [],
        }
    
    token_logprobs, token_entropies = [], []
    for logits in scores:
        logits = logits[0].float()
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        
        # Get token
        token_id = gen_ids[len(token_logprobs)].item()
        token_logprobs.append(logp[token_id].item())
        token_entropies.append(-(p * logp).sum().item())
        
        if token_id == tokenizer.eos_token_id:
            break
    
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    logger.debug(f"Generated: {text[:50]}... ({len(token_logprobs)} tokens)")
    
    return {
        'text': text,
        'token_logprobs': token_logprobs,
        'token_entropies': token_entropies,
    }
```

**Commit 2.2.1**: "Implement generation with token-level uncertainty extraction"

**Commit 2.2.2**: "Create prompt builders for different datasets and models"

### 2.3 Per-Model Evaluation

**Create `src/pipeline.py` extension**:

```python
def run_phase_2(self):
    """Phase 2: Multi-Model Testing."""
    logger.info("="*70)
    logger.info("PHASE 2: Multi-Model Testing")
    logger.info("="*70)
    
    from src.models.loader import ModelLoader
    from src.models.inference import generate_with_uncertainty
    
    loader = ModelLoader()
    
    for model_id in self.config.models[:1]:  # Start with 1 model for testing
        logger.info(f"\nEvaluating {model_id}")
        model, tokenizer = loader.load(model_id, device=self.config.device)
        
        # Evaluate on one dataset + fold
        for ds_name in self.config.datasets[:1]:
            logger.info(f"  On {ds_name}")
            # TODO: Full evaluation loop
        
        loader.clear_cache()
    
    logger.info("✓ Phase 2 complete")
```

**Commit 2.3.1**: "Implement per-model evaluation loop"

**Commit 2.3.2**: "Add full Qwen model roster to config"

**Commit 2.3.3**: "Add Llama, Mistral, Phi model roster"

**Commit 2.3.4**: "Save per-model results to JSONL"

**Commit 2.3.5**: "Aggregate per-model statistics"

**After Phase 2, you should have:**
- ✅ Model loader with caching
- ✅ Generation + uncertainty extraction
- ✅ Per-model evaluation on all datasets/folds
- ✅ Results saved to JSONL
- ✅ Per-model summaries

---

## Phases 3–7: Continue with Similar Structure

For brevity, outline remaining phases:

### Phase 3: Feature Engineering (Commits ~36–50)
- Extract token-level features (entropy, logprob, variance)
- Compute sequence-level aggregates
- Add advanced features (embedding similarity, attention entropy)
- Validate ranges, log statistics

### Phase 4: Calibration & Evaluation (Commits ~51–70)
- Baseline metrics (accuracy, hallucination rate)
- Implement calibration methods (LogReg, temperature scaling, isotonic, Platt)
- Evaluate on held-out fold
- Abstention strategy comparison
- Generate final summary JSON

### Phase 5: Visualization (Commits ~71–85)
- Uncertainty distributions by correctness
- AUROC heatmap
- Reliability diagrams
- Feature importance
- Strategy comparison plots
- Generalization analysis
- Compilation into report

### Phase 6: Reproducibility (Commits ~86–95)
- Write docs (REPRODUCIBILITY.md, DATA_SPLITS.md, FEATURE_SPECS.md)
- Add unit tests
- Create sanity_checks.py
- Validate all outputs

### Phase 7: Publication (Commits ~96–100)
- Paper outline
- Publication figures
- Final README
- License + contribution guidelines

---

## Entry Point Script

**Create `scripts/run_full_pipeline.py`**:

```python
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging import setup_logging
from src.config import Config
from src.pipeline import HallucinationDetectionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config YAML file')
    parser.add_argument('--phase', type=int, default=0, help='Start from phase (0-7)')
    parser.add_argument('--models', nargs='+', help='Override models to test')
    parser.add_argument('--datasets', nargs='+', help='Override datasets to test')
    args = parser.parse_args()
    
    # Setup
    logger, log_path = setup_logging()
    logger.info(f"Starting HallucinationDetectionPipeline")
    logger.info(f"Log file: {log_path}")
    
    # Load config
    if Path(args.config).exists():
        config = Config.load_yaml(args.config)
    else:
        config = Config()
    
    # Override if specified
    if args.models:
        config.models = args.models
    if args.datasets:
        config.datasets = args.datasets
    
    # Log config
    config.log_config(logger)
    
    # Run pipeline
    pipeline = HallucinationDetectionPipeline(config)
    
    if args.phase <= 0:
        datasets, splits = pipeline.run_phase_1()
    if args.phase <= 1:
        pipeline.run_phase_2()
    # ... phases 3-7
    
    logger.info("="*70)
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: results/")
    logger.info("="*70)

if __name__ == '__main__':
    main()
```

**Commit 7.0.3**: "Create main entry point script"

---

## Git Commit Frequency Checklist

✅ **After every small task, commit:**
- Data loader for one dataset → 1 commit
- K-fold split generation → 1 commit
- Feature extraction function → 1 commit
- Calibration method → 1 commit
- Plot/figure → 1 commit
- Documentation file → 1 commit
- Test file → 1 commit

✅ **Commit messages should be clear and traceable:**
```
✓ 1.1.1: Load TriviaQA from HF datasets with validation
✓ 1.2.3: Verify no data leakage in k-fold splits
✓ 3.1.5: Compute entropy/logprob/variance features with validation
✓ 4.2.6: Fit and evaluate all calibration methods on test set
```

✅ **After EACH commit, push:**
```bash
git add -A
git commit -m "✓ X.X.X: Description"
git push origin main
```

---

## Logging Best Practices

**Log at key points:**

```python
# Start of major section
logger.info("="*70)
logger.info("PHASE X: Description")
logger.info("="*70)

# Data loading
logger.info(f"Loaded {dataset_name}: {len(dataset)} examples")
logger.info(f"  Example: {dataset[0]['question'][:50]}...")

# Splits creation
logger.info(f"Generated {n_splits}-fold CV")
logger.info(f"  Fold 0: train={len(train)}, test={len(test)}")

# Model evaluation
logger.info(f"Evaluating {model_id} on {dataset_name} fold {fold_id}")
logger.info(f"  Generated {n_examples} answers")
logger.info(f"  Accuracy: {acc:.3f}")

# Results aggregation
logger.info(f"AUROC by model:")
for model, auroc in results.items():
    logger.info(f"  {model}: {auroc:.3f}")

# Success
logger.info("✓ All sanity checks passed")
```

---

## Expected Directory Structure After Completion

```
hallucination-detection-llm/
├── .git/
├── .gitignore
├── README.md (updated with results)
├── COMPREHENSIVE_BLUEPRINT.md
├── CLAUDE_CODE_PROMPT.md
├── AUDIT_REPORT.md
├── setup.py
├── pyproject.toml
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── logging.py
│   ├── pipeline.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py (all 5+ datasets)
│   │   ├── splits.py (k-fold CV)
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── inference.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── token_level.py
│   │   ├── sequence_level.py
│   │   ├── attention.py
│   │   ├── semantic.py
│   │   └── ensemble.py
│   ├── labels/
│   │   ├── __init__.py
│   │   ├── correctness.py
│   │   ├── mc1_scoring.py
│   │   └── validation.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── temperature_scaling.py
│   │   ├── isotonic.py
│   │   ├── platt.py
│   │   ├── ensemble.py
│   │   └── evaluation.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── plotting.py
│       └── reporting.py
│
├── scripts/
│   ├── run_full_pipeline.py
│   ├── run_single_model.py
│   ├── run_kfold_cv.py
│   ├── generate_reports.py
│   └── sanity_checks.py
│
├── notebooks/
│   ├── 01_explore_datasets.ipynb
│   ├── 02_verify_splits.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_splits.py
│   ├── test_inference.py
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_calibration.py
│
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── DATA_SPLITS.md
│   ├── FEATURE_SPECS.md
│   ├── RESULTS_FORMAT.md
│   └── API_REFERENCE.md
│
├── results/
│   ├── logs/
│   │   ├── run_20260101_120000.log
│   │   └── ... (one per run)
│   ├── data/
│   │   ├── splits_all_datasets.json
│   │   ├── qwen2.5-0.5b_fold0_triviaqa.jsonl
│   │   ├── ... (per-model results)
│   │   ├── baseline_metrics.json
│   │   └── ...
│   ├── figures/
│   │   ├── uncertainty_distributions.png
│   │   ├── auroc_matrix.png
│   │   ├── reliability_diagrams.png
│   │   ├── feature_importance.png
│   │   ├── strategy_comparison.png
│   │   ├── generalization_heatmap.png
│   │   ├── scaling_curves.png
│   │   ├── ece_comparison.png
│   │   └── ... (12–15 PNGs total)
│   ├── summaries/
│   │   ├── final_summary.json
│   │   ├── qwen2.5-0.5b_summary.json
│   │   └── ... (per-model)
│   └── reports/
│       ├── README.md (index)
│       ├── FINDINGS.md (key findings)
│       ├── GENERALIZATION.md (feature transfer)
│       ├── qwen2.5-0.5b.md (per-model)
│       ├── triviaqa.md (per-dataset)
│       ├── FIGURES.md (gallery)
│       ├── report.html (self-contained)
│       └── paper_figures/ (publication quality)
│
└── config.yaml (user-facing configuration)
```

---

## Success Criteria

After running the full pipeline:

✅ **100 commits** with descriptive messages  
✅ **No data leakage** (verified by sanity_checks.py)  
✅ **10+ models** from 3+ families tested  
✅ **5+ datasets** evaluated (TriviaQA, TruthfulQA, SQuAD, HotpotQA, MMLU, custom)  
✅ **k-fold CV** on all datasets (5 folds per dataset)  
✅ **AUROC reported** for every model × dataset combination (50+ numbers)  
✅ **Generalization analysis** (which features transfer?)  
✅ **Multiple calibration methods** compared (LogReg, TemperatureScaling, Isotonic, Platt)  
✅ **Publication-quality figures** (8–15 PNG files)  
✅ **Comprehensive documentation** (REPRODUCIBILITY.md, FEATURE_SPECS.md, etc.)  
✅ **Unit tests pass** (pytest tests/ -v)  
✅ **README suitable for arXiv** (self-contained, clear results, reproducible)  

---

## Start Here

1. **Review `COMPREHENSIVE_BLUEPRINT.md`** (this covers all 7 phases in detail)
2. **Review `AUDIT_REPORT.md`** (understand current gaps)
3. **Start Phase 0**: Run `python scripts/run_full_pipeline.py --phase 0`
4. **After each task, commit + push** (see Git Workflow section)
5. **Log everything** (see Logging section)
6. **Aim for 100+ commits by the end**

**Expected timeline**: 7 weeks working on this (can parallelize with multiple GPUs).

**Good luck! 🚀**
