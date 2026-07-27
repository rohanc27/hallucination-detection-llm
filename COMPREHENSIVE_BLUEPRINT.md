# Hallucination Detection LLM — Comprehensive Implementation Blueprint

**Goal**: Build a publication-ready, model-agnostic hallucination detector with verified generalization across 10+ LLMs, 5+ datasets, with rigorous train/test isolation, extensive logging, and reproducible results.

**Expected output**: 
- Clean Python package with end-to-end pipeline
- 20+ experiment artifacts (JSON results, figures, tables)
- ~50+ git commits documenting every step
- Research-quality codebase suitable for arxiv/publication

---

## Phase 0: Project Setup & Infrastructure (Commits 1–5)

### 0.1 Git & Directory Structure
**Commit 0.1.1**: Initialize git repository structure
```
hallucination-detection-llm/
├── .git/
├── .gitignore (ignore large model weights, data caches)
├── README.md (updated)
├── requirements.txt (pin all versions)
├── pyproject.toml (optional, but professional)
├── setup.py (make it installable as a package)
├── 
├── src/
│   ├── __init__.py
│   ├── config.py (centralized config, all hyperparameters)
│   ├── logging.py (setup logging with timestamps, file output)
│   ├── constants.py (model names, dataset constants, paths)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py (load TriviaQA, TruthfulQA, SQuAD, HotpotQA, custom)
│   │   ├── splits.py (k-fold CV logic, deterministic split generation)
│   │   ├── preprocessing.py (question formatting, answer normalization)
│   │   └── validation.py (check for data leakage, splits integrity)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py (HuggingFace model/tokenizer loading with caching)
│   │   └── inference.py (generation + uncertainty extraction)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── token_level.py (entropy, logprob, variance)
│   │   ├── sequence_level.py (avg_entropy, min_logprob, seq_len)
│   │   ├── attention.py (attention entropy, if available)
│   │   ├── semantic.py (embedding similarity, reference-aware features)
│   │   └── ensemble.py (combine all features into feature matrix)
│   ├── labels/
│   │   ├── __init__.py
│   │   ├── correctness.py (substring + semantic similarity labeling)
│   │   ├── mc1_scoring.py (multiple-choice scoring)
│   │   └── validation.py (label quality checks)
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py (baseline)
│   │   ├── temperature_scaling.py
│   │   ├── isotonic.py
│   │   ├── platt.py
│   │   ├── ensemble.py (voting / weighted avg)
│   │   └── evaluation.py (AUROC, ECE, MCE, Brier, calibration metrics)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py (all evaluation metrics in one place)
│   │   ├── plotting.py (figures: reliability diagrams, feature importance, etc.)
│   │   └── reporting.py (JSON/CSV summaries, tables)
│   └── pipeline.py (orchestrate full end-to-end run)
├── notebooks/
│   ├── 01_explore_datasets.ipynb (EDA)
│   ├── 02_verify_splits.ipynb (ensure no leakage)
│   └── 03_results_analysis.ipynb (post-hoc analysis)
├── scripts/
│   ├── run_full_pipeline.py (entry point: python run_full_pipeline.py)
│   ├── run_single_model.py (debug one model)
│   ├── run_kfold_cv.py (k-fold evaluation)
│   ├── generate_reports.py (aggregate results, make figures)
│   └── sanity_checks.py (verify splits, check for data leakage)
├── results/
│   ├── logs/ (per-run logs with timestamps)
│   ├── data/ (per-model JSONL, per-fold CSVs)
│   ├── figures/ (PNG plots)
│   ├── summaries/ (aggregated JSON results)
│   └── reports/ (final tables, markdown summaries)
├── tests/
│   ├── test_data_loading.py (ensure datasets load correctly)
│   ├── test_splits.py (verify k-fold logic, no leakage)
│   ├── test_inference.py (model loading, generation)
│   ├── test_features.py (feature extraction matches spec)
│   ├── test_labels.py (labeling logic is correct)
│   └── test_calibration.py (calibration methods work)
└── docs/
    ├── PHASES.md (this file, but more detailed)
    ├── DATA_SPLITS.md (exact split definitions)
    ├── FEATURE_SPECS.md (description of each feature)
    ├── RESULTS_FORMAT.md (schema of all output JSONs)
    └── REPRODUCIBILITY.md (how to reproduce every result)
```

**Commit 0.1.2**: Create initial `setup.py`, `pyproject.toml`, `.gitignore`
```python
# setup.py
from setuptools import setup, find_packages
setup(
    name='hallucination-detection',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'transformers==4.44.2',
        'datasets==2.21.0',
        'sentence-transformers==3.1.1',
        'scikit-learn==1.5.2',
        'numpy>=1.24',
        'pandas>=2.0',
        'matplotlib>=3.8',
        'seaborn>=0.13',
        'tqdm>=4.66',
        'pyyaml>=6.0',
    ]
)
```

**Commit 0.1.3**: Set up logging infrastructure
- Create `src/logging.py` with:
  - File logging to `results/logs/{timestamp}.log`
  - Console logging with timestamps
  - Structured logging (include run ID, fold ID, model name in every log)
  - JSON logging option for automated parsing

**Commit 0.1.4**: Create centralized config system
- `src/config.py`: dataclass-based configuration
  - All model names, dataset paths, hyperparameters
  - Feature flags (which features to compute, which datasets to use)
  - Calibration method choices
  - K-fold settings (n_folds=5, random_state=42)
  - Output paths
- Load from YAML if provided, else use defaults
- Log full config at start of run

**Commit 0.1.5**: Initialize constants and paths
- `src/constants.py`: hardcode model IDs, dataset names
- Create output directories (`results/logs`, `results/data`, etc.)
- Verify write permissions

---

## Phase 1: Data & Methodology (Commits 6–25)

### 1.1 Data Collection (Commits 6–12)

**Commit 1.1.1**: Load TriviaQA
```python
# src/data/loaders.py
def load_triviaqa(split='validation', n_samples=None, random_state=42):
    """Load TriviaQA from HF datasets. Returns list of dicts with:
    - question
    - best_answer
    - correct_answers: List[str]
    - incorrect_answers: List[str]
    
    Log: number of examples, example structure
    """
```

**Commit 1.1.2**: Load TruthfulQA (generation + MC1)
```python
def load_truthfulqa(split='validation', n_samples=None):
    """Load TruthfulQA in both 'generation' and 'multiple_choice' formats.
    Returns two aligned datasets.
    
    Log: structure, number of MC choices per question
    """
```

**Commit 1.1.3**: Load SQuAD v2
```python
def load_squad_v2(split='validation', n_samples=None, include_unanswerable=True):
    """Load SQuAD v2 from HF datasets. Returns list of dicts with:
    - question
    - context
    - answers: List[str]
    - is_impossible
    
    Note: extractive QA, not generative. Measure if generated spans match reference.
    Log: split sizes, number of unanswerable questions
    """
```

**Commit 1.1.4**: Load HotpotQA
```python
def load_hotpotqa(split='validation', n_samples=None, difficulty='all'):
    """Load HotpotQA from HF. Returns list of dicts with:
    - question
    - supporting_facts: List[Tuple[title, sent_idx]]
    - answer
    - type (comparison / bridge)
    
    Log: split sizes, question types distribution
    """
```

**Commit 1.1.5**: Load MMLU (sampled)
```python
def load_mmlu_sampled(n_samples_per_subject=20, random_state=42):
    """Load 5-10 subjects from MMLU. Return multiple-choice QA examples.
    
    Log: subjects loaded, total examples
    """
```

**Commit 1.1.6**: Custom benchmark creation interface
```python
def load_custom_benchmark(filepath='data/custom_benchmark.json'):
    """Load custom benchmark from JSON. Expect format:
    [{
        "question": str,
        "correct_answer": str,
        "incorrect_answers": List[str],
        "reasoning": str (why this is correct),
        "difficulty": int (1-5)
    }, ...]
    
    Log: loaded N examples, difficulty distribution
    """
    
def create_custom_benchmark_template():
    """Generate empty template for manual curation."""
```

**Commit 1.1.7**: Data validation & schema enforcement
```python
def validate_dataset_schema(dataset, required_fields):
    """Assert all examples have required fields. Log any missing data."""
    
def check_dataset_overlap(ds1, ds2):
    """Warn if same questions appear in multiple datasets."""
```

**Commit 1.1.8**: Unified dataset registry
```python
# All datasets accessible via single function
DATASET_REGISTRY = {
    'triviaqa': load_triviaqa,
    'truthfulqa': load_truthfulqa,
    'squad_v2': load_squad_v2,
    'hotpotqa': load_hotpotqa,
    'mmlu': load_mmlu_sampled,
    'custom': load_custom_benchmark,
}

def load_dataset(name, **kwargs):
    """Load any dataset by name. Log structure."""
```

### 1.2 K-Fold CV & Splits (Commits 13–19)

**Commit 1.2.1**: Implement k-fold split generation
```python
# src/data/splits.py
def generate_kfold_splits(dataset, n_splits=5, random_state=42):
    """
    Stratified k-fold split.
    Returns:
    {
        'fold_0': {'train': [idx0, idx1, ...], 'test': [idx_n, ...]},
        'fold_1': {...},
        ...
    }
    
    CRITICAL: Use deterministic random_state. Log fold sizes.
    """
```

**Commit 1.2.2**: Save & load splits to disk
```python
def save_splits(splits, output_path='results/data/splits.json'):
    """Serialize splits to JSON for reproducibility."""
    
def load_splits(path='results/data/splits.json'):
    """Load previously generated splits."""
```

**Commit 1.2.3**: Verify no data leakage
```python
def verify_no_leakage(splits, dataset_name):
    """Check that:
    1. No question appears in train AND test for same fold
    2. No question appears in multiple folds
    3. All questions are covered
    
    Log any anomalies and raise AssertionError if leakage detected.
    """
```

**Commit 1.2.4**: Multi-dataset split coordination
```python
def generate_coordinated_splits(datasets_dict, n_splits=5, random_state=42):
    """
    For datasets with aligned examples (TriviaQA + TruthfulQA generation + MC1),
    ensure same fold assignment across all.
    
    For independent datasets, generate separate folds but use same seed
    for reproducibility.
    """
```

**Commit 1.2.5**: Split statistics logging
```python
def log_split_statistics(splits, dataset):
    """For each fold, log:
    - Train size, test size
    - Class balance (if applicable, e.g., correct/wrong ratio)
    - Question length statistics
    - Dataset-specific stats
    """
```

**Commit 1.2.6**: Create splits for all 5 datasets
- Call `generate_coordinated_splits` for TriviaQA + TruthfulQA
- Call `generate_kfold_splits` for SQuAD, HotpotQA, MMLU, custom
- Save all to `results/data/splits_all_datasets.json`
- Log: total splits, dataset coverage

**Commit 1.2.7**: Splits documentation
- Create `docs/DATA_SPLITS.md` with exact fold definitions
- Include seed, fold sizes, example questions from each fold
- Add visualization of split composition

**Commit 1.2.8**: Splits validation notebook
- Create `notebooks/02_verify_splits.ipynb`
- Load all splits, verify no leakage
- Show per-fold statistics table
- Save report to `results/reports/split_validation_report.md`

### 1.3 Labeling & Correctness (Commits 20–25)

**Commit 1.3.1**: Substring & semantic similarity labeling
```python
# src/labels/correctness.py
def label_substring(generated: str, correct_answers: List[str], 
                   incorrect_answers: List[str]) -> Optional[int]:
    """
    Returns 1 if any correct substring in generated and no incorrect substring.
    Log: substring matches found, false positives/negatives (if known).
    """

def label_semantic(generated: str, reference: str, 
                  threshold=0.65, model='all-MiniLM-L6-v2') -> Tuple[int, float]:
    """
    Returns (1 if cosine_sim >= threshold else 0, similarity score).
    Cache embeddings to avoid recomputation.
    Log: similarity distribution, threshold crossing rate.
    """
```

**Commit 1.3.2**: MC1 scoring
```python
def score_mc1(model, tokenizer, question: str, choices: List[str]) -> Dict:
    """
    Compute log-prob for each choice. Return:
    {
        'logprobs': [lp1, lp2, ...],
        'predicted_idx': argmax,
        'is_correct': bool,
        'top_prob': softmax(logprobs)[argmax]
    }
    
    Log: per-choice logprobs for this question
    """
```

**Commit 1.3.3**: Multi-label labeling system
```python
class CorrectnessLabeler:
    """
    Unified interface to get all labels for a question:
    - substring_correct
    - semantic_correct
    - mc1_correct
    
    Returns Dict with all labels + scores.
    Log: label agreement, conflicts.
    """
```

**Commit 1.3.4**: Label validation & quality checks
```python
def check_label_agreement(results_list):
    """Compare substring vs semantic vs MC1 labels. 
    Log: agreement rates, conflicting examples.
    """
    
def detect_label_noise(results_list):
    """Find examples where model gives very different answers on retries
    (if doing multiple generations per question).
    """
```

**Commit 1.3.5**: Labeling pipeline integration
- Integrate into main inference loop
- Ensure labels are computed for all datasets
- Cache labels to avoid recomputation
- Log label statistics per dataset, per model

---

## Phase 2: Multi-Model Testing (Commits 26–40)

### 2.1 Model Loading & Caching (Commits 26–29)

**Commit 2.1.1**: Unified model loader
```python
# src/models/loader.py
class ModelLoader:
    """Load HF models with:
    - Automatic dtype selection (float16 on GPU, float32 on CPU)
    - Local caching to avoid re-downloading
    - VRAM monitoring and warnings
    - Quantization support (8-bit, 4-bit) for large models
    """
    
def load_model_and_tokenizer(model_id, device='cuda', dtype='auto'):
    """Returns (model, tokenizer) with logging."""
```

**Commit 2.1.2**: Model registry
```python
# Complete roster of 10+ models
MODEL_REGISTRY = {
    # Qwen (3 sizes)
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
    
    # Llama (2-3 sizes)
    'llama-1b': 'meta-llama/Llama-2-1b-hf',
    'llama-7b': 'meta-llama/Llama-2-7b-hf',
    'llama-13b': 'meta-llama/Llama-2-13b-hf',
    
    # Mistral
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral-7b-v02': 'mistralai/Mistral-7B-Instruct-v0.2',
    
    # Phi
    'phi-2': 'microsoft/phi-2',
    'phi-3.8b': 'microsoft/Phi-3-mini-4k-instruct',
    
    # Gemma
    'gemma-2b': 'google/gemma-2b-it',
    'gemma-7b': 'google/gemma-7b-it',
}
```

**Commit 2.1.3**: VRAM management
```python
def monitor_vram():
    """Log current GPU memory usage."""
    
def clear_gpu_cache():
    """Empty PyTorch cache between models."""
    
class VRAMBudget:
    """Track VRAM usage, warn if approaching limit."""
```

**Commit 2.1.4**: Model metadata logging
```python
def log_model_info(model, model_id):
    """Log: parameter count, dtype, device, HF config."""
```

### 2.2 Generation & Inference (Commits 30–35)

**Commit 2.2.1**: Prompt templates (dataset + model aware)
```python
# src/models/inference.py
class PromptTemplate:
    """Build prompts for TriviaQA, SQuAD, HotpotQA, MMLU.
    
    For each dataset:
    - Few-shot examples if beneficial
    - Chat template if instruction-tuned
    - Plain prompt if base model
    """
    
def build_prompt(question, dataset_name, model_id, include_fewshot=False):
    """Return prompt string."""
```

**Commit 2.2.2**: Generation with uncertainty extraction
```python
@torch.no_grad()
def generate_with_uncertainty(model, tokenizer, prompt, max_new_tokens=50):
    """
    Returns Dict:
    {
        'text': generated answer,
        'token_logprobs': [lp1, lp2, ...],
        'token_entropies': [h1, h2, ...],
        'logits_per_token': [logits1, logits2, ...],  # full dist if needed
    }
    
    Log: per-token statistics.
    """
```

**Commit 2.2.3**: Token-level uncertainty signals
```python
def extract_token_entropy(logprobs):
    """Entropy of each token's distribution."""
    
def extract_token_logprob(logprobs):
    """Log-probability of each token."""
    
def extract_token_variance(logits):
    """Variance across token probability mass."""
```

**Commit 2.2.4**: Attention extraction (optional, for advanced features)
```python
@torch.no_grad()
def extract_attention_maps(model, tokenizer, prompt, max_new_tokens=50):
    """If model exposes attention, extract and aggregate attention entropy."""
```

**Commit 2.2.5**: Inference loop per model per fold
```python
def evaluate_model_on_fold(model_id, fold_data, dataset_name, fold_id, config):
    """
    For a single (model, fold, dataset) combination:
    1. Load model + tokenizer
    2. For each question:
       - Build prompt
       - Generate answer
       - Extract uncertainty
       - Compute labels
       - Save to JSONL
    3. Log per-question metrics
    4. Clear GPU cache
    
    Returns: path to results JSONL
    """
```

**Commit 2.2.6**: Batch processing & parallelization
```python
def evaluate_all_models(model_ids, fold_ids, dataset_names, config):
    """
    Schedule evaluation of all (model, fold, dataset) combinations.
    Options:
    - Sequential (simplest, safe)
    - Parallel datasets (different GPU if available)
    - Checkpoint/resume (save progress, relaunch if interrupted)
    
    Log: progress bar, ETA, save path
    """
```

**Commit 2.2.7**: Error handling & robustness
```python
def safe_evaluate(model_id, fold_data, dataset_name, fold_id, config):
    """Wrap evaluate_model_on_fold with:
    - Try/catch for model loading errors
    - Timeout handling (skip stuck questions)
    - VRAM error recovery
    - Checkpointing (resume from last saved question)
    
    Log: errors, recoveries.
    """
```

**Commit 2.2.8**: Per-model results aggregation
```python
def aggregate_model_fold_results(result_jsonls):
    """Read all JSONLs for a model.
    Compute:
    - Per-dataset statistics
    - Per-fold statistics
    - Cross-fold mean + std
    
    Save to results/data/{model_name}_summary.json
    """
```

---

## Phase 3: Feature Engineering & Enhancement (Commits 36–50)

### 3.1 Core Features (Commits 36–40)

**Commit 3.1.1**: Sequence-level features from token-level
```python
# src/features/sequence_level.py
def compute_avg_entropy(token_entropies):
    """Mean of all token entropies."""
    
def compute_avg_logprob(token_logprobs):
    """Mean log-probability across tokens."""
    
def compute_min_logprob(token_logprobs):
    """Minimum log-probability (weakest token)."""
    
def compute_entropy_std(token_entropies):
    """Variance / std of entropy (high if inconsistent)."""
```

**Commit 3.1.2**: Length-based features
```python
def compute_seq_length(token_logprobs):
    """Number of generated tokens."""
    
def compute_token_length_stats(answer_text):
    """Whitespace count, word count, char count."""
```

**Commit 3.1.3**: Position-aware features
```python
def compute_confidence_by_position(token_logprobs):
    """
    Segment into thirds: beginning, middle, end.
    Return (conf_begin, conf_mid, conf_end).
    High if model is consistently confident. Low if confidence drops mid-answer.
    """
    
def compute_entropy_trend(token_entropies):
    """Is entropy increasing/decreasing over sequence?"""
```

**Commit 3.1.4**: Self-confidence elicitation
```python
def build_confidence_prompt(question, answer, model_id, dataset_name):
    """Prompt: "On a scale 0-1, how confident in this answer?"
    Format varies by dataset + model family.
    """
    
def extract_self_confidence(confidence_completion):
    """Parse "0.75" from model output. Return float in [0, 1]."""
    
def log_confidence_parsing(successes, failures):
    """Log: parse success rate, examples of failures."""
```

**Commit 3.1.5**: Feature computation pipeline
```python
class FeatureExtractor:
    """Compute all features for a single question's result.
    
    Input: model, tokenizer, question, generated_text, token_logprobs, 
           token_entropies, answer_labels
    
    Output: Dict with all features:
    {
        'avg_entropy': ...,
        'avg_logprob': ...,
        'min_logprob': ...,
        'seq_length': ...,
        'entropy_std': ...,
        'confidence_by_position': [...],
        'entropy_trend': ...,
        'self_confidence': ...,
        ...
    }
    """
    
def log_feature_statistics(features_list):
    """For a model on a fold:
    - Per-feature mean, std, min, max
    - Correlation with correctness
    - Feature importance (via sklearn)
    """
```

### 3.2 Advanced Features (Commits 41–46)

**Commit 3.2.1**: Embedding-based features
```python
# src/features/semantic.py
def compute_embedding_similarity_to_reference(generated, reference, model='all-MiniLM-L6-v2'):
    """Cosine similarity to correct answer. High = similar to ground truth."""
    
def compute_embedding_self_similarity(generated_tokens):
    """How self-similar are the generated tokens? (detects repetition)."""
```

**Commit 3.2.2**: Attention entropy (if available)
```python
# src/features/attention.py
def extract_attention_entropy(attention_heads):
    """For each layer/head, compute entropy of attention distribution.
    Aggregate into per-layer and per-head summaries.
    
    Intuition: uncertain models attend broadly; confident ones focus.
    """
```

**Commit 3.2.3**: Answer consistency (retrial)
```python
def generate_with_different_seeds(model, tokenizer, prompt, n_generations=3):
    """Generate N times with different random seeds (sampling-based if needed).
    Measure agreement: if all 3 generations are identical, model is confident.
    If they differ, model is uncertain.
    
    Compute: agreement_rate, Jaccard similarity, BLEU among generations.
    """
```

**Commit 3.2.4**: Dataset-specific features
```python
# For SQuAD: extractive QA
def compute_span_coverage(generated, context):
    """Does generated answer appear as a span in context?"""
    
# For HotpotQA: multi-hop
def compute_supporting_fact_coverage(generated, supporting_facts):
    """Do supporting facts appear in generated answer?"""
    
# For MMLU: multiple choice
def compute_option_confidence_spread(logprobs_per_option):
    """Is top choice much more likely than others? (high spread = confident)."""
```

**Commit 3.2.5**: Feature interaction & engineering
```python
def compute_feature_interactions(features_dict):
    """Create polynomial features, ratios, etc.
    E.g., entropy/confidence, length-normalized entropy.
    """
    
def select_important_features(features_list, labels, top_k=15):
    """Use sklearn's feature_importances_ or correlation to identify
    most predictive features. Log top_k features + importances.
    """
```

**Commit 3.2.6**: Feature validation & debugging
```python
def validate_feature_values(features):
    """Check for NaN, inf, out-of-range values. Log anomalies."""
    
def visualize_feature_distributions(features_list, labels):
    """For each feature, create histogram by correctness (correct vs wrong).
    Save to results/figures/feature_distributions/.
    """
```

### 3.3 Feature Pipeline Integration (Commits 47–50)

**Commit 3.3.1**: Unified feature computation
```python
def compute_all_features(model_id, question_result, config):
    """Single entry point: given a question's inference result,
    compute all enabled features based on config.
    
    Return: feature_dict
    """
```

**Commit 3.3.2**: Feature caching & persistence
```python
def cache_features(model_id, dataset_name, fold_id, features_list, output_dir):
    """Save features to CSV for faster reloading."""
    
def load_cached_features(model_id, dataset_name, fold_id, input_dir):
    """Load from cache if available, else recompute."""
```

**Commit 3.3.3**: Feature documentation
- Create `docs/FEATURE_SPECS.md` with:
  - Definition of each feature
  - Expected range / distribution
  - Why it's predictive
  - Example values

**Commit 3.3.4**: Feature correlation analysis
```python
def analyze_feature_correlations(features_df, output_path):
    """Compute correlation matrix. Save heatmap to PNG.
    Identify redundant features.
    """
```

---

## Phase 4: Calibration & Evaluation (Commits 51–70)

### 4.1 Baseline Evaluation (Commits 51–56)

**Commit 4.1.1**: Per-model baseline metrics
```python
# src/evaluation/metrics.py
def compute_baseline_metrics(results_dict, labels_key='semantic_correct'):
    """On a full model's results (no abstention):
    - Accuracy
    - Hallucination rate (wrong & confident)
    - Per-class accuracy (correct vs wrong)
    - Confidence distribution
    """
    
def log_baseline_metrics(model_id, metrics_dict):
    """Pretty-print to console + log file."""
```

**Commit 4.1.2**: Calibration metrics
```python
def compute_ece(confidences, labels, n_bins=10):
    """Expected Calibration Error.
    For each confidence bin, compute |predicted_confidence - actual_accuracy|.
    Return: ECE, per-bin statistics.
    """
    
def compute_mce(confidences, labels, n_bins=10):
    """Maximum Calibration Error (max bin-wise gap)."""
    
def compute_brier_score(confidences, labels):
    """Mean squared error between predicted and actual."""
```

**Commit 4.1.3**: AUROC & AUPRC
```python
def compute_auroc(model_confidences, labels):
    """Area under ROC curve (main metric for hallucination detection)."""
    
def compute_auprc(model_confidences, labels):
    """Area under precision-recall curve."""
    
def compute_auroc_per_dataset(results_by_dataset):
    """AUROC on each dataset separately, then macro-average."""
```

**Commit 4.1.4**: Uncertainty signal analysis
```python
def analyze_signal_separability(signal, labels):
    """For entropy, logprob, self_confidence individually:
    - Compute AUC
    - Visualize distribution (correct vs wrong)
    - Log separability statistics
    """
```

**Commit 4.1.5**: Reliability diagrams
```python
def plot_reliability_diagram(confidences, labels, output_path, title=''):
    """Scatter: (confidence, accuracy) with bin sizes.
    Diagonal line = perfect calibration.
    Save PNG.
    """
```

**Commit 4.1.6**: Baseline results aggregation
```python
def aggregate_all_baseline_metrics(model_results, output_dir):
    """Compute baseline metrics for all models × datasets × folds.
    Save to results/data/baseline_metrics.json.
    Log: tables of results per model.
    """
```

### 4.2 Calibration Methods (Commits 57–63)

**Commit 4.2.1**: Logistic Regression (baseline)
```python
# src/calibration/logistic_regression.py
class LogisticRegressionCalibration:
    """Train on fold features, test on held-out fold.
    
    Methods:
    - fit(X_train, y_train)
    - predict_proba(X_test) -> confidences
    - save(path), load(path)
    
    Log: coefficients, feature importance.
    """
```

**Commit 4.2.2**: Temperature Scaling
```python
# src/calibration/temperature_scaling.py
class TemperatureScaling:
    """Multiply model logits by learned T before softmax.
    1.0 = no scaling, T > 1.0 = soften, T < 1.0 = sharpen.
    
    Fit on held-out calibration set, apply to test set.
    """
```

**Commit 4.2.3**: Isotonic Regression
```python
# src/calibration/isotonic.py
class IsotonicCalibration:
    """Non-parametric monotonic regression.
    More flexible than temperature scaling, can fit complex distortions.
    """
```

**Commit 4.2.4**: Platt Scaling
```python
# src/calibration/platt.py
class PlattScaling:
    """Fit logistic regression to binary labels.
    Similar to LogReg but stricter assumptions (sigmoid shape).
    """
```

**Commit 4.2.5**: Ensemble calibration
```python
# src/calibration/ensemble.py
class EnsembleCalibration:
    """Combine multiple calibration methods:
    - Majority voting (if methods > 0.5, predict correct)
    - Weighted averaging (weight by each method's AUROC)
    - Stacking (train meta-classifier on outputs of all methods)
    """
```

**Commit 4.2.6**: Calibration training pipeline
```python
def fit_calibration_methods(X_train, y_train, methods=['logistic', 'temp_scaling', 'isotonic']):
    """Fit all requested methods on same training data.
    Return dict of fitted calibrators.
    """
    
def evaluate_calibration_methods(calibrators, X_test, y_test):
    """For each method, evaluate AUROC, ECE, Brier.
    Log comparison table.
    """
```

**Commit 4.2.7**: Per-model calibration
```python
def calibrate_all_models(features_by_model_fold, config):
    """For each model:
    - Folds 1-4: pool data, fit calibration heads
    - Fold 5: evaluate on test set
    - Save fitted calibrators
    
    Log: calibration metrics per model.
    """
```

### 4.3 Abstention Strategies (Commits 64–67)

**Commit 4.3.1**: Abstention sweep
```python
def sweep_confidence_thresholds(confidences, labels, thresholds=np.linspace(0, 1, 11)):
    """For each threshold t:
    - Answered: predict if confidence >= t
    - Conditional accuracy: accuracy on answered questions
    - Hallucination rate: (answered & wrong) / total
    - Abstention rate: (1 - answered) / total
    
    Return: per-threshold metrics.
    """
```

**Commit 4.3.2**: Strategy comparison (matched abstention rate)
```python
def compare_strategies_at_fixed_abstention(results, target_abstention=0.30):
    """Compare at a fixed abstention budget (e.g., 30%):
    - No calibration (baseline)
    - Single-signal (entropy threshold)
    - Single-signal (self-confidence threshold)
    - Trained calibration head (logistic regression)
    
    At target abstention rate, measure:
    - Answered accuracy
    - Hallucination rate
    
    Log: comparison table, figures.
    """
```

**Commit 4.3.3**: Multi-dataset abstention
```python
def analyze_abstention_across_datasets(results_by_dataset, target_abstention=0.30):
    """Show how abstention strategy generalizes:
    - Train on TriviaQA
    - Test on TruthfulQA, SQuAD, HotpotQA, MMLU
    
    Log: per-dataset hallucination rate with abstention.
    """
```

**Commit 4.3.4**: Optimal threshold selection
```python
def find_optimal_threshold(confidences, labels, objective='auroc'):
    """Find threshold that maximizes:
    - AUROC (default)
    - F1 at fixed precision
    - Cost-weighted loss (false negative cost vs false positive cost)
    """
```

### 4.4 Evaluation Reporting (Commits 68–70)

**Commit 4.4.1**: Comprehensive results table
```python
def generate_results_table(results_by_model_dataset, output_path):
    """
    Rows: model
    Cols: AUROC@TriviaQA, AUROC@TruthfulQA, AUROC@SQuAD, AUROC@HotpotQA, AUROC@MMLU, Avg
    
    Also: ECE, Brier, Best Abstention Strategy
    
    Save as CSV + formatted Markdown table.
    Log: print to console.
    """
```

**Commit 4.4.2**: Generalization analysis
```python
def analyze_generalization(results_by_model_dataset):
    """
    For each feature:
    - Correlation with correctness on TriviaQA
    - Correlation on TruthfulQA, SQuAD, HotpotQA, MMLU
    
    Identify:
    - Which features transfer (high corr across datasets)
    - Which are dataset-specific
    - Which break (negative corr on adversarial benchmarks)
    
    Create: feature generalization heatmap.
    """
```

**Commit 4.4.3**: Final summary JSON
```python
def save_final_summary(all_results, output_path='results/summaries/final_summary.json'):
    """
    Nested JSON with all results:
    {
        'config': {...},
        'models': {
            'qwen2.5-0.5b': {
                'datasets': {
                    'triviaqa': {
                        'folds': {
                            '0': {'auroc': 0.85, 'ece': 0.04, ...},
                            ...
                        },
                        'mean_auroc': 0.85,
                        'std_auroc': 0.02,
                    },
                    ...
                },
            },
            ...
        },
        'cross_model_comparison': {...},
        'generalization_analysis': {...},
    }
    """
```

---

## Phase 5: Visualization & Reporting (Commits 71–85)

### 5.1 Core Figures (Commits 71–78)

**Commit 5.1.1**: Uncertainty distributions
```python
# src/evaluation/plotting.py
def plot_uncertainty_by_correctness(results, feature_names=['avg_entropy', 'avg_logprob']):
    """Box plots of each feature split by correct/wrong.
    Row per feature, column per model.
    Save: results/figures/uncertainty_distributions.png
    """
```

**Commit 5.1.2**: AUROC by model & dataset
```python
def plot_auroc_matrix(results_by_model_dataset):
    """Heatmap: rows=models, cols=datasets, values=AUROC.
    Show generalization at a glance.
    Save: results/figures/auroc_matrix.png
    """
```

**Commit 5.1.3**: Reliability diagrams (multi-model)
```python
def plot_reliability_diagrams_multi(results_by_model):
    """Subplots (one per model) of confidence vs accuracy.
    Save: results/figures/reliability_diagrams.png
    """
```

**Commit 5.1.4**: Feature importance across models
```python
def plot_feature_importance(calibrators_by_model):
    """For logistic regression, plot coefficients for each model.
    Show which features matter where.
    Save: results/figures/feature_importance.png
    """
```

**Commit 5.1.5**: Abstention strategy comparison
```python
def plot_strategy_comparison(strategy_results, target_abstention=0.30):
    """
    Grouped bar chart:
    - X: model
    - Groups: no-calibration vs entropy vs self-conf vs trained-head
    - Y: hallucination rate at target abstention
    
    Save: results/figures/strategy_comparison.png
    """
```

**Commit 5.1.6**: Generalization heatmap
```python
def plot_generalization_analysis(feature_correlations_by_dataset):
    """
    Heatmap: rows=features, cols=datasets, values=correlation with correctness.
    Show which features generalize.
    Save: results/figures/feature_generalization.png
    """
```

**Commit 5.1.7**: Scaling curves (model size vs performance)
```python
def plot_scaling_curves(results_by_model_size):
    """
    X: parameter count (log scale)
    Y: AUROC (or other metric)
    Multiple curves (one per dataset)
    
    Verify: larger models = better hallucination detection?
    Save: results/figures/scaling_curves.png
    """
```

**Commit 5.1.8**: ECE comparison
```python
def plot_ece_comparison(ece_results_by_model_calibration_method):
    """
    Bar chart: models on X, ECE on Y.
    Grouped by calibration method (logistic, temp_scaling, isotonic, etc.)
    
    Show: which calibration method best for each model?
    Save: results/figures/ece_comparison.png
    """
```

### 5.2 Summary Reports (Commits 79–82)

**Commit 5.2.1**: Main findings report
```python
def generate_findings_markdown():
    """results/reports/FINDINGS.md
    
    Sections:
    - Executive Summary (1 paragraph)
    - Key Findings (3-5 bullet points)
    - AUROC by Model & Dataset (table)
    - Best Practices (what works)
    - Limitations (what doesn't)
    - Future Work
    """
```

**Commit 5.2.2**: Per-model deep dive
```python
def generate_model_reports():
    """For each model, create results/reports/{model_name}.md
    
    - Metadata (params, training data, etc.)
    - AUROC per dataset
    - Feature importance
    - Calibration quality
    - Ablation study (which features matter most)
    - Example questions (confident correct, confident wrong, etc.)
    """
```

**Commit 5.2.3**: Cross-dataset analysis
```python
def generate_dataset_reports():
    """For each dataset, create results/reports/{dataset_name}.md
    
    - Dataset overview (size, question types, etc.)
    - AUROC by model
    - Which models struggle (lowest AUROC)?
    - Feature importance on this dataset
    - Dataset-specific insights
    """
```

**Commit 5.2.4**: Generalization report
```python
def generate_generalization_report():
    """results/reports/GENERALIZATION.md
    
    Main finding: which uncertainty signals generalize?
    - Train on TriviaQA, test on {TruthfulQA, SQuAD, HotpotQA, MMLU}
    - Show feature correlation matrix
    - Explain breakdowns (e.g., why self-confidence fails on TruthfulQA)
    """
```

### 5.3 Figure Compilation & Web Report (Commits 83–85)

**Commit 5.3.1**: Figure gallery
```python
def compile_figure_gallery():
    """
    Generate results/reports/FIGURES.md
    Markdown file with all PNG figures embedded.
    Render in browser or markdown viewer.
    """
```

**Commit 5.3.2**: HTML report (optional)
```python
def generate_html_report():
    """
    results/reports/report.html
    
    Self-contained HTML with:
    - Embedded figures (PNGs as base64)
    - Tables (sortable, searchable)
    - Navigation menu
    - Results summary
    """
```

**Commit 5.3.3**: Report index
```python
def generate_report_index():
    """results/reports/README.md
    
    Links to all reports:
    - FINDINGS.md
    - GENERALIZATION.md
    - Per-model reports
    - Per-dataset reports
    - FIGURES.md
    - figures/ (raw PNG directory)
    """
```

---

## Phase 6: Reproducibility & Documentation (Commits 86–95)

### 6.1 Reproducibility Guide (Commits 86–89)

**Commit 6.1.1**: REPRODUCIBILITY.md
```markdown
# Reproducibility Guide

## Full replication (all models, all datasets, all folds)
1. Clone repo + install: `pip install -e .`
2. Configure GPU (or use CPU): `config.yaml`
3. Run: `python scripts/run_full_pipeline.py --config config.yaml`
4. Expected runtime: X hours on GPU Y
5. Expected disk usage: X GB

## Partial replication (debug one model)
python scripts/run_single_model.py --model qwen2.5-0.5b --dataset triviaqa --fold 0

## Checking for data leakage
python scripts/sanity_checks.py

## Comparing to published results
python scripts/verify_baseline.py
```

**Commit 6.1.2**: DATA_SPLITS.md
```markdown
# Data Splits Documentation

## Fold Generation
- Method: Stratified k-fold (5 folds)
- Random seed: 42
- Stratification: by correct/wrong ratio

## Split Sizes
- TriviaQA: 1000 questions → 800 train (folds 1-4) + 200 test (fold 5)
- TruthfulQA: 817 questions → 653 train + 164 test
- SQuAD: ... etc

## Verification
Run: `python scripts/run_kfold_cv.py --verify`
This confirms no data leakage.

## Reproducibility
All splits are deterministic given seed=42.
See splits.json for exact assignments.
```

**Commit 6.1.3**: FEATURE_SPECS.md
```markdown
# Feature Specifications

## avg_entropy (nats)
- Definition: Mean entropy of token probability distributions
- Range: [0, log(vocab_size)] ≈ [0, 10]
- Interpretation: High = uncertain, Low = confident
- Correlation with correctness: Negative (uncertain = wrong)
- Dataset generalization: Generalizes well

## avg_logprob (nats)
- Definition: Mean log-probability of generated tokens
- Range: [-log(vocab_size), 0] ≈ [-10, 0]
- Interpretation: High (close to 0) = confident, Low (very negative) = uncertain
- Correlation with correctness: Positive
- Dataset generalization: Generalizes well

## self_confidence [0, 1]
- Definition: Model's self-reported confidence on 0-1 scale
- Method: Prompt engineering + parsing
- Interpretation: Directly self-reported
- Correlation with correctness: Positive for large models, near-zero for small
- Dataset generalization: Poor (TruthfulQA designed to elicit false confidence)

... (etc. for all features)
```

**Commit 6.1.4**: RESULTS_FORMAT.md
```markdown
# Output Format Specification

## Per-question JSONL (results/data/{model_id}_fold{fold_id}_{dataset}.jsonl)
```json
{
  "idx": 0,
  "fold_id": 0,
  "question": "Who was the first president of the US?",
  "generated_answer": "George Washington",
  "labels": {
    "substring_correct": 1,
    "semantic_correct": 1,
    "semantic_similarity": 0.98,
    "mc1_correct": 1
  },
  "features": {
    "avg_entropy": 0.45,
    "avg_logprob": -0.12,
    "min_logprob": -2.3,
    "seq_length": 5,
    "self_confidence": 0.92,
    ...
  },
  "tokens": {
    "text": ["George", "Washington"],
    "logprobs": [-0.1, -0.14],
    "entropies": [0.5, 0.4]
  }
}
```

## Per-model summary (results/summaries/{model_id}_summary.json)
```json
{
  "model_id": "qwen2.5-0.5b",
  "datasets": {
    "triviaqa": {
      "fold_0": {
        "auroc": 0.85,
        "ece": 0.04,
        "accuracy": 0.30,
        "n_examples": 200
      },
      ...
    },
    "mean_auroc": 0.85,
    "mean_ece": 0.04
  }
}
```

## Final summary (results/summaries/final_summary.json)
(See Phase 4, Commit 4.4.3)
```

### 6.2 Code Documentation (Commits 90–92)

**Commit 6.2.1**: Docstrings & type hints
```python
# Every function has:
# - Type hints on all params + return
# - Comprehensive docstring (description, params, returns, raises, example)
# - Inline comments for non-obvious logic

def compute_auroc(model_confidences: np.ndarray, 
                  labels: np.ndarray) -> float:
    """
    Compute area under the receiver operating characteristic curve.
    
    This is the primary metric for hallucination detection. AUC of 0.5 = random,
    1.0 = perfect discrimination between correct and hallucinated answers.
    
    Args:
        model_confidences: Predicted confidence [0, 1] for each question. Shape (N,).
        labels: Ground truth labels {0, 1}. Shape (N,). 1 = correct, 0 = wrong.
    
    Returns:
        auroc: Float in [0, 1].
    
    Raises:
        ValueError: If labels do not have at least two classes.
    
    Example:
        >>> auroc = compute_auroc(np.array([0.9, 0.1]), np.array([1, 0]))
        >>> auroc
        1.0
    """
    ...
```

**Commit 6.2.2**: Module documentation (README in each subdir)
```
src/data/README.md
src/models/README.md
src/features/README.md
src/calibration/README.md
src/evaluation/README.md
```

**Commit 6.2.3**: API reference
```markdown
# API Reference

## src.data

### load_dataset(name: str, **kwargs) -> List[Dict]
Load a dataset by name. See DATASET_REGISTRY for available datasets.

### generate_kfold_splits(dataset, n_splits=5, random_state=42) -> Dict
Generate k-fold splits.

### CorrectnessLabeler.label(question, answer, dataset_name) -> Dict
Compute all labels (substring, semantic, MC1) for a Q&A pair.

## src.models

### ModelLoader.load(model_id: str, device='cuda') -> Tuple[Model, Tokenizer]
Load HF model with dtype/device handling.

### generate_with_uncertainty(...) -> Dict
Generate answer + extract uncertainty signals.

... (etc.)
```

### 6.3 Testing & CI (Commits 93–95)

**Commit 6.3.1**: Unit tests
```python
# tests/test_data_loading.py
def test_load_triviaqa():
    """Verify TriviaQA loads correctly."""
    ds = load_dataset('triviaqa', n_samples=10)
    assert len(ds) == 10
    assert 'question' in ds[0]
    assert 'best_answer' in ds[0]

# tests/test_splits.py
def test_kfold_no_leakage():
    """Verify no question in train and test."""
    ds = load_dataset('triviaqa', n_samples=100)
    splits = generate_kfold_splits(ds, n_splits=5)
    verify_no_leakage(splits, 'triviaqa')  # raises AssertionError if leakage

# tests/test_inference.py
def test_model_loading():
    """Verify model loads without error."""
    model, tok = ModelLoader.load('qwen2.5-0.5b', device='cpu')
    assert model is not None

# tests/test_features.py
def test_feature_extraction():
    """Verify feature values are in valid ranges."""
    features = extract_all_features(...)
    assert 0 <= features['entropy'] <= 10
    assert -10 <= features['avg_logprob'] <= 0
```

**Commit 6.3.2**: GitHub Actions CI (optional, if on GitHub)
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - run: pip install -e .
      - run: pytest tests/ -v
```

**Commit 6.3.3**: Sanity checks script
```python
# scripts/sanity_checks.py
def check_data_integrity():
    """Load all datasets, verify no missing data."""
    
def check_splits_integrity():
    """Load all splits, verify no leakage."""
    
def check_model_loading():
    """Try loading each model in MODEL_REGISTRY."""
    
def check_feature_ranges():
    """Load sample results, verify features in valid ranges."""

if __name__ == '__main__':
    check_data_integrity()
    check_splits_integrity()
    check_model_loading()
    check_feature_ranges()
    print("✓ All sanity checks passed!")
```

---

## Phase 7: Final Writeup & Publication (Commits 96–100)

**Commit 7.0.1**: Paper outline
```markdown
# Paper Outline: Hallucination Detection via Multi-Signal Uncertainty Estimation

## 1. Introduction
- Motivation: Hallucination is a major failure mode of LLMs
- Gap: Prior work tests on 1-2 datasets, 1-2 models
- Our contribution: Systematic evaluation across 10+ models, 5+ datasets

## 2. Background
- Related work: Kadavath et al., Burns et al., other calibration literature
- Definition: hallucination = confident incorrect answer

## 3. Method
- Dataset overview (TriviaQA, TruthfulQA, etc.)
- Feature engineering (entropy, logprob, self-confidence, etc.)
- Calibration methods (LogReg, temperature scaling, isotonic, etc.)
- Evaluation setup (k-fold CV, metrics: AUROC, ECE, etc.)

## 4. Results
- AUROC by model and dataset
- Generalization analysis: which features transfer?
- Ablation study: which features matter most?
- Calibration method comparison

## 5. Discussion
- Key findings (e.g., "self-confidence emerges at scale")
- Limitations (e.g., "poor on adversarial benchmarks")
- Implications (how to use this in practice)

## 6. Conclusion & Future Work
```

**Commit 7.0.2**: Final results table & figures for paper
- Create publication-quality figures (larger fonts, proper legends)
- Compile main results table (AUROC by model × dataset)
- Add supplementary tables (feature importance, per-fold breakdown, etc.)
- Save to results/reports/paper_figures/

**Commit 7.0.3**: README update (for arXiv, GitHub)
```markdown
# Hallucination Detection in LLMs via Uncertainty Estimation — Generalization Study

**TL;DR**: We systematically evaluate hallucination detection across 10+ LLMs and 5+ datasets,
finding that uncertainty signals (entropy, logprob) generalize well but adversarial benchmarks
require stronger methods.

## Key Results
- AUROC 0.80–0.92 on general-knowledge QA (TriviaQA, TruthfulQA-generation)
- AUROC 0.50–0.60 on adversarial benchmarks (TruthfulQA designed for false confidence)
- Self-reported confidence emerges as a scaling law
- [Other key findings]

## Quick Start
```bash
git clone https://github.com/rchakrav/hallucination-detection-llm
cd hallucination-detection-llm
pip install -e .

# Full pipeline
python scripts/run_full_pipeline.py --config config.yaml

# Single model (debug)
python scripts/run_single_model.py --model qwen2.5-0.5b --dataset triviaqa --fold 0

# View results
open results/reports/FINDINGS.md
```

## Project Structure
- `src/`: Core library (data, models, features, calibration, evaluation)
- `scripts/`: Entry points (pipeline, single-model, sanity checks)
- `results/`: Outputs (logs, data, figures, summaries, reports)
- `notebooks/`: EDA, verification, analysis

## Reproducibility
See `docs/REPRODUCIBILITY.md` for full replication instructions.
All splits, seeds, and hyperparameters are documented and deterministic.

## Citation
If you use this work, please cite:
```bibtex
@article{chakravarthi2026hallucination,
  title={Hallucination Detection in LLMs via Uncertainty Estimation: A Systematic Generalization Study},
  author={Chakravarthi, Rohan},
  journal={arXiv preprint},
  year={2026}
}
```
```

**Commit 7.0.4**: License & contribution guidelines
- Add LICENSE (MIT, Apache 2.0, etc.)
- Add CONTRIBUTING.md (if open-sourcing)
- Add CODE_OF_CONDUCT.md

---

## Git Workflow Summary (100 Commits Total)

| Phase | Commits | Description |
|-------|---------|-------------|
| 0 (Setup) | 1–5 | Git, structure, logging, config, constants |
| 1 (Data) | 6–25 | Data loaders, k-fold CV, splits, labels |
| 2 (Models) | 26–40 | Model loading, generation, inference pipeline |
| 3 (Features) | 36–50 | Core features, advanced features, pipeline |
| 4 (Calibration) | 51–70 | Baseline metrics, calibration methods, abstention, reporting |
| 5 (Visualization) | 71–85 | Figures, reports, galleries |
| 6 (Reproducibility) | 86–95 | Docs, tests, sanity checks, CI |
| 7 (Publication) | 96–100 | Paper outline, figures, README, license |

**Each commit message should be descriptive**:
```
✓ 1.1.1: Load TriviaQA from HF datasets with validation
✓ 1.2.3: Verify no data leakage in k-fold splits
✓ 2.2.5: Inference loop with uncertainty extraction per model/fold
✓ 4.2.6: Fit + evaluate all calibration methods
✓ 5.1.1: Box plots of uncertainty distributions by correctness
✓ 6.1.2: Document data splits with sizes and verification
✓ 7.0.2: Compile publication-quality figures for paper
```

---

## Expected Outputs

### Data & Artifacts
- `results/data/splits_all_datasets.json` (fold definitions)
- `results/data/{model_id}_fold{fold_id}_{dataset}.jsonl` (per-question results, 50+ files)
- `results/data/baseline_metrics.json` (accuracy, hallucination rate)

### Analysis & Metrics
- `results/summaries/final_summary.json` (comprehensive results)
- `results/summaries/{model_id}_summary.json` (per-model breakdown)
- `results/reports/FINDINGS.md` (key findings)
- `results/reports/GENERALIZATION.md` (feature generalization analysis)

### Figures (12–15 PNG files)
- `uncertainty_distributions.png` (feature distributions by correctness)
- `auroc_matrix.png` (models × datasets heatmap)
- `reliability_diagrams.png` (calibration curves)
- `feature_importance.png` (coefficient magnitudes)
- `strategy_comparison.png` (abstention strategies)
- `generalization_heatmap.png` (feature transfer)
- `scaling_curves.png` (model size trends)
- `ece_comparison.png` (calibration methods)
- Plus 5–7 more custom analysis plots

### Logs & Documentation
- `results/logs/run_{timestamp}.log` (per-run logs, verbose)
- `docs/REPRODUCIBILITY.md` (replication instructions)
- `docs/DATA_SPLITS.md` (fold definitions)
- `docs/FEATURE_SPECS.md` (feature documentation)
- `docs/RESULTS_FORMAT.md` (output schema)
- `tests/` (10+ unit tests)

### Code Quality
- 95+ modules with type hints + docstrings
- Configurable via YAML
- Fully modular (easy to reuse components)
- Git history with 100 small, focused commits

---

## Success Criteria

✅ **All criteria below should be "yes"**:
1. [ ] No data leakage (verified by sanity checks)
2. [ ] k-fold CV on all datasets (5 folds per dataset)
3. [ ] 10+ models from 3+ families tested
4. [ ] 5 datasets evaluated (TriviaQA, TruthfulQA, SQuAD, HotpotQA, custom)
5. [ ] AUROC reported per model × dataset (50+ numbers)
6. [ ] Generalization analysis (feature transfer across datasets)
7. [ ] Multiple calibration methods compared
8. [ ] Publication-quality figures + tables
9. [ ] 100 git commits with descriptive messages
10. [ ] Full reproducibility (all code, seeds, splits documented)
11. [ ] Unit tests pass (sanity_checks.py)
12. [ ] README suitable for arXiv

---

## Timeline Estimate

- **Week 1–2**: Phase 0 + Phase 1 (setup, data, splits)
- **Week 3–4**: Phase 2 (model evaluation)
- **Week 4–5**: Phase 3 (features, calibration)
- **Week 5–6**: Phase 4 + Phase 5 (evaluation, visualization)
- **Week 6–7**: Phase 6 (reproducibility, tests)
- **Week 7**: Phase 7 (writeup, publication)

**Total: 7 weeks** (can parallelize Week 2–3 if GPU available)

---

This blueprint is **comprehensive and modular**. Start with Phase 0 (setup) and work through sequentially. Each commit should be small, atomic, and testable. Use the logging system to track progress and debug issues.

**Good luck!** 🚀
