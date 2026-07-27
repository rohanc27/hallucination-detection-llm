# Hallucination Detection Project — Audit Report
**vs. Multi-Model, Multi-Dataset Generalization Blueprint**

---

## Executive Summary

Your current project is **course-work quality** but **not yet research-ready**. The core issue: you're claiming AUROC 0.85 on TriviaQA, but the evaluation is done on a single 70/30 split of **the same dataset you're training on**. This is data leakage. The blueprint requires proper train/test isolation + validation on independent datasets.

**Gap severity:**
- 🔴 **Critical**: Train/test contamination (no k-fold CV, single dataset split)
- 🔴 **Critical**: Only 1 model family tested (3 Qwen sizes), not 10+ models
- 🟠 **High**: Only 2 datasets (TriviaQA + TruthfulQA), need 5+
- 🟠 **High**: No documented data splits / pipeline documentation
- 🟡 **Medium**: Features are reasonable but could be enhanced (token-level variance, attention entropy missing)

---

## Phase-by-Phase Audit

### Phase 1: Data & Methodology ✅ PARTIAL

**Goal**: Implement clean k-fold CV with no contamination. Collect 5+ datasets with documented splits.

**Current state:**
- ✅ TriviaQA loaded (1000 questions sampled)
- ✅ TruthfulQA loaded (used for stress-testing)
- ✅ Two labeling methods (substring + semantic similarity)
- ❌ **NO k-fold CV**: Uses single 70/30 split on combined data
- ❌ Only 2 datasets (need 5+): TriviaQA, TruthfulQA. Missing SQuAD, HotpotQA, custom benchmark.
- ❌ **Data leakage**: All 4 models evaluated on same TriviaQA+TruthfulQA split → reported AUROC inflated

**What's broken:**
```python
# CURRENT (problematic):
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
# This is applied AFTER all features are extracted from the SAME dataset.
# Every fold sees data from both "train" and "test".
```

**What you need:**
```
1. Implement k-fold CV (5 folds minimum):
   - Fold 1-4: train calibration head
   - Fold 5: HELD OUT TEST (never touch until final report)
   
2. Each fold MUST be isolated at data collection time:
   - Split dataset → generate answers separately → extract features → train/test

3. Add 3 more datasets:
   - SQuAD: span-based QA (different from open-ended)
   - HotpotQA: multi-hop reasoning (harder)
   - Custom benchmark: questions you manually verify (ground truth)
```

**Recommendation**: Fix this first. It's the foundation for everything else.

---

### Phase 2: Multi-Model Testing ❌ NOT STARTED

**Goal**: Test on 10+ LLMs across 3+ families.

**Current state:**
- Only 4 models: distilgpt2, gpt2, gpt2-medium, Qwen2.5-0.5B-Instruct
  - ❌ distilgpt2, gpt2, gpt2-medium are **base models**, not instruction-tuned → poor for QA
  - ✅ Only Qwen2.5-0.5B has been used in results (README shows 0.5B/1.5B/3B)
  - ❌ But 1.5B and 3B are missing from config — unclear if they were run

**What you need:**
| Family | Models to test |
|--------|---|
| Qwen | 0.5B, 1.5B, 3B (you have config for 0.5B only) |
| Llama | 1B, 8B, 70B (Meta, huggingface) |
| Mistral | 7B, 7B-Instruct |
| Phi | 3.8B, 3.8B-mini |
| OpenAI | GPT-2 small is okay, but consider a 125M/355M baseline |
| Gemma | Open Gemma (Google, free) 2B or 7B |

**Recommendation**: Update `MODELS` dict to include all three Qwen sizes + at least 2 from Llama/Mistral. Run in batches to manage VRAM.

---

### Phase 3: Better Uncertainty Signals 🟡 PARTIAL

**Goal**: Add token-level variance, embedding similarity, attention entropy, prediction confidence by position.

**Current features (6):**
- ✅ avg_entropy
- ✅ avg_logprob
- ✅ min_logprob
- ✅ seq_len
- ✅ self_confidence (prompt-based)
- ✅ mc1_top_prob (MC1 confidence)

**Missing features:**
- ❌ Token-level variance (per-token entropy variance across sequence)
- ❌ Embedding similarity to correct answer (how close is generated embedding to reference?)
- ❌ Attention entropy (if you extract attention heads)
- ❌ Prediction confidence by token position (first token more/less confident than last?)
- ❌ Repetition penalty (does the model repeat itself = hallucination indicator?)

**Recommendation**: These are "nice to have" for phase 3. Defer until after Phase 1 + Phase 2 are solid.

---

### Phase 4: Better Calibration 🟡 PARTIAL

**Goal**: Try temperature scaling, isotonic regression, Platt scaling. Measure ECE, MCE, Brier.

**Current state:**
- ✅ Logistic regression with `class_weight='balanced'`
- ✅ Brier score computed
- ✅ ECE computed (reliability diagrams generated)
- ❌ NO temperature scaling tried
- ❌ NO isotonic regression tried
- ❌ NO Platt scaling tried
- ❌ NO ensemble of calibration methods

**Recommendation**: Defer until after multi-model testing. Once you have 10+ models × 5 folds, you can compare calibration methods systematically.

---

## Detailed Code Issues

### Issue 1: Train/Test Contamination (CRITICAL)

**Location**: `hallucination_detection.ipynb`, Section 10

**Problem**: All models are evaluated on the same data split:
```python
for model_key, results in all_results.items():
    df = pd.DataFrame(results)
    X = df[FEATURES].values
    y = df['semantic_correct'].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
```

Every model's results go through the same 70/30 split. This means:
- You're not measuring generalization across different questions
- AUROC 0.85 may be inflated due to data artifacts
- You can't claim "verified 0.85+ across all models/datasets"

**Fix**:
1. Implement k-fold CV at the **dataset level**, not the row level
2. For each model:
   - Fold 1: use questions 1-200 for training, 201-250 for test
   - Fold 2: use questions 201-450 for training, 451-500 for test
   - ...etc
3. Report mean + std across folds

---

### Issue 2: Only One Model Family (CRITICAL)

**Location**: `hallucination_detection.ipynb`, Section 1

**Problem**: Config has 4 models, but only 1 instruction-tuned model is used:
```python
MODELS = {
    'distilgpt2':       'distilgpt2',
    'gpt2':             'gpt2',
    'gpt2-medium':      'gpt2-medium',
    'qwen2.5-0.5b':     'Qwen/Qwen2.5-0.5B-Instruct',
}
```

Base GPT-2 models are not instruction-tuned → poor at QA. README claims Qwen 0.5B/1.5B/3B, but config only has 0.5B.

**Fix**:
```python
MODELS = {
    # Qwen family (instruction-tuned)
    'qwen2.5-0.5b':     'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b':     'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b':       'Qwen/Qwen2.5-3B-Instruct',
    
    # Llama family
    'llama2-1b':        'meta-llama/Llama-2-1b-hf',  # or any 1B variant
    'llama2-7b':        'meta-llama/Llama-2-7b-hf',
    
    # Mistral family
    'mistral-7b':       'mistralai/Mistral-7B-Instruct-v0.1',
    
    # Phi family (lightweight)
    'phi-3.8b':         'microsoft/phi-2',  # or Phi-3
}
```

---

### Issue 3: Missing Independent Datasets (CRITICAL)

**Location**: Data loading section

**Problem**: Only TriviaQA + TruthfulQA. These are not independent:
- TriviaQA: open-ended, factual questions
- TruthfulQA: deliberately adversarial (designed to elicit hallucinations)

No diversity in question types.

**Fix**: Add:
1. **SQuAD v2**: Machine reading comprehension (extractive + unanswerable questions)
2. **HotpotQA**: Multi-hop reasoning (requires combining facts)
3. **MMLU (sampled)**: Multiple-choice covering 57 domains
4. **Natural Questions**: Google's web questions
5. **Custom benchmark**: 100 questions you manually verify (prevents label noise)

---

### Issue 4: Undocumented Data Splits

**Location**: README + notebook

**Problem**: No clear documentation of which questions go where. Hard to reproduce.

**Fix**: Add to repo:
```
data/
├── splits.json          # {fold_id: {train: [q1, q2, ...], test: [q_n, ...]}}
├── triviaqa_split.pkl
├── hotpotqa_split.pkl
├── etc.
└── README.md            # "Splits are fixed by seed=42. See splits.json for fold assignments."
```

---

## Roadmap to Fix (Priority Order)

### Week 1: Fix Data Pipeline (Phase 1)
- [ ] Implement 5-fold CV with documented splits
- [ ] Verify no train/test leakage
- [ ] Re-run Qwen 0.5B on clean splits, report mean±std across folds
- [ ] Start loading SQuAD + HotpotQA (don't need to evaluate yet, just integration test)

### Week 2: Multi-Model Testing (Phase 2)
- [ ] Add Llama 1B + 7B to config
- [ ] Add Mistral 7B-Instruct to config
- [ ] Run all Qwen sizes (0.5B, 1.5B, 3B) on folded data
- [ ] Generate per-model results JSON
- [ ] Create generalization table: AUROC by model, show which features transfer

### Week 3: Multi-Dataset Evaluation (Phase 1 + 2 combined)
- [ ] Finish loading all 5 datasets
- [ ] Run 1-2 models (Qwen-3B + Llama-7B) across all 5 datasets, all 5 folds
- [ ] Report: AUROC on TriviaQA, SQuAD, HotpotQA separately
- [ ] Identify which signals generalize vs. which are dataset-specific

### Week 4: Finalization
- [ ] Add feature engineering (token variance, attention entropy if time)
- [ ] Compare calibration methods (temperature scaling, isotonic)
- [ ] Write clean code with proper logging
- [ ] Generate final report: "Hallucination detection generalizes across 10 models and 5 datasets"

---

## Red Flags from README

The README claims:
> "AUROC 0.85–0.89 for correctness prediction across three Qwen2.5 sizes, with **no LM fine-tuning**."

**But current state:**
- ✅ No LM fine-tuning (just a logistic regression head)
- ❌ **Only evaluated on single 70/30 split of same dataset**
- ❌ No claimed generalization to other datasets/models yet

**Recommendation**: Update README to say:
- "Early results on TriviaQA (single 70/30 split): AUROC 0.85–0.89"
- "Full multi-dataset, multi-model evaluation in progress"
- "**Note**: These are preliminary. Final paper will report k-fold CV + cross-dataset generalization."

---

## Summary Table

| Phase | Criterion | Status | Gap |
|-------|-----------|--------|-----|
| **Phase 1** | K-fold CV + no leakage | ❌ | Single 70/30 split |
| **Phase 1** | 5+ datasets | ❌ | 2 datasets (need SQuAD, HotpotQA, etc.) |
| **Phase 1** | Documented splits | ❌ | No splits.json |
| **Phase 2** | 10+ models | ❌ | 4 models, mostly base (need LLM family diversity) |
| **Phase 2** | 3+ families | ❌ | 1 family (only Qwen, need Llama/Mistral/Phi) |
| **Phase 3** | Enhanced features | ❌ | 6 features, missing token variance, attention entropy |
| **Phase 4** | Calibration comparison | 🟡 | Only LogReg, no temp scaling / isotonic |
| **Overall** | Generalization claim | ❌ | Can't claim it yet (single dataset, single family) |

---

## Next Action Items

1. **THIS WEEK**: Fix Phase 1 — implement k-fold CV and split documentation
2. **NEXT WEEK**: Expand Phase 2 — add Llama, Mistral to model roster
3. **WEEK 3**: Add datasets — SQuAD, HotpotQA integration
4. **WEEK 4**: Measure generalization — report AUROC by model + dataset
5. **WEEK 5**: Finalize — calibration methods, clean code, paper write-up

---

## Questions for You

1. **Timeline**: When do you want this ready? (For publication? Resume portfolio? Class project?)
2. **Compute budget**: How much GPU time can you afford? (VRAM, cloud cost)
3. **Data priority**: Which datasets are most important to you?
4. **Models to focus on**: Any specific models you want to test first?

Answers will help prioritize the roadmap.
