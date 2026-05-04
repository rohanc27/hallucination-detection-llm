"""
build_report_assets.py
======================
Generates all figures for the final report from the experimental result JSONs.

Usage:
    python build_report_assets.py

Reads:
    report_assets/triviaqa_summary.json    (primary results, 3 Qwen sizes)
    report_assets/final_summary.json       (secondary results, 5 models on TruthfulQA)

Writes:
    report_assets/figures/fig_scaling.png
    report_assets/figures/fig_tqa_uncertainty.png
    report_assets/figures/fig_tqa_strategy.png
    report_assets/figures/fig_tqa_coefs.png
    report_assets/figures/fig_benchmark_compare.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "report_assets"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# Load both summaries (TriviaQA primary, TruthfulQA secondary)
# ---------------------------------------------------------------
with open(RESULTS_DIR / "triviaqa_summary.json") as f:
    tqa = json.load(f)
with open(RESULTS_DIR / "final_summary.json") as f:
    tfqa = json.load(f)

QWEN_KEYS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b"]
SIZES_M   = [500, 1540, 3000]

# ---------------------------------------------------------------
# Figure 1: Accuracy and AUROC scale with model size (TriviaQA)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

acc_sem = [tqa["baseline_metrics"][k]["semantic_correct"]["accuracy"] for k in QWEN_KEYS]
acc_sub = [tqa["baseline_metrics"][k]["substring_correct"]["accuracy"] for k in QWEN_KEYS]
auroc_sem = [tqa["calibration_head"][k]["semantic"]["auroc_mean"] for k in QWEN_KEYS]
auroc_sem_std = [tqa["calibration_head"][k]["semantic"]["auroc_std"] for k in QWEN_KEYS]
auroc_sub = [tqa["calibration_head"][k]["substring"]["auroc_mean"] for k in QWEN_KEYS]
auroc_sub_std = [tqa["calibration_head"][k]["substring"]["auroc_std"] for k in QWEN_KEYS]

axes[0].plot(SIZES_M, acc_sub, "o-", label="Substring (with aliases)",
             color="#dd8452", linewidth=2, markersize=9)
axes[0].plot(SIZES_M, acc_sem, "s-", label="Semantic similarity",
             color="#4c72b0", linewidth=2, markersize=9)
axes[0].set_xlabel("Model size (M parameters)")
axes[0].set_ylabel("TriviaQA accuracy")
axes[0].set_title("Accuracy scales with model size")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(SIZES_M)
axes[0].set_xticklabels([f"{s}M" for s in SIZES_M])

axes[1].errorbar(SIZES_M, auroc_sub, yerr=auroc_sub_std, fmt="o-",
                 label="Substring (with aliases)", color="#dd8452",
                 linewidth=2, markersize=9, capsize=4)
axes[1].errorbar(SIZES_M, auroc_sem, yerr=auroc_sem_std, fmt="s-",
                 label="Semantic similarity", color="#4c72b0",
                 linewidth=2, markersize=9, capsize=4)
axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="random")
axes[1].set_xlabel("Model size (M parameters)")
axes[1].set_ylabel("Calibration head AUROC (5-fold CV)")
axes[1].set_title("Trained-head AUROC scales with model size")
axes[1].legend(loc="center right")
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(SIZES_M)
axes[1].set_xticklabels([f"{s}M" for s in SIZES_M])
axes[1].set_ylim(0.45, 0.95)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_scaling.png")

# ---------------------------------------------------------------
# Figure 2: Token entropy and probability split by correctness
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
xs    = np.arange(len(QWEN_KEYS))
width = 0.35

h_corr  = [tqa["baseline_metrics"][k]["semantic_correct"]["avg_entropy_correct"] for k in QWEN_KEYS]
h_wrong = [tqa["baseline_metrics"][k]["semantic_correct"]["avg_entropy_wrong"]   for k in QWEN_KEYS]
c_corr  = [tqa["baseline_metrics"][k]["semantic_correct"]["avg_confidence_correct"] for k in QWEN_KEYS]
c_wrong = [tqa["baseline_metrics"][k]["semantic_correct"]["avg_confidence_wrong"]   for k in QWEN_KEYS]

axes[0].bar(xs - width/2, h_corr,  width, label="Correct answers", color="#4c72b0")
axes[0].bar(xs + width/2, h_wrong, width, label="Wrong answers",   color="#dd8452")
axes[0].set_xticks(xs)
axes[0].set_xticklabels(QWEN_KEYS, rotation=10)
axes[0].set_ylabel("Avg token entropy (nats)")
axes[0].set_title("Token entropy: correct vs wrong answers")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(xs - width/2, c_corr,  width, label="Correct answers", color="#4c72b0")
axes[1].bar(xs + width/2, c_wrong, width, label="Wrong answers",   color="#dd8452")
axes[1].set_xticks(xs)
axes[1].set_xticklabels(QWEN_KEYS, rotation=10)
axes[1].set_ylabel("Avg token probability")
axes[1].set_title("Token probability: correct vs wrong answers")
axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

fig.suptitle("Uncertainty signals on TriviaQA: clear separation in expected direction", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_tqa_uncertainty.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_tqa_uncertainty.png")

# ---------------------------------------------------------------
# Figure 3: Strategy comparison at 30% abstention budget
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
strats = ["no_calibration", "entropy_threshold", "self_conf_threshold", "trained_head"]
labels = ["No cal.", "Entropy", "Self-conf", "Trained head"]
colors = ["#888", "#4c72b0", "#dd8452", "#55a868"]
xs     = np.arange(len(QWEN_KEYS))
width  = 0.2

for ax, metric, title in [(axes[0], "cond_accuracy",      "Conditional accuracy on answered (↑ better)"),
                          (axes[1], "hallucination_rate", "Hallucination rate (↓ better)")]:
    for i, (s, l) in enumerate(zip(strats, labels)):
        vals = [tqa["strategy_comparison"][mk][s][metric] for mk in QWEN_KEYS]
        ax.bar(xs + (i - 1.5)*width, vals, width, label=l, color=colors[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(QWEN_KEYS, rotation=10)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].legend(loc="best", fontsize=10)

fig.suptitle("Strategy comparison @ 30% abstention budget (TriviaQA, 1000 questions)", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_tqa_strategy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_tqa_strategy.png")

# ---------------------------------------------------------------
# Figure 4: Calibration head feature coefficients
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
features = ["avg_entropy", "avg_logprob", "min_logprob", "seq_len", "self_confidence"]
xs       = np.arange(len(features))
width    = 0.27

for i, k in enumerate(QWEN_KEYS):
    coefs = tqa["calibration_head"][k]["semantic"]["mean_coef"]
    vals  = [coefs[f] for f in features]
    ax.bar(xs + (i - 1)*width, vals, width, label=k)

ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(features, rotation=15)
ax.set_ylabel("Signed coefficient (predicting correctness)")
ax.set_title("Calibration head: feature importance across Qwen sizes (TriviaQA)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_tqa_coefs.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_tqa_coefs.png")

# ---------------------------------------------------------------
# Figure 5: TriviaQA vs TruthfulQA AUROC comparison
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
xs    = np.arange(len(QWEN_KEYS))
width = 0.35

tqa_auroc     = [tqa["calibration_head"][k]["semantic"]["auroc_mean"] for k in QWEN_KEYS]
tqa_auroc_std = [tqa["calibration_head"][k]["semantic"]["auroc_std"]  for k in QWEN_KEYS]

# TruthfulQA AUROC (only for the two Qwens we ran on both benchmarks)
tfqa_auroc_map = {k: tfqa["calibration_head_results"][k]["auroc"]
                  for k in tfqa["calibration_head_results"]
                  if k.startswith("qwen")}
tfqa_auroc = [tfqa_auroc_map.get("qwen2.5-0.5b", np.nan),
              tfqa_auroc_map.get("qwen2.5-1.5b", np.nan),
              np.nan]   # no Qwen-3B run on TruthfulQA

ax.bar(xs - width/2, tqa_auroc,  width, yerr=tqa_auroc_std,
       label="TriviaQA (general knowledge)", color="#4c72b0", capsize=4)
ax.bar(xs + width/2, tfqa_auroc, width,
       label="TruthfulQA (adversarial)", color="#dd8452")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="random")
ax.set_xticks(xs)
ax.set_xticklabels(QWEN_KEYS, rotation=10)
ax.set_ylabel("Calibration head AUROC")
ax.set_title("Method works on general-knowledge QA, degrades on adversarial benchmarks")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0.45, 0.95)
ax.text(2 + width/2, 0.50, "no run", ha="center", fontsize=9, color="gray")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_benchmark_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_benchmark_compare.png")

print(f"\nAll 5 figures saved to {FIG_DIR}")
