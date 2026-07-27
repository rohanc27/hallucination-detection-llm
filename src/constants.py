MODEL_REGISTRY = {
    # Qwen family (instruction-tuned)
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    # Pythia family (EleutherAI, base — CPU-friendly sizes)
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    # TinyLlama family (Llama architecture, instruction-tuned, small enough for CPU)
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # SmolLM family (instruction-tuned, very small)
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    # GPT-2 family (base, kept as a non-instruction-tuned baseline)
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
}

DATASET_REGISTRY = {
    "triviaqa": "load_triviaqa",
    "truthfulqa": "load_truthfulqa",
    "squad_v2": "load_squad_v2",
    "hotpotqa": "load_hotpotqa",
    "mmlu": "load_mmlu_sampled",
    "custom": "load_custom_benchmark",
}

LABEL_METHODS = ["substring", "semantic", "mc1"]
FEATURE_NAMES = [
    "avg_entropy",
    "avg_logprob",
    "min_logprob",
    "seq_length",
    "self_confidence",
    "mc1_top_prob",
]
CALIBRATION_METHODS = ["logistic_regression", "temperature_scaling", "isotonic", "platt"]
