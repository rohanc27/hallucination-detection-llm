import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


def build_prompt(question: str, dataset_name: str, model_family: str) -> str:
    """Build a prompt for a question. Varies by dataset + model."""
    
    if 'qwen' in model_family.lower():
        # Qwen uses chat template
        return f"Q: {question}\nA:"
    elif 'tiny' in model_family.lower() or 'smol' in model_family.lower():
        # TinyLlama, SmolLM: few-shot QA prompt
        return (
            "Q: What is the capital of France?\nA: Paris.\n\n"
            "Q: What is 2+2?\nA: 4.\n\n"
            f"Q: {question}\nA:"
        )
    elif 'pythia' in model_family.lower():
        # Pythia: base model, use few-shot
        return (
            "Q: What is the capital of France?\nA: Paris.\n\n"
            f"Q: {question}\nA:"
        )
    else:
        # Default: simple format
        return f"Q: {question}\nA:"


@torch.no_grad()
def generate_with_uncertainty(
    model, tokenizer, prompt: str, max_new_tokens: int = 50, device: str = 'cpu'
) -> dict:
    """
    Generate answer + extract token-level uncertainty.
    
    Returns:
    {
        'text': generated answer,
        'token_logprobs': [lp1, lp2, ...],
        'token_entropies': [h1, h2, ...],
    }
    """
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    input_len = inputs['input_ids'].shape[1]
    
    # Generate with logits
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding
        return_dict_in_generate=True,
        output_scores=True,  # capture token logits
        pad_token_id=tokenizer.eos_token_id,
    )
    
    gen_ids = out.sequences[0, input_len:]
    scores = out.scores  # tuple of logits, one per generated token
    
    if len(scores) == 0:
        return {
            'text': '',
            'token_logprobs': [],
            'token_entropies': [],
        }
    
    token_logprobs = []
    token_entropies = []
    
    for step, logits in enumerate(scores):
        logits = logits[0].float()  # vocab_size
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        
        # Get the actual token that was generated
        token_id = gen_ids[step].item()
        
        # Log-prob of this token
        token_logprobs.append(logp[token_id].item())
        
        # Entropy of the distribution
        entropy = -(p * logp).sum().item()
        token_entropies.append(entropy)
        
        if token_id == tokenizer.eos_token_id:
            break
    
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Truncate at common stop points
    for stop in ['\nQ:', '\n\n', 'Q:', 'A:']:
        if stop in text:
            text = text.split(stop)[0].strip()
            break
    
    return {
        'text': text,
        'token_logprobs': token_logprobs,
        'token_entropies': token_entropies,
    }


def extract_sequence_features(token_logprobs: list, token_entropies: list) -> dict:
    """Aggregate token-level into sequence-level features."""
    if not token_logprobs:
        return {
            'avg_logprob': 0.0,
            'min_logprob': 0.0,
            'avg_entropy': 0.0,
            'seq_length': 0,
        }
    
    return {
        'avg_logprob': float(np.mean(token_logprobs)),
        'min_logprob': float(np.min(token_logprobs)),
        'avg_entropy': float(np.mean(token_entropies)),
        'seq_length': len(token_logprobs),
    }


def load_model_and_tokenizer(model_id: str, device: str = 'cpu'):
    """Load HF model + tokenizer."""
    logger.info(f"Loading {model_id} on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use float32 on CPU, float16 on GPU if available
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device != 'cpu' else None,
        ).eval()
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Loaded {model_id}: {n_params/1e6:.1f}M params")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load {model_id}: {e}")
        raise


def unload_model(model):
    """Free GPU/CPU memory."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded")