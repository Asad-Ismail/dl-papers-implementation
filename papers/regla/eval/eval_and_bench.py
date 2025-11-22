import os
import time
import json
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from regla.core.transformer import TransformerConfig, TransformerLM
from regla.train.datasets_and_tokenizer import build_tokenizer, get_wt103_dataloaders


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    # Defaults aligned with regla/configs/regla_wt103.yaml
    default = {
        "model": {
            "vocab_size": 50257,
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "d_head": 64,
            "mlp_hidden_dim": 3072,
            "attn_type": "regla",
            "m": 64,
            "rope": True,
            "max_seq_len": 2048,
            "dropout": 0.1,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "stable_norm": "rmsnorm",
            "stable_norm_eps": 1e-5,
            "use_sum_norm": False,
            "alpha_scaling": True,
            "gate_share_across_heads": False,
            "tie_embeddings": True,
            "use_final_norm": True,
            "hybrid_sa_ratio": 0.5,
            "hybrid_pattern": "alternate",
            "hybrid_map": None,
        },
        "evaluation": {
            "dataset": "wikitext-103",
            "seq_len": 2048,
            "batch_size": 8,
            "num_workers": 2,
            "shuffle": False,
            "cache_dir": None,
        },
        "bench": {
            "mixed_precision": "bf16",  # bf16|fp16|fp32
            "prompt_texts": [
                "The theory of attention mechanisms in neural networks suggests that",
                "In a shocking finding, scientists discovered that",
                "Once upon a time in a land far away, there was a",
            ],
            "lengths": [32, 128, 512, 1024],
            "top_k": 0,
            "temperature": 1.0,
        },
    }
    if config_path is None or yaml is None:
        return default
    with open(config_path, "r") as f:
        user = yaml.safe_load(f)
    # shallow merge
    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out
    return merge(default, user or {})


def build_model_cfg(conf: Dict[str, Any], vocab_size: int) -> TransformerConfig:
    m = conf.get("model", {})
    return TransformerConfig(
        vocab_size=vocab_size,
        d_model=m.get("d_model", 768),
        n_layers=m.get("n_layers", 12),
        n_heads=m.get("n_heads", 12),
        d_head=m.get("d_head", 64),
        mlp_hidden_dim=m.get("mlp_hidden_dim", 3072),
        attn_type=m.get("attn_type", "regla"),
        hybrid_sa_ratio=m.get("hybrid_sa_ratio", 0.5),
        hybrid_pattern=m.get("hybrid_pattern", "alternate"),
        hybrid_map=m.get("hybrid_map", None),
        m=m.get("m", 64),
        rope=m.get("rope", True),
        max_seq_len=m.get("max_seq_len", 2048),
        dropout=m.get("dropout", 0.1),
        norm_type=m.get("norm_type", "rmsnorm"),
        norm_eps=m.get("norm_eps", 1e-5),
        stable_norm=m.get("stable_norm", "rmsnorm"),
        stable_norm_eps=m.get("stable_norm_eps", 1e-5),
        use_sum_norm=m.get("use_sum_norm", False),
        alpha_scaling=m.get("alpha_scaling", True),
        gate_share_across_heads=m.get("gate_share_across_heads", False),
        tie_embeddings=m.get("tie_embeddings", True),
        use_final_norm=m.get("use_final_norm", True),
    )


@torch.no_grad()
def evaluate_ppl(model: TransformerLM, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids, state=None, start_pos=0, return_state=False)
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        valid = (labels.view(-1) != -100).sum().item()
        total_nll += loss.item()
        total_tokens += valid
    avg_nll = total_nll / max(1, total_tokens)
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())
    return avg_nll, ppl


def _setup_autocast(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
    elif mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
    else:
        from contextlib import nullcontext
        return nullcontext()


def _prepare_model_and_tokenizer(conf: Dict[str, Any], device: torch.device) -> Tuple[TransformerLM, Any]:
    tokenizer = build_tokenizer(cache_dir=conf.get("evaluation", {}).get("cache_dir", None))
    cfg = build_model_cfg(conf, tokenizer.vocab_size)
    model = TransformerLM(cfg).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_streaming(
    model: TransformerLM,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
    mixed_precision: str = "bf16",
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[List[str], float]:
    # Returns generated texts and avg latency per token in ms
    from contextlib import nullcontext
    autocast_ctx = _setup_autocast(mixed_precision)

    # Initialize states for batch
    batch_size = len(prompts)
    states = model.init_state(batch_size, device=device)

    # Tokenize prompts
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # Prime the model with the prompt sequence
    prompt_len = input_ids.size(1)
    with autocast_ctx:
        logits, states = model(input_ids, state=states, start_pos=0, return_state=True)

    # Generation loop
    generated = input_ids.clone()
    total_time = 0.0
    total_steps = 0
    for t in range(max_new_tokens):
        last_tokens = generated[:, -1:]
        start_pos = prompt_len + t
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with autocast_ctx:
            logits, states = model(last_tokens, state=states, start_pos=start_pos, return_state=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_steps += 1
        next_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / max(1e-5, temperature)
        if top_k and top_k > 0:
            values, indices = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
            mask = torch.full_like(next_logits, float('-inf'))
            next_logits = mask.scatter(-1, indices, values)
        probs = F.softmax(next_logits, dim=-1)
        next_ids = torch.argmax(probs, dim=-1)
        generated = torch.cat([generated, next_ids.unsqueeze(-1)], dim=1)

    # Decode
    texts = [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in generated]
    avg_ms_per_token = (total_time / max(1, total_steps)) * 1000.0
    return texts, avg_ms_per_token


@torch.no_grad()
def benchmark_generation(
    model: TransformerLM,
    tokenizer,
    prompts: List[str],
    lengths: List[int],
    device: torch.device,
    mixed_precision: str = "bf16",
    temperature: float = 1.0,
    top_k: int = 0,
) -> Dict[str, Any]:
    results = {}
    for L in lengths:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        texts, ms_per_token = generate_streaming(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=L,
            device=device,
            mixed_precision=mixed_precision,
            temperature=temperature,
            top_k=top_k,
        )
        peak_mem = None
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem = peak_bytes / (1024 * 1024)
        results[str(L)] = {
            "avg_ms_per_token": ms_per_token,
            "peak_mem_mb": peak_mem,
        }
    return results


def run_ppl_eval(conf_path: Optional[str] = None) -> None:
    set_seed(42)
    conf = load_config(conf_path)
    device = get_device()
    # Build tokenizer and dataloaders
    tokenizer = build_tokenizer(cache_dir=conf.get("evaluation", {}).get("cache_dir", None))
    seq_len = conf.get("evaluation", {}).get("seq_len", 2048)
    batch_size = conf.get("evaluation", {}).get("batch_size", 8)
    num_workers = conf.get("evaluation", {}).get("num_workers", 2)
    shuffle = conf.get("evaluation", {}).get("shuffle", False)
    train_dl, val_dl, test_dl = get_wt103_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        tokenizer=tokenizer,
        num_workers=num_workers,
        shuffle=shuffle,
        cache_dir=conf.get("evaluation", {}).get("cache_dir", None),
    )
    # Build model
    cfg = build_model_cfg(conf, tokenizer.vocab_size)
    model = TransformerLM(cfg).to(device)
    # Evaluate on test
    avg_nll, ppl = evaluate_ppl(model, test_dl, device)
    out = {
        "avg_nll": avg_nll,
        "ppl": ppl,
        "config": asdict(cfg),
    }
    print(json.dumps(out, indent=2))


def run_bench(conf_path: Optional[str] = None) -> None:
    set_seed(42)
    conf = load_config(conf_path)
    device = get_device()
    model, tokenizer = _prepare_model_and_tokenizer(conf, device)
    prompts = conf.get("bench", {}).get("prompt_texts", [])
    lengths = conf.get("bench", {}).get("lengths", [32, 128, 512, 1024])
    mixed_precision = conf.get("bench", {}).get("mixed_precision", "bf16")
    temperature = conf.get("bench", {}).get("temperature", 1.0)
    top_k = conf.get("bench", {}).get("top_k", 0)

    results = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        lengths=lengths,
        device=device,
        mixed_precision=mixed_precision,
        temperature=temperature,
        top_k=top_k,
    )
    out = {
        "bench_results": results,
        "config": asdict(build_model_cfg(conf, tokenizer.vocab_size)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="REGLA Evaluation and Benchmarks")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--mode", type=str, choices=["ppl", "bench"], default="ppl")
    args = parser.parse_args()

    if args.mode == "ppl":
        run_ppl_eval(args.config)
    else:
        run_bench(args.config)
