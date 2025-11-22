import os
import math
import time
import json
import random
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    import yaml  # for config loading
except Exception:
    yaml = None

from regla.core.transformer import TransformerConfig, TransformerLM
from regla.train.datasets_and_tokenizer import (
    build_tokenizer,
    get_wt103_dataloaders,
)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WarmupScheduler:
    """Linear warmup to base_lr for warmup_steps, then constant."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(int(warmup_steps), 0)
        self.step_id = 0

    def step(self):
        self.step_id += 1
        if self.warmup_steps > 0 and self.step_id <= self.warmup_steps:
            scale = float(self.step_id) / float(self.warmup_steps)
        else:
            scale = 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale


# -----------------------------
# Checkpointing
# -----------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    epoch: int,
    cfg: TransformerConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": asdict(cfg),
        "extra": extra or {},
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    extra = ckpt.get("extra", {})
    return step, epoch, extra


# -----------------------------
# Config loading/mapping
# -----------------------------

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    default = {
        "seed": 42,
        "vocab_size": 50257,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_head": 64,
        "mlp_hidden_dim": 3072,
        "attn_type": "regla",
        "hybrid_sa_ratio": 0.5,
        "hybrid_pattern": "alternate",
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
        # Training hyperparams
        "batch_size": 8,
        "seq_len": 1024,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "grad_clip": 1.0,
        "total_steps": 50000,
        "warmup_steps": 1000,
        "eval_interval": 1000,
        "save_interval": 5000,
        "precision": "bf16",  # options: fp32, fp16, bf16
        "num_workers": 2,
        "output_dir": "outputs/wt103_regla",
        "resume": None,
        "dataset_cache_dir": None,
        "tokenizer_name": "gpt2",
    }
    if config_path is None:
        return default
    if yaml is None:
        print("YAML not available, using defaults.")
        return default
    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return default
    # Merge
    default.update(loaded)
    return default


def build_model_cfg(conf: Dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=int(conf.get("vocab_size", 50257)),
        d_model=int(conf.get("d_model", 768)),
        n_layers=int(conf.get("n_layers", 12)),
        n_heads=int(conf.get("n_heads", 12)),
        d_head=int(conf.get("d_head", 64)),
        mlp_hidden_dim=int(conf.get("mlp_hidden_dim", 3072)),
        attn_type=str(conf.get("attn_type", "regla")),
        hybrid_sa_ratio=float(conf.get("hybrid_sa_ratio", 0.5)),
        hybrid_pattern=str(conf.get("hybrid_pattern", "alternate")),
        m=int(conf.get("m", 64)),
        rope=bool(conf.get("rope", True)),
        max_seq_len=int(conf.get("max_seq_len", 2048)),
        dropout=float(conf.get("dropout", 0.1)),
        norm_type=str(conf.get("norm_type", "rmsnorm")),
        norm_eps=float(conf.get("norm_eps", 1e-5)),
        stable_norm=str(conf.get("stable_norm", "rmsnorm")),
        stable_norm_eps=float(conf.get("stable_norm_eps", 1e-5)),
        use_sum_norm=bool(conf.get("use_sum_norm", False)),
        alpha_scaling=bool(conf.get("alpha_scaling", True)),
        gate_share_across_heads=bool(conf.get("gate_share_across_heads", False)),
        tie_embeddings=bool(conf.get("tie_embeddings", True)),
        use_final_norm=bool(conf.get("use_final_norm", True)),
    )


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(model: TransformerLM, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, state=None, start_pos=0, return_state=False)
            vocab_size = logits.size(-1)
            loss = ce(logits.view(-1, vocab_size), labels.view(-1))
            n_valid = (labels.view(-1) != -100).sum().item()
            total_loss += loss.item() * n_valid
            total_tokens += n_valid
    avg_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return avg_nll, ppl


# -----------------------------
# Training
# -----------------------------

def train_lm(config_path: Optional[str] = None) -> None:
    conf = load_config(config_path)
    set_seed(int(conf.get("seed", 42)))

    device = get_device()
    print(f"Using device: {device}")

    # Tokenizer and data
    tokenizer = build_tokenizer(conf.get("tokenizer_name", "gpt2"), cache_dir=conf.get("dataset_cache_dir"))
    seq_len = int(conf.get("seq_len", 1024))
    batch_size = int(conf.get("batch_size", 8))
    train_loader, val_loader, test_loader = get_wt103_dataloaders(
        seq_len=seq_len,
        batch_size=batch_size,
        tokenizer=tokenizer,
        num_workers=int(conf.get("num_workers", 2)),
        shuffle=True,
        cache_dir=conf.get("dataset_cache_dir"),
    )

    # Build model
    model_cfg = build_model_cfg(conf)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = TransformerLM(model_cfg)
    model.to(device)

    # Optimizer and mixed precision
    lr = float(conf.get("lr", 2e-4))
    wd = float(conf.get("weight_decay", 0.01))
    betas = tuple(conf.get("betas", (0.9, 0.999)))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    scheduler = WarmupScheduler(optimizer, base_lr=lr, warmup_steps=int(conf.get("warmup_steps", 0)))

    precision = str(conf.get("precision", "bf16")).lower()
    use_amp = precision in ("fp16", "bf16") and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and use_amp))
    autocast_dtype = None
    if use_amp:
        if precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif precision == "fp16":
            autocast_dtype = torch.float16
    grad_clip = float(conf.get("grad_clip", 1.0))

    # Resume
    step_start = 0
    epoch_start = 0
    resume_path = conf.get("resume")
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        s, e, extra = load_checkpoint(resume_path, model, optimizer, scaler)
        step_start = s
        epoch_start = e
        print(f"Resumed at step {s}, epoch {e}")

    # Training loop
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = int(conf.get("total_steps", 50000))
    eval_interval = int(conf.get("eval_interval", 1000))
    save_interval = int(conf.get("save_interval", 5000))
    out_dir = conf.get("output_dir", "outputs/wt103_regla")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    model.train()
    step = step_start
    epoch = epoch_start

    # Iterate over epochs until reaching total_steps
    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # Forward
            if use_amp and autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    logits, _ = model(input_ids, state=None, start_pos=0, return_state=False)
                    vocab_size = logits.size(-1)
                    loss = ce(logits.view(-1, vocab_size), labels.view(-1))
            else:
                logits, _ = model(input_ids, state=None, start_pos=0, return_state=False)
                vocab_size = logits.size(-1)
                loss = ce(logits.view(-1, vocab_size), labels.view(-1))

            # Backward
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            if step % 50 == 0 or step == 1:
                n_valid = (labels.view(-1) != -100).sum().item()
                avg_loss = loss.item()
                print(f"step {step}/{total_steps} | loss {avg_loss:.4f} | tokens {n_valid}")
                with open(log_path, "a") as lf:
                    lf.write(json.dumps({"step": step, "loss": avg_loss}) + "\n")

            # Eval
            if step % eval_interval == 0:
                val_nll, val_ppl = evaluate(model, val_loader, device)
                print(f"[val] step {step} | nll {val_nll:.4f} | ppl {val_ppl:.3f}")
                with open(log_path, "a") as lf:
                    lf.write(
                        json.dumps({
                            "step": step,
                            "val_nll": val_nll,
                            "val_ppl": val_ppl,
                        })
                        + "\n"
                    )

            # Save
            if step % save_interval == 0:
                save_checkpoint(ckpt_path, model, optimizer, scaler, step, epoch, model_cfg)
                print(f"Saved checkpoint to {ckpt_path}")

            if step >= total_steps:
                break

    # Final eval on validation and test
    val_nll, val_ppl = evaluate(model, val_loader, device)
    test_nll, test_ppl = evaluate(model, test_loader, device)
    print(f"Final validation ppl: {val_ppl:.3f} | test ppl: {test_ppl:.3f}")
    metrics = {
        "final_val_nll": val_nll,
        "final_val_ppl": val_ppl,
        "final_test_nll": test_nll,
        "final_test_ppl": test_ppl,
        "steps": total_steps,
        "epoch": epoch,
    }
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save final checkpoint
    save_checkpoint(ckpt_path, model, optimizer, scaler, step, epoch, model_cfg, extra={"final": True})
    print(f"Training complete. Checkpoint and metrics saved to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer LM with REGLA and baselines on WikiText-103")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()
    train_lm(args.config)
