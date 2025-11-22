import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:
    yaml = None

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from regla.core.transformer import TransformerConfig, TransformerLM
from regla.train.datasets_and_tokenizer import (
    get_slimpajama_stream_loader,
)


# ---------------------- Utilities ----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Config handling ----------------------
def load_config(config_path: Optional[str]) -> Dict:
    cfg = {
        "model": {
            "vocab_size": 50257,
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "d_head": 64,
            "mlp_hidden_dim": 3072,
            "attn_type": "hybrid",  # default to hybrid for post-linearization
            "hybrid_sa_ratio": 0.5,
            "hybrid_pattern": "alternate",
            "hybrid_map": None,
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
        },
        "training": {
            "seq_len": 2048,
            "batch_size": 8,
            "num_workers": 0,
            "seed": 42,
            "steps": 50000,
            "warmup_steps": 1000,
            "lr": 2e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "grad_clip": 1.0,
            "mixed_precision": "bf16",  # bf16 preferred
            "log_interval": 100,
            "save_interval": 1000,
            "eval_interval": 0,
            "out_dir": "checkpoints/post_linearize_pythia160m",
            "jsonl_path": "logs/post_linearize_pythia160m.jsonl",
            "cache_dir": None,
        },
        "hf": {
            "model_name": "EleutherAI/pythia-160m",
            "tokenizer_name": None,  # defaults to model_name
            "use_auth_token": False,
        },
    }
    if config_path is not None and yaml is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f)
        # shallow merge
        for k in loaded:
            if k in cfg and isinstance(cfg[k], dict) and isinstance(loaded[k], dict):
                cfg[k].update(loaded[k])
            else:
                cfg[k] = loaded[k]
    return cfg


def build_model_cfg(conf: Dict, vocab_size: int) -> TransformerConfig:
    m = conf["model"].copy()
    m["vocab_size"] = vocab_size
    # Ensure hybrid selection is honored
    return TransformerConfig(
        vocab_size=m.get("vocab_size"),
        d_model=m.get("d_model"),
        n_layers=m.get("n_layers"),
        n_heads=m.get("n_heads"),
        d_head=m.get("d_head"),
        mlp_hidden_dim=m.get("mlp_hidden_dim"),
        attn_type=m.get("attn_type", "hybrid"),
        hybrid_sa_ratio=m.get("hybrid_sa_ratio", 0.5),
        hybrid_pattern=m.get("hybrid_pattern", "alternate"),
        hybrid_map=m.get("hybrid_map"),
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


# ---------------------- HF model loading ----------------------
def load_hf_pythia(model_name: str, tokenizer_name: Optional[str] = None, use_auth_token: bool = False):
    tok_name = tokenizer_name or model_name
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.eval()
    return hf_model, tokenizer


# ---------------------- Hybrid mapping ----------------------
def select_hybrid_layers(n_layers: int, ratio: float = 0.5, pattern: str = "alternate", seed: int = 42) -> List[str]:
    """Return a list of length n_layers with entries in {"sa", "regla"}."""
    num_sa = int(round(n_layers * ratio))
    kinds = ["regla"] * n_layers
    if pattern == "alternate":
        # Fill every other with SA until count met
        idxs = list(range(0, n_layers, 2)) + list(range(1, n_layers, 2))
        for i in idxs[:num_sa]:
            kinds[i] = "sa"
    elif pattern == "first_sa":
        for i in range(num_sa):
            kinds[i] = "sa"
    elif pattern == "last_sa":
        for i in range(n_layers - num_sa, n_layers):
            kinds[i] = "sa"
    else:
        random.seed(seed)
        idxs = list(range(n_layers))
        random.shuffle(idxs)
        for i in idxs[:num_sa]:
            kinds[i] = "sa"
    return kinds


# ---------------------- Weight transfer ----------------------
def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)
        elif dst.bias is not None and src.bias is None:
            dst.bias.zero_()


def transfer_pythia_weights(local: TransformerLM, hf_model, layer_kinds: List[str]) -> None:
    """Transfer weights from HF GPT-NeoX (Pythia) into our local model.
    Assumes matching dims (d_model, n_layers, n_heads, d_head, mlp dims).
    """
    sd = hf_model.state_dict()
    # Embeddings
    try:
        emb_w = sd["gpt_neox.embed_in.weight"]
        with torch.no_grad():
            local.tok_emb.weight[: emb_w.shape[0]].copy_(emb_w)
    except KeyError:
        pass
    # Final norm and lm head
    try:
        fn_w = sd["gpt_neox.final_layer_norm.weight"]
        if hasattr(local, "ln_f") and local.ln_f is not None:
            with torch.no_grad():
                local.ln_f.weight.copy_(fn_w)
    except KeyError:
        pass
    try:
        out_w = sd["embed_out.weight"]
        with torch.no_grad():
            local.lm_head.weight[: out_w.shape[0]].copy_(out_w)
    except KeyError:
        pass

    n_layers = len(local.blocks)
    for i in range(n_layers):
        # Norms
        try:
            in_ln_w = sd[f"gpt_neox.layers.{i}.input_layernorm.weight"]
            post_ln_w = sd[f"gpt_neox.layers.{i}.post_attention_layernorm.weight"]
            with torch.no_grad():
                if hasattr(local.blocks[i].norm1, "weight"):
                    local.blocks[i].norm1.weight.copy_(in_ln_w)
                if hasattr(local.blocks[i].norm2, "weight"):
                    local.blocks[i].norm2.weight.copy_(post_ln_w)
        except KeyError:
            pass
        # MLP
        try:
            fc1_w = sd[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"]
            fc1_b = sd[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"]
            fc2_w = sd[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"]
            fc2_b = sd[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"]
            with torch.no_grad():
                local.blocks[i].mlp.fc1.weight.copy_(fc1_w)
                local.blocks[i].mlp.fc1.bias.copy_(fc1_b)
                local.blocks[i].mlp.fc2.weight.copy_(fc2_w)
                local.blocks[i].mlp.fc2.bias.copy_(fc2_b)
        except KeyError:
            pass
        # Attention projections
        try:
            qkv_w = sd[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
            qkv_b = sd[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
            out_w = sd[f"gpt_neox.layers.{i}.attention.dense.weight"]
            out_b = sd[f"gpt_neox.layers.{i}.attention.dense.bias"]
            # Split qkv
            qkv_chunks_w = torch.chunk(qkv_w, 3, dim=0)
            qkv_chunks_b = torch.chunk(qkv_b, 3, dim=0)
            attn = local.blocks[i].attn
            with torch.no_grad():
                if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                    attn.q_proj.weight.copy_(qkv_chunks_w[0])
                    attn.k_proj.weight.copy_(qkv_chunks_w[1])
                    attn.v_proj.weight.copy_(qkv_chunks_w[2])
                    if getattr(attn.q_proj, "bias", None) is not None:
                        attn.q_proj.bias.copy_(qkv_chunks_b[0])
                        attn.k_proj.bias.copy_(qkv_chunks_b[1])
                        attn.v_proj.bias.copy_(qkv_chunks_b[2])
                if hasattr(attn, "out_proj"):
                    attn.out_proj.weight.copy_(out_w)
                    if getattr(attn.out_proj, "bias", None) is not None:
                        attn.out_proj.bias.copy_(out_b)
        except KeyError:
            pass


# ---------------------- Training ----------------------
class WarmupScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * float(self.step_num) / float(self.warmup_steps)
        else:
            lr = self.base_lr
        for pg in self.opt.param_groups:
            pg["lr"] = lr


def continual_pretrain(cfg: Dict) -> None:
    device = get_device()
    set_seed(cfg["training"]["seed"])

    hf_model, tokenizer = load_hf_pythia(
        cfg["hf"]["model_name"], cfg["hf"].get("tokenizer_name"), cfg["hf"].get("use_auth_token", False)
    )

    # Build local model
    model_cfg = build_model_cfg(cfg, vocab_size=tokenizer.vocab_size)
    model = TransformerLM(model_cfg).to(device)

    # Build hybrid mapping and transfer weights
    kinds = select_hybrid_layers(
        model_cfg.n_layers, ratio=model_cfg.hybrid_sa_ratio, pattern=model_cfg.hybrid_pattern, seed=cfg["training"]["seed"]
    )
    # Transfer weights (works for both SA and REGLA in our implementation)
    transfer_pythia_weights(model, hf_model, kinds)

    # DataLoader for SlimPajama streaming
    train_loader: DataLoader = get_slimpajama_stream_loader(
        seq_len=cfg["training"]["seq_len"],
        batch_size=cfg["training"]["batch_size"],
        tokenizer=tokenizer,
        num_workers=cfg["training"].get("num_workers", 0),
        shuffle=False,
        subset_fraction=None,
        cache_dir=cfg["training"].get("cache_dir"),
    )
    data_iter = iter(train_loader)

    # Optimizer
    betas = tuple(cfg["training"].get("betas", [0.9, 0.999]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], betas=betas, weight_decay=cfg["training"]["weight_decay"])
    scheduler = WarmupScheduler(optimizer, cfg["training"]["lr"], cfg["training"]["warmup_steps"])

    # Mixed precision
    mp = cfg["training"].get("mixed_precision", "bf16").lower()
    use_scaler = False
    scaler = None
    amp_dtype = torch.float32
    if device.type == "cuda":
        if mp == "bf16":
            amp_dtype = torch.bfloat16
        elif mp == "fp16":
            amp_dtype = torch.float16
            use_scaler = True
            scaler = torch.cuda.amp.GradScaler()
        else:
            amp_dtype = torch.float32

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    os.makedirs(cfg["training"]["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["training"]["jsonl_path"]), exist_ok=True)
    log_f = open(cfg["training"]["jsonl_path"], "a", buffering=1)

    model.train()
    step = 0
    while step < cfg["training"]["steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        with torch.cuda.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            logits, _ = model(input_ids, state=None, start_pos=0, return_state=False)
            loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        if use_scaler and scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

        step += 1
        if step % cfg["training"]["log_interval"] == 0:
            record = {"step": step, "loss": float(loss.detach().cpu()), "lr": optimizer.param_groups[0]["lr"]}
            log_f.write(json.dumps(record) + "\n")
            print(f"Step {step} loss {record['loss']:.4f} lr {record['lr']:.6f}")

        if step % cfg["training"]["save_interval"] == 0:
            ckpt_path = os.path.join(cfg["training"]["out_dir"], f"step_{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, model_cfg)

    # Final save
    ckpt_path = os.path.join(cfg["training"]["out_dir"], f"final_step_{step}.pt")
    save_checkpoint(ckpt_path, model, optimizer, scaler, step, model_cfg)
    log_f.close()


def save_checkpoint(path: str, model: TransformerLM, optimizer: torch.optim.Optimizer, scaler, step: int, cfg: TransformerConfig):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": asdict(cfg),
    }
    torch.save(payload, path)


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description="Post-linearize Pythia-160M and continual pretrain with REGLA hybrid")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    continual_pretrain(cfg)


if __name__ == "__main__":
    main()
