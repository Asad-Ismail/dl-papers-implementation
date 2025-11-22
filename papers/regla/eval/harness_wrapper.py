# Copyright (c) 2025
# Bridge to EleutherAI lm-evaluation-harness for the REGLA TransformerLM
# Provides a minimal LM-compatible wrapper implementing loglikelihood and generate_until.
#
# Usage (programmatic):
#   from regla.eval.harness_wrapper import REGLAHarnessModel
#   lm = REGLAHarnessModel(conf_path="regla/configs/post_linearize_sp.yaml", mixed_precision="bf16")
#   res = lm.loglikelihood([(ctx, cont), ...])
#
# Usage (CLI with lm-eval installed):
#   python -m regla.eval.harness_wrapper --config regla/configs/post_linearize_sp.yaml \
#       --tasks boolq,piqa,hellaswag,winogrande,truthfulqa_mc1,truthfulqa_mc2 --shots 0
#
# Note: This wrapper gracefully degrades if lm_eval is not installed by providing
#       only the model-side utilities; CLI evaluation requires lm_eval.

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    # lm-eval >= 0.4.x
    from lm_eval.api.model import LM as _LMBase  # type: ignore
except Exception:  # pragma: no cover
    class _LMBase:  # minimal shim if harness not installed
        def __init__(self, *args, **kwargs):
            pass

        def greedy_until(self, requests):  # pragma: no cover
            raise NotImplementedError

        def loglikelihood(self, requests):  # pragma: no cover
            raise NotImplementedError

        def loglikelihood_rolling(self, requests):  # pragma: no cover
            raise NotImplementedError


try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers package is required for harness_wrapper; please install transformers"
    ) from e

# Reuse config and model build utilities from eval_and_bench
from regla.core.transformer import TransformerConfig, TransformerLM
from regla.eval.eval_and_bench import build_model_cfg, load_config, set_seed, get_device


def _setup_autocast(mixed_precision: str):
    if mixed_precision == "bf16":
        return torch.cuda.amp.autocast if torch.cuda.is_available() else _nullcontext
    elif mixed_precision == "fp16":
        return torch.cuda.amp.autocast if torch.cuda.is_available() else _nullcontext
    else:
        return _nullcontext


class _nullcontext:  # pragma: no cover
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def top_k_filtering(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)
    return logits


class REGLAHarnessModel(_LMBase):
    def __init__(
        self,
        conf_path: Optional[str] = None,
        *,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        mixed_precision: str = "bf16",
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.conf = load_config(conf_path)
        if cfg_overrides:
            # shallow merge overrides
            for k, v in cfg_overrides.items():
                self.conf.setdefault("model", {})[k] = v
        # tokenizer
        tok_name = self.conf.get("hf", {}).get("tokenizer_name", "gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # build model cfg and model
        model_cfg: TransformerConfig = build_model_cfg(self.conf, vocab_size=len(self.tokenizer))
        self.model = TransformerLM(model_cfg)
        self.model.eval()
        self.device = device or get_device()
        self.model.to(self.device)
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        self.max_seq_len = model_cfg.max_seq_len
        self._eos_id = int(self.tokenizer.eos_token_id)
        set_seed(self.conf.get("training", {}).get("seed", 42))

    # lm-eval interface bits
    @property
    def eot_token_id(self) -> int:  # EOS token id
        return self._eos_id

    @property
    def max_length(self) -> int:
        return self.max_seq_len

    @property
    def tokenizer_encode(self):  # pragma: no cover
        return self.tok_encode

    def tok_encode(self, s: str, add_bos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        if add_bos:
            return [self._eos_id] + ids
        return ids

    def tok_decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    def _prepare_batch_ll(self, requests: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        # Tokenize and bound by max seq len
        encoded: List[Dict[str, Any]] = []
        for context, continuation in requests:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            # ensure we have at least 1 context token to predict first continuation token
            if len(ctx_ids) == 0:
                ctx_ids = [self._eos_id]
            # trim to fit model context window
            total_len = len(ctx_ids) + len(cont_ids)
            max_allowed = self.max_seq_len
            if total_len > max_allowed:
                # keep as much right context as possible
                overflow = total_len - max_allowed
                # drop from the left of context
                ctx_ids = ctx_ids[max(0, overflow) :]
            encoded.append({
                "context_ids": ctx_ids,
                "continuation_ids": cont_ids,
            })
        return encoded

    @torch.no_grad()
    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        # Prepare data
        batch_items = self._prepare_batch_ll(requests)
        results: List[Tuple[float, bool]] = []
        autocast = _setup_autocast(self.mixed_precision)
        # process one by one or in small batches due to variable lengths
        for item in batch_items:
            ctx_ids = item["context_ids"]
            cont_ids = item["continuation_ids"]
            # Build input tensor: context + continuation
            input_ids = torch.tensor([ctx_ids + cont_ids], device=self.device, dtype=torch.long)
            with autocast():
                logits, _ = self.model(input_ids, state=None, start_pos=0, return_state=False)
                # logits shape: (1, L, V)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                # Positions corresponding to continuation labels
                ctx_len = len(ctx_ids)
                total_len = input_ids.size(1)
                cont_len = len(cont_ids)
                # The continuation labels occupy the last cont_len positions of shift_labels
                # index range: (ctx_len) .. (ctx_len + cont_len - 1) in labels space
                start = ctx_len
                end = ctx_len + cont_len
                # gather logprobs
                logprobs = F.log_softmax(shift_logits, dim=-1)
                selected = logprobs[:, start:end, :].gather(-1, shift_labels[:, start:end].unsqueeze(-1)).squeeze(-1)
                ll = selected.sum().item()
                # greedy flag: whether argmax equals target for all continuation tokens
                greedy = (logprobs[:, start:end, :].argmax(dim=-1) == shift_labels[:, start:end]).all().item()
                results.append((float(ll), bool(greedy)))
        return results

    @torch.no_grad()
    def generate_until(self, requests: List[Dict[str, Any]]) -> List[str]:
        # requests: [{"prompt": str, "until": List[str] | None, "max_gen_toks": int | None}]
        outputs: List[str] = []
        autocast = _setup_autocast(self.mixed_precision)
        for req in requests:
            prompt = req.get("prompt", "")
            until: Optional[List[str]] = req.get("until", None)
            max_gen = int(req.get("max_gen_toks", 128))
            temperature = float(req.get("temperature", 1.0))
            top_k = int(req.get("top_k", 0))
            # Tokenize prompt and constrain length
            ctx_ids = self.tok_encode(prompt)
            if len(ctx_ids) == 0:
                ctx_ids = [self._eos_id]
            if len(ctx_ids) > self.max_seq_len - 1:
                ctx_ids = ctx_ids[-(self.max_seq_len - 1) :]
            # Prepare stateful generation
            generated: List[int] = []
            input_ids = torch.tensor([ctx_ids], device=self.device, dtype=torch.long)
            state = None
            start_pos = 0
            with autocast():
                # Prime the model on the prompt
                _, state = self.model(input_ids, state=state, start_pos=start_pos, return_state=True)
                start_pos += input_ids.size(1)
                # Generate tokens
                for _ in range(max_gen):
                    last_token = torch.tensor([[generated[-1] if generated else ctx_ids[-1]]], device=self.device, dtype=torch.long)
                    logits, state = self.model(last_token, state=state, start_pos=start_pos, return_state=True)
                    start_pos += 1
                    logits = logits[:, -1, :]
                    logits = logits / max(temperature, 1e-5)
                    logits = top_k_filtering(logits, top_k=top_k)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1).item()
                    generated.append(int(next_token))
                    # Check stop conditions
                    text = self.tok_decode(generated)
                    if until is not None and any(stop in text for stop in until):
                        break
                    if next_token == self._eos_id:
                        break
            outputs.append(self.tok_decode(generated))
        return outputs


def run_harness_cli():  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Run lm-eval-harness using REGLA model wrapper")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config for model")
    parser.add_argument("--tasks", type=str, default="boolq,piqa,hellaswag,winogrande,truthfulqa_mc1,truthfulqa_mc2")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    try:
        from lm_eval.evaluator import simple_evaluate  # type: ignore
    except Exception as e:
        raise RuntimeError("lm-evaluation-harness is not installed. Please 'pip install lm-eval'.") from e

    lm = REGLAHarnessModel(conf_path=args.config, mixed_precision=args.mixed_precision, batch_size=args.batch_size)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.shots,
        limit=args.limit,
        bootstrap_iters=1000,
    )
    # Print results JSON
    print(json.dumps(results, indent=2, default=lambda o: asdict(o) if isinstance(o, TransformerConfig) else str(o)))


if __name__ == "__main__":  # pragma: no cover
    run_harness_cli()
