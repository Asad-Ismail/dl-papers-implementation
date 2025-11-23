# REGLA: Refining Gated Linear Attention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**REGLA** is a linear-time attention mechanism that combines:
- Safe exponential feature maps with running-max stabilization
- Principled variance-reduction scaling (alpha)
- Refined gating mechanism to alleviate sigmoid saturation
- Strong perplexity while preserving efficiency at long sequence lengths

### Key Features

- **Core Components**: Safe feature maps, refined gates, fast-weight recurrence
- **Multiple Attention Variants**: REGLA, Fast Decay GLA, LA-ELU+1, LA-ReLU, Softmax MHA
- **Flexible Architecture**: Hybrid stacking support (mix Softmax and REGLA layers)
- **Production Ready**: Data loaders for WikiText-103 and SlimPajama with streaming support
- **Comprehensive Evaluation**: PPL metrics, speed/memory benchmarks, lm-evaluation-harness integration
- **Well Tested**: Unit tests for feature maps, recurrence, and gating gradients

## Installation

### Using uv (Recommended - Fast & Modern)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the regla directory
cd papers/regla

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Optional: Set deterministic environment variables
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### Using pip (Alternative)

```bash
# Navigate to the regla directory
cd papers/regla

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Requirements

- **Python**: 3.9 - 3.11 (3.10 recommended)
- **GPU**: NVIDIA A100/V100 or similar for training (16-24GB VRAM sufficient for inference)

## Repository Structure

```
regla/
├── core/                      # Core attention mechanisms
│   ├── feature_maps.py        # Safe exp, ELU+1, ReLU maps; variance scaling
│   ├── gating_and_recurrence.py  # Refined gate; fast-weight recurrence
│   ├── attention.py           # REGLA, Fast Decay, LA baselines, Softmax MHA
│   ├── norms_rope.py          # RMSNorm/LayerNorm and RoPE utilities
│   ├── blocks.py              # Decoder block (prenorm) and MLP
│   └── transformer.py         # Transformer LM with hybrid mapping
├── train/                     # Training pipelines
│   ├── datasets_and_tokenizer.py  # WT103 and SlimPajama loaders
│   ├── train_lm.py            # From-scratch training loop
│   └── post_linearize_and_continual.py  # Post-linearization pipeline
├── eval/                      # Evaluation tools
│   ├── eval_and_bench.py      # PPL evaluation and benchmarks
│   └── harness_wrapper.py     # lm-evaluation-harness interface
├── configs/                   # Configuration files
│   ├── regla_wt103.yaml       # Default training config (WT103)
│   ├── post_linearize_sp.yaml # Continual pretraining on SlimPajama
│   └── pythia_160m.yaml       # Pythia-160M reference dimensions
├── scripts/                   # Convenience scripts
│   ├── run_train_wt.sh        # End-to-end training on WT103
│   ├── run_post_linearize.sh  # Post-linearization workflow
│   ├── run_eval_harness.sh    # Run commonsense evaluations
│   └── run_bench.sh           # Speed/memory benchmarks
└── tests/                     # Unit tests
    ├── test_feature_map_and_variance.py
    └── test_recurrence_and_grad.py
```

## Quick Start

### 1. Run Unit Tests

```bash
pytest tests/
```

### 2. Train from Scratch on WikiText-103

```bash
bash scripts/run_train_wt.sh configs/regla_wt103.yaml
```

**Note**: The script automatically sets deterministic environment variables. Checkpoints and logs are saved to `checkpoints/` and `logs/`.

### 3. Post-Linearization + Continual Pretraining

Convert a pretrained Softmax model to REGLA and continue training:

```bash
bash scripts/run_post_linearize.sh configs/post_linearize_sp.yaml
```

This loads EleutherAI/pythia-160m, replaces ~50% of layers with REGLA, and trains for 50k steps on SlimPajama.

### 4. Evaluate Perplexity and Benchmarks

```bash
# Run speed and memory benchmarks
bash scripts/run_bench.sh configs/regla_wt103.yaml

# Evaluate perplexity programmatically
python -c "from regla.eval.eval_and_bench import evaluate_ppl; evaluate_ppl('path/to/checkpoint')"
```

### 5. Run LM Evaluation Harness

Evaluate on commonsense reasoning tasks:

```bash
# 0-shot evaluation
bash scripts/run_eval_harness.sh configs/post_linearize_sp.yaml \
    boolq,piqa,hellaswag,winogrande,truthfulqa_mc1,truthfulqa_mc2 0

# 5-shot evaluation
bash scripts/run_eval_harness.sh configs/post_linearize_sp.yaml \
    boolq,piqa,hellaswag,winogrande 5
```

## Configuration

### Key Model Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `attn_type` | Attention mechanism type | `"regla"` | `"regla"`, `"softmax"`, `"fast_decay"`, `"la_elu1"`, `"la_relu"`, `"hybrid"` |
| `m` | Feature dimension for linear attention | `64` | Any positive integer |
| `rope` | Enable Rotary Positional Embeddings | `true` | `true`, `false` |
| `alpha_scaling` | Enable variance-reduction scaling | `true` | `true`, `false` |
| `use_sum_norm` | Enable sum normalization | `false` | `true`, `false` |
| `stable_norm` | Post-attention normalization type | `"rmsnorm"` | `"rmsnorm"`, `"layernorm"` |
| `n_heads` | Number of attention heads | `12` | Any positive integer |
| `d_model` | Model dimension | `768` | Any positive integer |
| `n_layers` | Number of transformer layers | `12` | Any positive integer |

### Training Parameters

See `configs/regla_wt103.yaml` for full training configuration including:
- Learning rate and scheduler
- Batch size and gradient accumulation
- Mixed precision training (bf16/fp16)
- Warmup steps and max iterations
- Gradient clipping

## Performance Benchmarks

**WikiText-103 validation perplexity** after 50k steps (12 layers × 768 dim, seq_len 1024-2048):

| Model | Validation PPL | Notes |
|-------|----------------|-------|
| Softmax Attention | 18.2 - 18.8 | Baseline |
| **REGLA (m=64)** | **18.8 - 19.4** | Linear time complexity |
| Fast Decay GLA | 20.3 - 21.2 | Alternative linear attention |
| LA-ELU+1 | > 19.4 | Simpler feature map |
| **Hybrid (50/50)** | **17.6 - 18.4** | Best performance |

*Results are approximate and may vary with hardware, random seed, and hyperparameters.*

### Hybrid Architecture Benefits

Mixing Softmax and REGLA layers often achieves **better perplexity than pure Softmax** while maintaining efficiency:
- Lower layers: Softmax for local context modeling
- Upper layers: REGLA for long-range dependencies
- Result: Strong performance with reduced computational cost

## Usage Examples

### Training from Scratch

```python
from regla import TransformerConfig, TransformerLM, train_lm

# Create model configuration
config = TransformerConfig(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    attn_type="regla",
    m=64,
    rope=True,
)

# Train using config file
train_lm("configs/regla_wt103.yaml")
```

### Inference with Streaming State

```python
import torch
from regla import TransformerConfig, TransformerLM

# Load pretrained model
model = TransformerLM.from_pretrained("path/to/checkpoint")
model.eval()

# Initialize state for streaming
batch_size = 1
state = model.init_state(batch_size, device="cuda")

# Process tokens incrementally
input_ids = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    logits, new_state = model(input_ids, state=state, return_state=True)

# Continue with new state
next_input = torch.tensor([[4]], device="cuda")
logits, new_state = model(next_input, state=new_state, return_state=True)
```

## Troubleshooting

### Training Instability (NaNs)

**Symptoms**: Loss becomes NaN or model outputs explode

**Solutions**:
- ✅ Ensure `alpha_scaling=True` in config
- ✅ Keep `stable_norm="rmsnorm"` enabled for REGLA
- ✅ Use `bf16` mixed precision on A100/H100 GPUs
- ✅ Set gradient clipping to 1.0 (`max_grad_norm: 1.0`)
- ✅ Lower learning rate if issues persist

### Streaming Inference

**Issue**: Need to process long sequences incrementally

**Solution**:
```python
# Initialize state once
state = model.init_state(batch_size, device=device)

# Process tokens one at a time
for token_id in token_sequence:
    input_ids = torch.tensor([[token_id]], device=device)
    logits, state = model(input_ids, state=state, return_state=True)
```

### Dataset Cache Issues

**Symptoms**: Out of disk space or corrupted dataset cache

**Solutions**:
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/

# Or set custom cache directory in config
# training:
#   cache_dir: "/path/to/custom/cache"
```

### GPU Out of Memory

**Solutions**:
- Reduce batch size in config
- Enable gradient checkpointing (if implemented)
- Use gradient accumulation: `gradient_accumulation_steps: 4`
- Reduce `max_seq_len` for training

## Citation

If you use REGLA in your research, please cite:

```bibtex
@article{regla2024,
  title={REGLA: Refining Gated Linear Attention},
  author={REGLA Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This repository is provided for research and educational purposes under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built using deepcode double checked by human