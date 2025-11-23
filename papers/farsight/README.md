# FarSight: Mitigating Hallucinations in MLLMs with Attention Causal Decoding

**FarSight** introduces a plug-and-play decoding strategy—composed of Upper-Triangular Attention Registers and Positional Awareness Encoding—within each self-attention layer of a multimodal LLM to absorb outlier attention and enforce a diminishing-rate causal masking, thereby reducing hallucinations without altering the base model weights.

---

## Project Structure

```text
project_root/
  README.md
  requirements.txt
  configs/
    default.yaml       # default hyperparameters and paths
    eval.yaml          # evaluation overrides
  src/                # source code modules
    config.py
    causal_mask.py
    farsight_attention.py
    utils.py
  tests/              # unit tests
```

---

## Installation

### Quick Start with UV (Recommended)

We use [UV](https://github.com/astral-sh/uv) for fast, reliable Python package management.

1. Navigate to the farsight directory:
    ```bash
    cd papers/farsight
    ```

2. Run the setup script (automatically installs UV if needed, creates venv, and installs dependencies):
    ```bash
    ./run.sh
    ```

3. Run the project:
    ```bash
    ./run.sh python run_farsight_llava.py --config configs/default.yaml
    ```

### Manual Installation

If you prefer manual setup:

1. Install UV:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Create virtual environment and install dependencies:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

3. (Optional) Verify GPU availability via PyTorch:
    ```python
    import torch
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    ```

---
