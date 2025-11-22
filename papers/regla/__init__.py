"""
REGLA: Refining Gated Linear Attention

Package entry and convenience imports.
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PackageNotFoundError

__all__ = [
    "__version__",
    # Core model
    "TransformerConfig",
    "TransformerLM",
    # Training entrypoints
    "train_lm",
    # Eval entrypoints
    "run_ppl_eval",
    "run_bench",
]

try:
    __version__ = _pkg_version("regla")
except _PackageNotFoundError:  # Local dev fallback
    __version__ = "0.0.0"

# Convenience imports
from .core.transformer import TransformerConfig, TransformerLM  # noqa: E402

# Optional training/eval helpers (import lazily to avoid heavy deps on package import)
def train_lm(config_path: str | None = None) -> None:  # type: ignore[override]
    from .train.train_lm import train_lm as _train
    _train(config_path)


def run_ppl_eval(conf_path: str | None = None) -> None:
    from .eval.eval_and_bench import run_ppl_eval as _run
    _run(conf_path)


def run_bench(conf_path: str | None = None) -> None:
    from .eval.eval_and_bench import run_bench as _run
    _run(conf_path)
