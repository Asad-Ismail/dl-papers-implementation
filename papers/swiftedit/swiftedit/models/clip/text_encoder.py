import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import open_clip
    _OPENCLIP_IMPORT_ERROR = None
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    open_clip = None
    _OPENCLIP_IMPORT_ERROR = e


def _str_dtype_to_torch(dtype_str: Optional[str]) -> torch.dtype:
    if dtype_str is None:
        return torch.float32
    s = str(dtype_str).lower()
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


class CLIPTextEncoder(torch.nn.Module):
    """
    Wrapper around OpenCLIP text encoder.

    Responsibilities:
    - Tokenize input texts
    - Encode text to pooled embedding (default)
    - Manage device/dtype and freezing

    Notes:
    - For stability, text encoder runs in float32 by default regardless of global mixed precision.
    - This wrapper exposes tokenize(...) and encode(...). The encode(...) returns pooled features of shape (B, D).
    - If future components require token-level features, they can be approximated by repeating pooled features per token
      or by extending this class to expose transformer hidden states (not guaranteed in all open_clip versions).
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[str] = "float32",
        freeze: bool = True,
        openclip_dir: Optional[str] = None,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        if open_clip is None:
            msg = "open-clip-torch is required. Please install open-clip-torch~=2.24.0"
            if _OPENCLIP_IMPORT_ERROR:
                msg += f"\nOriginal error: {_OPENCLIP_IMPORT_ERROR}"
            raise ImportError(msg)

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.torch_dtype = _str_dtype_to_torch(dtype)
        # Force float32 for numerical stability by default
        if dtype is None:
            self.torch_dtype = torch.float32
        self.freeze = freeze
        self.max_length = max_length

        # Resolve pretrained spec: allow local weight path if provided under openclip_dir
        pretrained_spec = pretrained
        if openclip_dir is not None and os.path.isdir(openclip_dir):
            # Attempt to find a *.pt weights file under this directory
            candidates = []
            for root, _dirs, files in os.walk(openclip_dir):
                for f in files:
                    if f.endswith(".pt"):
                        candidates.append(os.path.join(root, f))
            if len(candidates) > 0:
                # Prefer the first candidate; users can override by passing explicit path in 'pretrained'
                pretrained_spec = candidates[0]

        # Create model and tokenizer
        # Always create on CPU first to avoid CUDA initialization issues
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_spec,
            device='cpu',
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        # Text model lives under model.transformer/text_projection attrs depending on version.
        # We keep the full model to use encode_text which handles masking & pooling correctly.
        self.clip_model = model
        self.clip_model.eval()
        # Move to target device after creation
        self.clip_model.to(self.device)
        self.tokenizer = tokenizer

        # Freezing
        if self.freeze:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Dtype handling: keep in float32 by default for text
        self.clip_model = self.clip_model.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def tokenize(self, texts: Union[str, List[str]]) -> torch.LongTensor:
        """
        Tokenize a single string or a list of strings to CLIP tokens of shape (B, S).
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts)
        # Ensure max length if tokenizer supports truncation implicitly; open_clip tokenizers default to 77
        return tokens

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        return_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts to pooled CLIP text features.
        Returns dict with keys:
          - pooled: (B, D) pooled text embedding
          - tokens: (B, S) token ids (optional if return_tokens=True)
        """
        single = isinstance(texts, str)
        tokens = self.tokenize(texts)
        tokens = tokens.to(self.device)

        # Always run in float32
        with torch.no_grad():
            # Some open_clip versions accept normalize flag; we normalize explicitly if requested.
            text_features = self.clip_model.encode_text(tokens)
        if normalize:
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        out: Dict[str, torch.Tensor] = {"pooled": text_features}
        if return_tokens:
            out["tokens"] = tokens
        # If single input, keep batch dim but caller can squeeze if desired
        return out

    def forward(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        return self.encode(texts, normalize=normalize, return_tokens=False)["pooled"]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CLIPTextEncoder":
        """
        Factory from a nested config dict like configs/defaults.yaml -> models.clip.text
        Expected keys inside cfg:
          - provider (ignored if not 'openclip')
          - model_name
          - pretrained
          - dtype
          - freeze
          - max_length
        Also optionally parent provides paths.openclip_dir
        """
        provider = cfg.get("provider", "openclip").lower()
        if provider != "openclip":
            raise ValueError(f"Only 'openclip' provider is supported, got: {provider}")
        # Try to retrieve openclip_dir path from a sibling field if passed
        openclip_dir = cfg.get("openclip_dir")
        # Sometimes config nests paths at a top-level; accept 'paths' embedded dict
        if openclip_dir is None and "paths" in cfg:
            openclip_dir = cfg["paths"].get("openclip_dir")
        return cls(
            model_name=cfg.get("model_name", "ViT-L-14"),
            pretrained=cfg.get("pretrained", "laion2b_s32b_b82k"),
            device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "float32"),
            freeze=bool(cfg.get("freeze", True)),
            openclip_dir=openclip_dir,
            max_length=int(cfg.get("max_length", 77)),
        )


def build_text_encoder(config: Dict[str, Any], device: Optional[str] = None) -> CLIPTextEncoder:
    """
    Convenience builder using the top-level config structure; it will attempt to locate
    the nested models.clip.text section and the paths.openclip_dir for local weights.
    """
    text_cfg = (
        config.get("models", {})
        .get("clip", {})
        .get("text", {})
        .copy()
    )
    # Attach openclip_dir path if available
    paths = config.get("paths", {})
    if "openclip_dir" in paths:
        text_cfg["openclip_dir"] = paths["openclip_dir"]
    # Device override
    if device is not None:
        text_cfg["device"] = device
    return CLIPTextEncoder.from_config(text_cfg)
