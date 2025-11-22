from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrTexts = Union[torch.Tensor, str, List[str]]


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32", "full"}:
        return torch.float32
    return None


def _to_device(t: torch.Tensor, ref: torch.nn.Module) -> torch.Tensor:
    dev = next(ref.parameters()).device if any(True for _ in ref.parameters()) else getattr(ref, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return t.to(device=dev)


def _ensure_batch(x: torch.Tensor, target_b: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.size(0) == 1 and target_b > 1:
        x = x.expand(target_b, *x.shape[1:])
    return x


def _make_gaussian_1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Kernel size: cover ~3 sigma on each side
    k = int(2 * math.ceil(3 * max(1e-6, sigma)) + 1)
    coords = torch.arange(k, dtype=dtype, device=device) - (k // 2)
    g = torch.exp(-(coords**2) / (2 * sigma * sigma + 1e-12))
    g = g / (g.sum() + 1e-12)
    return g


class MaskExtractor:
    """
    Self-guided mask extractor using the inversion network F_θ.

    Given a latent z and two text conditions (source and edit), it computes:
      ε̂_src = F_θ(z, c_src), ε̂_edit = F_θ(z, c_edit)
      Δ = ε̂_src - ε̂_edit
      m = ||Δ||_2 across channels -> shape (B, H, W)
      per-sample min-max normalization to [0,1]
      optional Gaussian blur and soft-thresholding (continuous)

    Returns a soft mask M of shape (B, 1, H, W) in [0,1].
    """

    def __init__(
        self,
        inversion_net: nn.Module,
        text_encoder: Optional[Any] = None,
        blur_sigma: float = 1.0,
        soft_threshold: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        self.inversion_net = inversion_net
        self.inversion_net.eval()
        self.text_encoder = text_encoder
        self.blur_sigma = float(blur_sigma)
        self.soft_threshold = float(soft_threshold)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.dtype = _map_dtype_str(dtype) or None

        # Prepare Gaussian kernels (created lazily on first use for correct device)
        self._g1: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.inversion_net.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def compute_dtype(self) -> torch.dtype:
        if self.dtype is not None:
            return self.dtype
        # fallback to inversion net param dtype
        try:
            return next(self.inversion_net.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _encode_text_if_needed(self, texts: TensorOrTexts, batch: int) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            return _ensure_batch(_to_device(texts, self.inversion_net), batch)
        if self.text_encoder is None:
            raise ValueError("MaskExtractor requires a text_encoder to encode string prompts.")
        # text_encoder expected to return pooled features (B, D)
        with torch.no_grad():
            pooled = self.text_encoder.forward(texts)
        if not isinstance(pooled, torch.Tensor):
            # some encoders may return dict
            if isinstance(pooled, dict) and "pooled" in pooled:
                pooled = pooled["pooled"]
            else:
                raise RuntimeError("text_encoder.forward returned an unsupported type.")
        pooled = pooled.to(device=self.device)
        return _ensure_batch(pooled, batch)

    def _gaussian_blur(self, m: torch.Tensor) -> torch.Tensor:
        # m: (B, 1, H, W)
        if self.blur_sigma <= 0:
            return m
        dtype = m.dtype
        dev = m.device
        if self._g1 is None or self._g1.device != dev or self._g1.dtype != dtype:
            self._g1 = _make_gaussian_1d(self.blur_sigma, dtype=dtype, device=dev)
        g1 = self._g1
        k = g1.shape[0]
        # separable convolution with reflect padding
        pad = (k // 2, k // 2)
        m = F.pad(m, (0, 0, pad[0], pad[1]), mode="reflect")
        m = F.conv2d(m, g1.view(1, 1, k, 1))
        m = F.pad(m, (pad[0], pad[1], 0, 0), mode="reflect")
        m = F.conv2d(m, g1.view(1, 1, 1, k))
        return m

    @torch.no_grad()
    def compute_mask(
        self,
        z: torch.Tensor,
        c_src: TensorOrTexts,
        c_edit: TensorOrTexts,
        batch_concat: bool = True,
        apply_blur: Optional[bool] = None,
        apply_soft_threshold: Optional[bool] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute self-guided mask.

        Args:
            z: (B, C=4, H, W) latent tensor.
            c_src: source prompt embedding tensor (B,D) or string/list[str].
            c_edit: edit prompt embedding tensor (B,D) or string/list[str].
            batch_concat: if True, run a single batched forward for eps predictions.
            apply_blur: override for Gaussian blur; defaults to True if blur_sigma>0.
            apply_soft_threshold: override for soft threshold application.
            return_intermediate: if True, also return a dict of intermediates.

        Returns:
            M: (B, 1, H, W) tensor in [0,1]
            optionally, dict with keys: eps_src, eps_edit, delta, magnitude, magnitude_norm
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        assert z.dim() == 4, "z must be (B,C,H,W)"
        B, C, H, W = z.shape

        c_src_emb = self._encode_text_if_needed(c_src, B)
        c_edit_emb = self._encode_text_if_needed(c_edit, B)

        # ensure device
        z = z.to(device=self.device, dtype=self.compute_dtype)
        c_src_emb = c_src_emb.to(device=self.device, dtype=torch.float32)
        c_edit_emb = c_edit_emb.to(device=self.device, dtype=torch.float32)

        if batch_concat:
            z_cat = torch.cat([z, z], dim=0)
            c_cat = torch.cat([c_src_emb, c_edit_emb], dim=0)
            eps_cat = self.inversion_net(z_cat, c_cat)
            eps_src, eps_edit = torch.chunk(eps_cat, 2, dim=0)
        else:
            eps_src = self.inversion_net(z, c_src_emb)
            eps_edit = self.inversion_net(z, c_edit_emb)

        delta = eps_src - eps_edit  # (B,C,H,W)
        mag = torch.linalg.norm(delta, ord=2, dim=1, keepdim=False)  # (B,H,W)

        # per-sample min-max normalization
        mag_2d = mag.view(B, -1)
        m_min = mag_2d.min(dim=1).values.view(B, 1, 1)
        m_max = mag_2d.max(dim=1).values.view(B, 1, 1)
        denom = (m_max - m_min).clamp_min(1e-6)
        m_norm = (mag - m_min) / denom
        m_norm = m_norm.clamp(min=self.clamp_min, max=self.clamp_max)

        # reshape to (B,1,H,W)
        M = m_norm.unsqueeze(1)

        if apply_blur is None:
            apply_blur = self.blur_sigma > 0
        if apply_blur:
            M = self._gaussian_blur(M)
            M = M.clamp(min=self.clamp_min, max=self.clamp_max)

        if apply_soft_threshold is None:
            apply_soft_threshold = self.soft_threshold > 0
        if apply_soft_threshold and self.soft_threshold > 0:
            th = float(self.soft_threshold)
            # linear soft-threshold preserving continuity in [0,1]
            M = (M - th) / max(1e-6, (1.0 - th))
            M = M.clamp(min=0.0, max=1.0)

        if return_intermediate:
            return M, {
                "eps_src": eps_src,
                "eps_edit": eps_edit,
                "delta": delta,
                "magnitude": mag,
                "magnitude_norm": m_norm,
            }
        return M


def build_mask_extractor(
    config: Dict[str, Any],
    inversion_net: nn.Module,
    text_encoder: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> MaskExtractor:
    """
    Factory to create MaskExtractor from config and provided inversion_net.

    Reads defaults from:
      config["aram"]["mask"]["blur_sigma"]
      config["aram"]["mask"]["soft_threshold"]
      config["aram"]["mask"]["clamp_min"], ["clamp_max"]
      config["system"]["dtype"]
    """
    aram_mask = (config.get("aram", {}) or {}).get("mask", {}) or {}
    blur_sigma = float(aram_mask.get("blur_sigma", 1.0))
    soft_threshold = float(aram_mask.get("soft_threshold", 0.0))
    clamp_min = float(aram_mask.get("clamp_min", 0.0))
    clamp_max = float(aram_mask.get("clamp_max", 1.0))
    dtype = (config.get("system", {}) or {}).get("dtype", None)

    me = MaskExtractor(
        inversion_net=inversion_net,
        text_encoder=text_encoder,
        blur_sigma=blur_sigma,
        soft_threshold=soft_threshold,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        dtype=dtype,
    )
    # optional device move: inversion_net dictates device; ensure kernels align lazily
    if device is not None:
        # move is handled by inversion_net state; MaskExtractor does not own parameters
        pass
    return me


@torch.no_grad()
def self_guided_mask(
    inversion_net: nn.Module,
    z: torch.Tensor,
    c_src: TensorOrTexts,
    c_edit: TensorOrTexts,
    text_encoder: Optional[Any] = None,
    blur_sigma: float = 1.0,
    soft_threshold: float = 0.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Convenience functional API for self-guided mask extraction.

    Args:
        inversion_net: trained F_θ model mapping (z, c_y) -> ε̂
        z: latent tensor (B,C,H,W)
        c_src: source condition (tensor or string/list[str])
        c_edit: edit condition (tensor or string/list[str])
        text_encoder: optional encoder to convert strings to embeddings
        blur_sigma: Gaussian blur sigma in latent pixels; <=0 disables
        soft_threshold: soft threshold in [0,1]; 0 disables
        clamp_min/clamp_max: clamp range for mask after normalization/blur

    Returns:
        M: soft mask (B,1,H,W) in [0,1]
    """
    extractor = MaskExtractor(
        inversion_net=inversion_net,
        text_encoder=text_encoder,
        blur_sigma=blur_sigma,
        soft_threshold=soft_threshold,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    return extractor.compute_mask(z, c_src, c_edit)


__all__ = [
    "MaskExtractor",
    "build_mask_extractor",
    "self_guided_mask",
]
