"""Hi-SAM text-stroke segmentation.

Wraps the vendored Hi-SAM code in third_party/Hi-SAM. Two things are handled
here that the upstream demo does not:

  * checkpoints are loaded from explicit paths (upstream hardcodes a
    cwd-relative "pretrained_checkpoint/" directory inside build.py), and
  * regions wider than the encoder's 1024px input are tiled with tapered
    blending, so a full-width subtitle keeps its native stroke detail instead
    of being downscaled to fit.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

_HI_SAM_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "Hi-SAM")


def _ensure_import_path() -> None:
    if not os.path.isdir(_HI_SAM_ROOT):
        raise RuntimeError(
            f"Hi-SAM source not found at {_HI_SAM_ROOT}.\n"
            "Fetch it with:  git clone https://github.com/ymy-k/Hi-SAM.git third_party/Hi-SAM"
        )
    if _HI_SAM_ROOT not in sys.path:
        sys.path.insert(0, _HI_SAM_ROOT)


# Encoder geometry per SAM backbone size.
_VIT_CONFIG = {
    "vit_b": dict(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11]),
    "vit_l": dict(embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23]),
    "vit_h": dict(embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31]),
}

# Filenames of the base SAM checkpoints, used to locate them inside --models-dir.
SAM_CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def build_hisam(
    hisam_checkpoint: str,
    sam_checkpoint: str,
    model_type: str = "vit_l",
    device: str = "cuda",
    attn_layers: int = 1,
    prompt_len: int = 12,
    verbose: bool = False,
):
    """Assemble a Hi-SAM text-stroke model from explicit checkpoint paths.

    The released `sam_tss_*.pth` checkpoints hold only the trained parts (the
    encoder adapters, the modal aligner and the mask decoder), so the frozen
    SAM backbone weights have to be merged in from the base SAM checkpoint.
    """
    _ensure_import_path()
    from hi_sam.modeling.hi_sam import HiSam
    from hi_sam.modeling.image_encoder import ImageEncoderViT
    from hi_sam.modeling.mask_decoder import MaskDecoder
    from hi_sam.modeling.modal_aligner import ModalAligner
    from hi_sam.modeling.prompt_encoder import PromptEncoder
    from hi_sam.modeling.transformer import TwoWayTransformer

    if model_type not in _VIT_CONFIG:
        raise ValueError(f"model_type must be one of {sorted(_VIT_CONFIG)}, got {model_type!r}")
    for path, label in ((hisam_checkpoint, "Hi-SAM"), (sam_checkpoint, "base SAM")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    cfg = _VIT_CONFIG[model_type]
    prompt_embed_dim, image_size, patch_size = 256, 1024, 16
    embedding_size = image_size // patch_size

    model = HiSam(
        image_encoder=ImageEncoderViT(
            depth=cfg["depth"],
            embed_dim=cfg["embed_dim"],
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=cfg["num_heads"],
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=cfg["global_attn_indexes"],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        modal_aligner=ModalAligner(prompt_embed_dim, attn_layers=attn_layers, prompt_len=prompt_len),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(embedding_size, embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    state = torch.load(hisam_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and any(k in state for k in ("model", "optimizer", "lr_scheduler", "epoch")):
        state = state.get("model", state)
    sam_state = torch.load(sam_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sam_state, dict) and "model" in sam_state:
        sam_state = sam_state["model"]

    merged = dict(state)
    for key, value in sam_state.items():
        merged.setdefault(key, value)
    del sam_state

    info = model.load_state_dict(merged, strict=False)
    if verbose:
        print(f"[hisam] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    # The hierarchical decoder is unused for stroke-only segmentation; anything
    # else missing means the checkpoints do not match the requested model_type.
    real_missing = [k for k in info.missing_keys if not k.startswith("hi_decoder")]
    if real_missing:
        raise RuntimeError(
            f"{len(real_missing)} weights missing after loading (e.g. {real_missing[:3]}). "
            f"Do the checkpoints match model_type={model_type!r}?"
        )

    model.eval()
    model.to(device)
    return model


def _taper(h: int, w: int, edge: int) -> np.ndarray:
    """Blend weights that fall off near the tile border, to hide seams."""
    def ramp(n: int) -> np.ndarray:
        v = np.ones(n, dtype=np.float32)
        e = min(edge, n // 2)
        if e > 0:
            fade = np.linspace(0.0, 1.0, e + 2, dtype=np.float32)[1:-1]
            v[:e] = fade
            v[-e:] = fade[::-1]
        return v

    return np.outer(ramp(h), ramp(w))


def _windows(extent: int, tile: int, stride: int):
    """Start offsets covering `extent` with `tile`-sized windows."""
    if extent <= tile:
        return [(0, extent)]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] + tile < extent:
        starts.append(extent - tile)
    return [(s, s + tile) for s in starts]


@dataclass
class SegmenterConfig:
    tile: int = 1024
    overlap: float = 0.25
    precision: str = "fp16"
    taper: int = 32


class StrokeSegmenter:
    """Runs Hi-SAM over image regions and returns per-pixel stroke probabilities."""

    def __init__(self, model, device: str = "cuda", config: Optional[SegmenterConfig] = None) -> None:
        _ensure_import_path()
        from hi_sam.modeling.predictor import SamPredictor

        self.model = model
        self.device = device
        self.cfg = config or SegmenterConfig()
        self.predictor = SamPredictor(model)
        self._use_amp = self.cfg.precision == "fp16" and device.startswith("cuda")

    def _infer(self, image: np.ndarray) -> np.ndarray:
        """Stroke logits for one image, at the image's own resolution."""
        ctx = torch.autocast("cuda", dtype=torch.float16) if self._use_amp else nullcontext()
        with ctx:
            self.predictor.set_image(np.ascontiguousarray(image))
            _, hr_masks, _, _ = self.predictor.predict(multimask_output=False, return_logits=True)
        return hr_masks[0].astype(np.float32)

    def segment(self, region: np.ndarray) -> np.ndarray:
        """Stroke probability map for an RGB region, same HxW as the input.

        Regions are fed to the encoder at up to `tile` px on the long side.
        Anything larger is covered by overlapping tiles so that stroke detail
        survives instead of being squashed into a 1024px input.
        """
        h, w = region.shape[:2]
        tile = self.cfg.tile
        if max(h, w) <= tile:
            return _sigmoid(self._infer(region))

        stride = max(1, int(tile * (1.0 - self.cfg.overlap)))
        acc = np.zeros((h, w), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)
        for y0, y1 in _windows(h, min(tile, h), stride):
            for x0, x1 in _windows(w, min(tile, w), stride):
                patch = region[y0:y1, x0:x1]
                logits = self._infer(patch)
                weight = _taper(y1 - y0, x1 - x0, self.cfg.taper)
                acc[y0:y1, x0:x1] += logits * weight
                wsum[y0:y1, x0:x1] += weight
        np.maximum(wsum, 1e-6, out=wsum)
        return _sigmoid(acc / wsum)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32)


def resolve_checkpoints(models_dir: str, model_type: str, hisam_ckpt: Optional[str]) -> Tuple[str, str]:
    """Locate the Hi-SAM and base SAM checkpoints inside `models_dir`."""
    sam_path = os.path.join(models_dir, SAM_CHECKPOINTS[model_type])
    if hisam_ckpt:
        return hisam_ckpt, sam_path
    # Released stroke-segmentation checkpoints are named sam_tss_<size>_<data>.pth
    size = model_type.split("_")[-1]
    candidates = sorted(
        f for f in os.listdir(models_dir) if f.startswith(f"sam_tss_{size}") and f.endswith(".pth")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No sam_tss_{size}*.pth checkpoint in {models_dir}. Pass --hisam-checkpoint explicitly."
        )
    return os.path.join(models_dir, candidates[0]), sam_path
