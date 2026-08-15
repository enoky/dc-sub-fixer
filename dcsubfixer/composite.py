"""Compositing clean glyph masks onto depth frames.

Depth frames arrive here as HxW uint16 luma (see video.py). Grey levels are
given by the user on the familiar 0-255 scale and scaled to the frame's range
internally, so `--text-value 255` means "brightest" whatever the bit depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


@dataclass
class CompositeConfig:
    text_value: str = "auto"   # "auto" or an integer 0-255
    mask_low: float = 0.35     # probability mapped to alpha 0
    mask_high: float = 0.65    # probability mapped to alpha 1
    binary: bool = False       # hard-threshold the mask instead of feathering
    dilate: int = 0            # >0 thickens glyphs, <0 thins them (pixels)
    heal: bool = False         # inpaint DepthCrafter's jagged halo before compositing
    heal_radius: int = 6
    opacity: float = 1.0


def probability_to_alpha(prob: np.ndarray, cfg: CompositeConfig) -> np.ndarray:
    """Map stroke probabilities to a compositing alpha.

    The soft window between `mask_low` and `mask_high` keeps glyph edges
    anti-aliased, which matters because the depth map is usually a good deal
    smaller than the source and a hard mask would alias badly on downscale.
    """
    if cfg.binary:
        alpha = (prob >= cfg.mask_high).astype(np.float32)
    else:
        lo, hi = cfg.mask_low, max(cfg.mask_high, cfg.mask_low + 1e-3)
        alpha = np.clip((prob - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    if cfg.dilate:
        k = abs(cfg.dilate) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        alpha = cv2.dilate(alpha, kernel) if cfg.dilate > 0 else cv2.erode(alpha, kernel)

    if cfg.opacity != 1.0:
        alpha = alpha * float(np.clip(cfg.opacity, 0.0, 1.0))
    return alpha


def resolve_text_value(
    depth: np.ndarray, alpha: np.ndarray, cfg: CompositeConfig, max_value: int
) -> float:
    """Pick the grey level to paint glyphs with, in the frame's own units.

    "auto" reads the level back out of the depth map: the pixels under the
    mangled text already carry the depth DepthCrafter meant to give it. A
    median would understate it, because the smearing that damaged the glyphs
    also mixed background depth into them, pulling every sampled pixel toward
    the surroundings. An extreme percentile recovers the value the smear was
    spreading from.

    Which extreme depends on the map's convention, so it is measured rather
    than assumed: text nearer than its surroundings is brighter in a disparity
    map and darker in a true depth map, and either can arrive here.
    """
    if cfg.text_value != "auto":
        return float(cfg.text_value) / 255.0 * max_value

    core = alpha > 0.5
    if int(core.sum()) < 8:
        core = alpha > 0.25
    if int(core.sum()) < 4:
        return float(np.percentile(depth, 99.0))

    core_vals = depth[core]
    k = max(3, cfg.heal_radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near = cv2.dilate(core.astype(np.uint8), kernel).astype(bool)
    surround = near & (alpha <= 0.05)

    brighter = True
    if int(surround.sum()) >= 8:
        brighter = float(core_vals.mean()) >= float(depth[surround].mean())
    return float(np.percentile(core_vals, 85.0 if brighter else 15.0))


def composite_frame(
    depth: np.ndarray,
    prob: np.ndarray,
    cfg: CompositeConfig,
    depth_boxes: Optional[Sequence[Box]] = None,
    region_mask: Optional[np.ndarray] = None,
    max_value: int = 65535,
) -> np.ndarray:
    """Paint glyphs described by `prob` onto a HxW uint16 depth frame.

    `prob` is a full-frame stroke probability map already warped into depth
    space. `depth_boxes` are the text regions in depth-space coordinates; the
    grey level is resolved once per region rather than once per frame, so a
    title card and a subtitle in the same frame each keep their own depth.

    Frames with nothing to paint are returned unchanged, so they can be written
    through bit-for-bit.
    """
    alpha = probability_to_alpha(prob, cfg)
    if not alpha.any():
        return depth

    base = depth.astype(np.float32)
    if cfg.heal and cfg.heal_radius > 0:
        base = _heal_halo(base, alpha, region_mask, cfg.heal_radius, max_value)

    out = base.copy()
    boxes = list(depth_boxes) if depth_boxes else [(0, 0, depth.shape[1], depth.shape[0])]
    for x0, y0, x1, y1 in boxes:
        sub_alpha = alpha[y0:y1, x0:x1]
        if sub_alpha.size == 0 or not sub_alpha.any():
            continue
        value = resolve_text_value(depth[y0:y1, x0:x1], sub_alpha, cfg, max_value)
        out[y0:y1, x0:x1] = base[y0:y1, x0:x1] * (1.0 - sub_alpha) + value * sub_alpha

    return np.clip(out, 0, max_value).astype(np.uint16)


def _heal_halo(
    depth: np.ndarray,
    alpha: np.ndarray,
    region_mask: Optional[np.ndarray],
    radius: int,
    max_value: int,
) -> np.ndarray:
    """Inpaint the jagged remains of DepthCrafter's text before repainting it.

    Overlaying clean glyphs does not by itself remove the blobby halo the model
    left around them. Depth maps are smooth, so inpainting the ring between the
    glyph and its surroundings reconstructs plausible background depth.

    cv2.inpaint only accepts 8-bit input, so the frame is scaled to 8 bits
    across the range actually present near the text rather than the full 0..max
    range. That keeps a typical depth region's precision to a few 16-bit units
    per step, and the loss is confined to the healed ring, which is invented
    data either way.
    """
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    solid = (alpha > 0.15).astype(np.uint8)
    halo = cv2.dilate(solid, kernel)
    if region_mask is not None:
        halo = halo & (region_mask > 0).astype(np.uint8)
    if not halo.any():
        return depth

    near = cv2.dilate(halo, kernel).astype(bool)
    lo = float(depth[near].min())
    hi = float(depth[near].max())
    if hi - lo < 1e-3:
        return depth

    scaled = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    healed = cv2.inpaint(scaled, halo, radius, cv2.INPAINT_TELEA).astype(np.float32)
    healed = healed / 255.0 * (hi - lo) + lo

    out = depth.copy()
    ring = halo.astype(bool)
    out[ring] = healed[ring]
    return out


def stack_region_masks(frame_size: Tuple[int, int], boxes, dtype=np.uint8) -> np.ndarray:
    """Build a full-frame mask marking every text region box."""
    w, h = frame_size
    mask = np.zeros((h, w), dtype=dtype)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 1
    return mask
