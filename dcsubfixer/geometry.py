"""Mapping between RGB-source pixel space and depth-map pixel space.

DepthCrafter output is typically both lower resolution and, depending on how the
clip was prepared, a different aspect ratio. Three ways a frame can land in the
depth video are supported:

  stretch  non-uniform scale, the whole RGB frame fills the whole depth frame
  fit      uniform scale, RGB fitted inside the depth frame with letterbox bars
  fill     uniform scale, RGB centre-cropped to fill the depth frame

All three reduce to an axis-aligned affine map, so a single warp covers them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MODES = ("stretch", "fit", "fill")


@dataclass
class Alignment:
    """x_depth = sx * x_rgb + tx,  y_depth = sy * y_rgb + ty."""

    mode: str
    sx: float
    sy: float
    tx: float
    ty: float
    rgb_size: Tuple[int, int]  # (w, h)
    depth_size: Tuple[int, int]  # (w, h)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.sx, 0.0, self.tx], [0.0, self.sy, self.ty]], dtype=np.float32)

    def warp(self, mask: np.ndarray) -> np.ndarray:
        """Warp an RGB-space map into depth space."""
        dw, dh = self.depth_size
        return cv2.warpAffine(
            mask,
            self.matrix,
            (dw, dh),
            flags=cv2.INTER_AREA if self.sx < 1.0 else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def map_box(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Map an RGB-space box into depth space, clamped to the depth frame."""
        dw, dh = self.depth_size
        x0 = int(np.floor(self.sx * box[0] + self.tx))
        y0 = int(np.floor(self.sy * box[1] + self.ty))
        x1 = int(np.ceil(self.sx * box[2] + self.tx))
        y1 = int(np.ceil(self.sy * box[3] + self.ty))
        return (max(0, x0), max(0, y0), min(dw, max(x0 + 1, x1)), min(dh, max(y0 + 1, y1)))

    def describe(self) -> str:
        return (
            f"{self.mode}: scale=({self.sx:.4f}, {self.sy:.4f}) "
            f"offset=({self.tx:.1f}, {self.ty:.1f})"
        )


def build_alignment(
    rgb_size: Tuple[int, int],
    depth_size: Tuple[int, int],
    mode: str,
) -> Alignment:
    rw, rh = rgb_size
    dw, dh = depth_size
    if mode == "stretch":
        sx, sy = dw / rw, dh / rh
        tx = ty = 0.0
    elif mode == "fit":
        s = min(dw / rw, dh / rh)
        sx = sy = s
        tx = (dw - s * rw) / 2.0
        ty = (dh - s * rh) / 2.0
    elif mode == "fill":
        s = max(dw / rw, dh / rh)
        sx = sy = s
        tx = (dw - s * rw) / 2.0
        ty = (dh - s * rh) / 2.0
    else:
        raise ValueError(f"unknown alignment mode {mode!r}, expected one of {MODES}")
    return Alignment(mode, sx, sy, tx, ty, rgb_size, depth_size)


def _bar_extent(frames: Sequence[np.ndarray], axis: int, tol: int = 6) -> Tuple[int, int]:
    """Measure uniform dark border thickness along one axis, across frames.

    Returns (leading, trailing) thickness in pixels. `axis=0` measures top and
    bottom bars, `axis=1` measures left and right.
    """
    lead, trail = [], []
    for frame in frames:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        profile = gray.max(axis=1 - axis)  # brightest pixel in each row (or column)
        dark = profile <= tol
        n = len(dark)
        i = 0
        while i < n and dark[i]:
            i += 1
        j = 0
        while j < n and dark[n - 1 - j]:
            j += 1
        # A fully dark frame says nothing about geometry.
        if i + j >= n:
            continue
        lead.append(i)
        trail.append(j)
    if not lead:
        return 0, 0
    return int(np.median(lead)), int(np.median(trail))


def detect_alignment(
    rgb_size: Tuple[int, int],
    depth_size: Tuple[int, int],
    depth_samples: Optional[Sequence[np.ndarray]] = None,
    ar_tol: float = 0.01,
) -> Tuple[Alignment, str]:
    """Guess how the RGB frame was mapped into the depth frame.

    Returns the alignment plus a human-readable note about how it was chosen.
    """
    rw, rh = rgb_size
    dw, dh = depth_size
    ar_rgb, ar_depth = rw / rh, dw / dh
    rel = abs(ar_rgb - ar_depth) / ar_rgb

    if rel <= ar_tol:
        note = f"aspect ratios match ({ar_rgb:.4f} vs {ar_depth:.4f})"
        return build_alignment(rgb_size, depth_size, "stretch"), note

    # Aspect ratios differ. If DepthCrafter was fed a letterboxed frame, the
    # depth map carries bars we can measure; their thickness tells us which.
    if depth_samples:
        fit = build_alignment(rgb_size, depth_size, "fit")
        expected_v = (dh - fit.sy * rh) / 2.0
        expected_h = (dw - fit.sx * rw) / 2.0
        top, bottom = _bar_extent(depth_samples, axis=0)
        left, right = _bar_extent(depth_samples, axis=1)
        measured_v = (top + bottom) / 2.0
        measured_h = (left + right) / 2.0
        # Only one axis can be letterboxed for a given AR mismatch.
        target, measured = (expected_v, measured_v) if expected_v > expected_h else (expected_h, measured_h)
        if target >= 2.0 and abs(measured - target) <= max(2.0, 0.15 * target):
            note = (
                f"aspect ratios differ ({ar_rgb:.4f} vs {ar_depth:.4f}); "
                f"found letterbox bars of {measured:.0f}px matching a 'fit' map"
            )
            return fit, note

    note = (
        f"aspect ratios differ ({ar_rgb:.4f} vs {ar_depth:.4f}) and no letterbox bars "
        f"were found; assuming the frame was stretched. Pass --align fit/fill if wrong."
    )
    return build_alignment(rgb_size, depth_size, "stretch"), note


def sample_frames(path: str, count: int = 8) -> List[np.ndarray]:
    """Grab a handful of frames spread through a video, for geometry probing."""
    from . import video

    return video.sample_depth(path, count)
