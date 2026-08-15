"""Text detection with PP-OCRv6.

Only the detection stage is used: the pipeline needs to know *which* frames
carry text and roughly *where*, never what the text says. Skipping recognition
removes most of the OCR cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import _paddle_env

DEFAULT_DET_MODEL = "PP-OCRv6_medium_det"


@dataclass
class DetectorConfig:
    model_name: str = DEFAULT_DET_MODEL
    device: str = "gpu:0"
    limit_side_len: int = 1280
    limit_type: str = "max"
    thresh: float = 0.3
    box_thresh: float = 0.5
    unclip_ratio: float = 1.8
    batch_size: int = 8


class TextDetector:
    """Thin wrapper over PaddleOCR's detection-only module."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        # Both must run before paddleocr (and hence paddle) is imported.
        _paddle_env.apply()
        _paddle_env.isolate()
        try:
            from paddleocr import TextDetection
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise RuntimeError(
                "paddleocr is not installed. See requirements.txt for the "
                "paddlepaddle-gpu + paddleocr install commands."
            ) from exc

        self._model = TextDetection(
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            limit_side_len=self.cfg.limit_side_len,
            limit_type=self.cfg.limit_type,
            thresh=self.cfg.thresh,
            box_thresh=self.cfg.box_thresh,
            unclip_ratio=self.cfg.unclip_ratio,
        )

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[np.ndarray]]:
        """Detect text quads in a batch of RGB frames.

        Returns, per frame, a list of 4x2 float arrays in (x, y) pixel
        coordinates of that frame.
        """
        if not frames:
            return []
        # PaddleOCR expects BGR, matching OpenCV's convention.
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        results = self._model.predict(bgr, batch_size=min(self.cfg.batch_size, len(bgr)))
        return [_extract_polys(r) for r in results]


def _extract_polys(result) -> List[np.ndarray]:
    polys = None
    if isinstance(result, dict):
        for key in ("dt_polys", "polys", "boxes"):
            if key in result:
                polys = result[key]
                break
    else:  # PaddleOCR result objects behave like mappings but are not dicts
        for key in ("dt_polys", "polys", "boxes"):
            try:
                polys = result[key]
                break
            except (KeyError, TypeError):
                continue
    if polys is None:
        return []
    out = []
    for poly in polys:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 3:
            out.append(arr)
    return out
