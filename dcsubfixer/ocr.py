"""Text detection with PP-OCRv6.

Only the detection stage is used: the pipeline needs to know *which* frames
carry text and roughly *where*, never what the text says. Skipping recognition
removes most of the OCR cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import _paddle_env

DEFAULT_DET_MODEL = "PP-OCRv6_medium_det"
DEFAULT_REC_MODEL = "PP-OCRv6_medium_rec"


@dataclass
class DetectorConfig:
    model_name: str = DEFAULT_DET_MODEL
    device: str = "gpu:0"
    limit_side_len: int = 1280
    limit_type: str = "max"
    thresh: float = 0.3
    box_thresh: float = 0.3  # detection floor; the real bar is applied per track
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

    def detect_batch(
        self, frames: Sequence[np.ndarray]
    ) -> List[Tuple[List[np.ndarray], List[float]]]:
        """Detect text quads in a batch of RGB frames.

        Returns, per frame, the 4x2 float quads in that frame's pixel
        coordinates and their confidences. The quads are kept rather than
        reduced to boxes here because their orientation is a useful signal:
        captions are level, scene text usually is not.
        """
        if not frames:
            return []
        # PaddleOCR expects BGR, matching OpenCV's convention.
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        results = self._model.predict(bgr, batch_size=min(self.cfg.batch_size, len(bgr)))
        return [_extract(r) for r in results]


def _extract(result) -> Tuple[List[np.ndarray], List[float]]:
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
        return [], []
    scores = None
    for key in ("dt_scores", "scores"):
        try:
            scores = result[key]
            break
        except (KeyError, TypeError):
            continue
    if scores is None or len(scores) != len(polys):
        scores = [1.0] * len(polys)

    out_polys, out_scores = [], []
    for poly, score in zip(polys, scores):
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 3:
            out_polys.append(arr)
            out_scores.append(float(score))
    return out_polys, out_scores


@dataclass
class RecognizerConfig:
    model_name: str = DEFAULT_REC_MODEL
    device: str = "gpu:0"
    batch_size: int = 8


class TextRecognizer:
    """Reads the text in a crop, and says how sure it is.

    Recognition is the one signal here that does not depend on how the footage
    was shot: scene texture that a detector calls text reads back as gibberish
    and scores low, whatever its angle, contrast or motion. It is skipped in
    the main pass because running it per frame would cost more than everything
    else combined - it is applied once per run of text instead.
    """

    def __init__(self, config: Optional[RecognizerConfig] = None) -> None:
        self.cfg = config or RecognizerConfig()
        _paddle_env.apply()
        _paddle_env.isolate()
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise RuntimeError("paddleocr is not installed") from exc
        self._model = TextRecognition(
            model_name=self.cfg.model_name, device=self.cfg.device
        )

    def read(self, crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognise a batch of RGB crops, returning (text, confidence)."""
        if not crops:
            return []
        bgr = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops]
        results = self._model.predict(bgr, batch_size=min(self.cfg.batch_size, len(bgr)))
        out = []
        for r in results:
            text, score = "", 0.0
            for key in ("rec_text", "text"):
                try:
                    text = str(r[key])
                    break
                except (KeyError, TypeError):
                    continue
            for key in ("rec_score", "score"):
                try:
                    score = float(r[key])
                    break
                except (KeyError, TypeError):
                    continue
            out.append((text, score))
        return out
