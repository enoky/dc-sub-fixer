"""Random-access frame reading, a disk mask cache, and a tuning session.

The batch pipeline streams both videos forward once and keeps stroke masks in
memory. A tuning tool needs the opposite: jump to any frame, and change
compositing settings repeatedly without paying for segmentation again.

The split that makes this cheap is already in the pipeline - detection gives
boxes, Hi-SAM gives a stroke probability per region, and compositing is a pure
function of (depth, probability, settings). Only the middle step needs a GPU,
so caching its output to disk turns every slider into plain array arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np

from . import composite as comp
from . import geometry, hisam, regions, video
from .regions import Box


class FrameReader:
    """Random access to video frames, by frame index.

    PyAV decodes forward, so a jump means seeking to the keyframe at or before
    the target and decoding up to it. Frame identity comes from the packet
    timestamp rather than a running count, because a seek lands wherever the
    keyframe is and counting from there drifts.
    """

    # Decoding forward is cheaper than a seek for small hops, which is the
    # common case when stepping through frames.
    FORWARD_LIMIT = 48

    def __init__(self, path: str, fmt: str = "rgb24", cache_size: int = 8) -> None:
        self.path = path
        self.fmt = fmt
        self.info = video.probe(path)
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._time_base = self._stream.time_base
        self._rate = self._stream.average_rate
        self._start = self._stream.start_time or 0
        self._decoder = None
        self._position = -1  # index of the last frame handed out
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size

    # -- index/timestamp conversion -------------------------------------
    def _index_of(self, frame) -> Optional[int]:
        pts = frame.pts if frame.pts is not None else frame.dts
        if pts is None:
            return None
        return int(round(float((pts - self._start) * self._time_base * self._rate)))

    def _pts_of(self, index: int) -> int:
        return int(index / self._rate / self._time_base) + self._start

    # -- cache ----------------------------------------------------------
    def _remember(self, index: int, arr: np.ndarray) -> None:
        if index in self._cache:
            return
        self._cache[index] = arr
        self._cache_order.append(index)
        while len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]

    # -- reading --------------------------------------------------------
    def _restart(self, index: int) -> None:
        self._container.seek(self._pts_of(index), stream=self._stream, backward=True)
        self._decoder = self._container.decode(self._stream)
        self._position = -1

    def frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        hit = self._cache.get(index)
        if hit is not None:
            return hit

        if self._decoder is None or index <= self._position or index - self._position > self.FORWARD_LIMIT:
            self._restart(index)

        last = None
        for frame in self._decoder:
            idx = self._index_of(frame)
            if idx is None:
                continue
            self._position = idx
            if idx > index:
                # Overshot: the seek landed past the target, back up further.
                break
            last = frame
            if idx == index:
                arr = frame.to_ndarray(format=self.fmt)
                self._remember(index, arr)
                return arr

        if last is not None and self._index_of(last) is not None:
            # Ran out of frames; the clip may be shorter than asked for.
            arr = last.to_ndarray(format=self.fmt)
            self._remember(index, arr)
            return arr
        raise IndexError(f"frame {index} not found in {self.path}")

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:
            pass

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MaskStore:
    """Stroke masks persisted next to the clip, so tuning never re-runs Hi-SAM.

    Keyed by frame index and region box. Probabilities are quantised to 8 bits,
    which is far finer than the alpha window they feed, and PNG-compressed.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path_for(self, frame: int, box: Box) -> str:
        x0, y0, x1, y1 = box
        return os.path.join(self.root, f"f{frame:06d}_{x0}_{y0}_{x1}_{y1}.png")

    def get(self, frame: int, box: Box) -> Optional[np.ndarray]:
        path = self.path_for(frame, box)
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img.astype(np.float32) / 255.0

    def put(self, frame: int, box: Box, prob: np.ndarray) -> None:
        quantised = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(self.path_for(frame, box), quantised)

    def clear(self) -> None:
        for name in os.listdir(self.root):
            if name.endswith(".png"):
                os.remove(os.path.join(self.root, name))


@dataclass
class FramePanels:
    """Everything needed to draw one frame of the comparison view."""

    index: int
    rgb: np.ndarray                 # HxWx3 uint8, source resolution
    boxes: List[Box]                # text regions, RGB coordinates
    depth_before: np.ndarray        # HxW uint16, depth resolution
    depth_after: np.ndarray         # HxW uint16
    prob: np.ndarray                # HxW float32 in depth space
    value_range: Tuple[float, float]  # shared display window for the two depths

    @property
    def changed(self) -> bool:
        return bool(self.boxes) and not np.array_equal(self.depth_before, self.depth_after)


@dataclass
class SessionPaths:
    rgb: str
    depth: str
    cache_dir: str

    @property
    def timeline(self) -> str:
        return os.path.join(self.cache_dir, "timeline.json")

    @property
    def masks(self) -> str:
        return os.path.join(self.cache_dir, "masks")


def default_cache_dir(rgb_path: str, depth_path: str) -> str:
    """A stable per-pair cache location under the user's temp-ish app dir."""
    key = hashlib.sha1(
        (os.path.abspath(rgb_path) + "|" + os.path.abspath(depth_path)).encode("utf-8")
    ).hexdigest()[:12]
    base = os.environ.get("DCSUBFIXER_CACHE") or os.path.join(
        os.path.expanduser("~"), ".dcsubfixer", "cache"
    )
    stem = os.path.splitext(os.path.basename(depth_path))[0]
    return os.path.join(base, f"{stem}-{key}")


class TuningSession:
    """Holds everything a GUI needs to show and retune one clip pair.

    Hi-SAM is loaded once and kept resident; the detection pass still runs in a
    subprocess, because Paddle and torch cannot share a process (see
    _paddle_env). Nothing here imports paddle.
    """

    def __init__(
        self,
        rgb_path: str,
        depth_path: str,
        models_dir: str = "models",
        cache_dir: Optional[str] = None,
        align: str = "auto",
        device: str = "cuda",
        model_type: str = "vit_l",
        hisam_checkpoint: Optional[str] = None,
        region: Optional[regions.RegionConfig] = None,
        segmenter: Optional[hisam.SegmenterConfig] = None,
    ) -> None:
        self.paths = SessionPaths(
            rgb_path, depth_path, cache_dir or default_cache_dir(rgb_path, depth_path)
        )
        os.makedirs(self.paths.cache_dir, exist_ok=True)

        self.rgb_info = video.probe(rgb_path)
        self.depth_info = video.probe(depth_path)
        self.pix_fmt = video.depth_format(self.depth_info)
        self.max_value = (1 << self.depth_info.bit_depth) - 1

        self.region = region or regions.RegionConfig()
        self.seg_cfg = segmenter or hisam.SegmenterConfig()
        self.models_dir = models_dir
        self.model_type = model_type
        self.hisam_checkpoint = hisam_checkpoint
        self.device = device

        self.alignment, self.align_note = self._resolve_alignment(align)
        self.rgb_reader = FrameReader(rgb_path, "rgb24")
        self.depth_reader = FrameReader(depth_path, self.pix_fmt)
        self.masks = MaskStore(self.paths.masks)

        self.timeline: List[List[Box]] = []
        self._segmenter: Optional[hisam.StrokeSegmenter] = None
        self._mem: Dict[Tuple[int, Box], np.ndarray] = {}

    # -- setup ----------------------------------------------------------
    def _resolve_alignment(self, align: str):
        rgb_size = (self.rgb_info.width, self.rgb_info.height)
        depth_size = (self.depth_info.width, self.depth_info.height)
        if align == "auto":
            return geometry.detect_alignment(
                rgb_size, depth_size, video.sample_depth(self.paths.depth, 6)
            )
        return geometry.build_alignment(rgb_size, depth_size, align), f"forced {align}"

    def set_alignment(self, align: str) -> None:
        self.alignment, self.align_note = self._resolve_alignment(align)

    @property
    def n_frames(self) -> int:
        counts = [n for n in (self.rgb_info.n_frames, self.depth_info.n_frames) if n]
        return min(counts) if counts else 0

    # -- detection ------------------------------------------------------
    def load_timeline(self) -> bool:
        if not os.path.isfile(self.paths.timeline):
            return False
        with open(self.paths.timeline, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.timeline = [[tuple(b) for b in boxes] for boxes in data["regions"]]
        return True

    def save_timeline(self, data: dict) -> None:
        with open(self.paths.timeline, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def text_runs(self) -> List[Tuple[int, int]]:
        """Contiguous spans that carry text, for a timeline strip."""
        frames = [i for i, b in enumerate(self.timeline) if b]
        if not frames:
            return []
        runs, start, prev = [], frames[0], frames[0]
        for f in frames[1:]:
            if f != prev + 1:
                runs.append((start, prev))
                start = f
            prev = f
        runs.append((start, prev))
        return runs

    # -- segmentation ---------------------------------------------------
    @property
    def segmenter(self) -> hisam.StrokeSegmenter:
        if self._segmenter is None:
            ckpt = hisam.resolve_checkpoints(
                self.models_dir, self.model_type, self.hisam_checkpoint
            )
            model = hisam.build_hisam(
                *ckpt, model_type=self.model_type, device=self.device
            )
            self._segmenter = hisam.StrokeSegmenter(
                model, device=self.device, config=self.seg_cfg
            )
        return self._segmenter

    @property
    def model_loaded(self) -> bool:
        return self._segmenter is not None

    def mask_for(self, frame: int, box: Box, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """Stroke probability for one region, from memory, disk, or Hi-SAM."""
        key = (frame, box)
        got = self._mem.get(key)
        if got is not None:
            return got
        got = self.masks.get(frame, box)
        if got is None:
            if rgb is None:
                rgb = self.rgb_reader.frame(frame)
            x0, y0, x1, y1 = box
            got = self.segmenter.segment(rgb[y0:y1, x0:x1])
            self.masks.put(frame, box, got)
        self._mem[key] = got
        if len(self._mem) > 64:
            self._mem.pop(next(iter(self._mem)))
        return got

    def is_cached(self, frame: int) -> bool:
        boxes = self.timeline[frame] if frame < len(self.timeline) else []
        return all(
            (frame, b) in self._mem or os.path.isfile(self.masks.path_for(frame, b))
            for b in boxes
        )

    # -- rendering ------------------------------------------------------
    def render(self, frame: int, cfg: comp.CompositeConfig, segment: bool = True) -> FramePanels:
        """Build the comparison panels for one frame.

        With `segment=False` no GPU work happens: uncached regions are simply
        left blank, so the view can stay responsive while scrubbing.
        """
        rgb = self.rgb_reader.frame(frame)
        planar = self.depth_reader.frame(frame)
        dh, dw = self.depth_info.height, self.depth_info.width
        before = video.luma(planar, dh, dw).copy()
        boxes = list(self.timeline[frame]) if frame < len(self.timeline) else []

        prob_rgb = np.zeros((self.rgb_info.height, self.rgb_info.width), np.float32)
        for box in boxes:
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            if not segment and not self.is_cached(frame):
                continue
            mask = self.mask_for(frame, box, rgb)
            np.maximum(prob_rgb[y0:y1, x0:x1], mask, out=prob_rgb[y0:y1, x0:x1])

        prob = self.alignment.warp(prob_rgb)
        depth_boxes = [self.alignment.map_box(b) for b in boxes]
        region_mask = comp.stack_region_masks((dw, dh), depth_boxes) if cfg.heal else None
        after = comp.composite_frame(before, prob, cfg, depth_boxes, region_mask, self.max_value)

        lo = float(min(before.min(), after.min()))
        hi = float(max(before.max(), after.max()))
        return FramePanels(frame, rgb, boxes, before, after, prob, (lo, hi))

    def close(self) -> None:
        self.rgb_reader.close()
        self.depth_reader.close()


def to_display(plane: np.ndarray, value_range: Tuple[float, float]) -> np.ndarray:
    """Normalise a depth plane to 8-bit for display over a shared window."""
    lo, hi = value_range
    span = max(hi - lo, 1.0)
    return np.clip((plane.astype(np.float32) - lo) / span * 255.0, 0, 255).astype(np.uint8)
