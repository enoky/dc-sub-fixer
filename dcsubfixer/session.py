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

import cv2
import numpy as np

from . import composite as comp
from . import geometry, hisam, regions, video
from .regions import Box
from .video import FrameReader  # re-exported: callers here have always used it


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

    def size(self) -> int:
        return sum(os.path.getsize(os.path.join(self.root, n))
                   for n in os.listdir(self.root) if n.endswith(".png"))

    def clear(self) -> int:
        """Delete every stored mask. Returns the bytes reclaimed."""
        freed = 0
        for name in os.listdir(self.root):
            if name.endswith(".png"):
                path = os.path.join(self.root, name)
                freed += os.path.getsize(path)
                os.remove(path)
        return freed


@dataclass
class FramePanels:
    """Everything needed to draw one frame of the comparison view."""

    index: int
    rgb: np.ndarray                 # HxWx3 uint8, source resolution
    regions: List[regions.Region]   # text regions, RGB coordinates
    depth_before: np.ndarray        # HxW uint16, depth resolution
    depth_after: np.ndarray         # HxW uint16
    prob: np.ndarray                # HxW float32 in depth space
    value_range: Tuple[float, float]  # shared display window for the two depths

    @property
    def boxes(self) -> List[Box]:
        return [r.box for r in self.regions]

    @property
    def changed(self) -> bool:
        return bool(self.regions) and not np.array_equal(self.depth_before, self.depth_after)


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
        gate: bool = True,
        gate_dilate: int = 8,
        gate_min_inside: float = 0.5,
        run_mask: bool = True,
        run_mask_samples: int = 5,
    ) -> None:
        # Segment a run once, at its strongest frame, instead of every frame
        # separately. See best_run_mask.
        self.run_mask = run_mask
        self.run_mask_samples = run_mask_samples
        self._run_masks: Dict[Tuple[int, Box], np.ndarray] = {}
        # Masks are cached raw and filtered on use, so these stay adjustable
        # without re-running Hi-SAM.
        self.gate = gate
        self.gate_dilate = gate_dilate
        self.gate_min_inside = gate_min_inside
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

        self.timeline: List[List[regions.Region]] = []
        # Runs the user has rejected by eye. Some false positives are real,
        # legible, level, stationary text on a prop or a screen, which no
        # measurement here can tell from a caption; saying so directly is more
        # honest than another threshold.
        self.excluded: set = set()
        self.timeline_note: str = ""  # why a cached timeline was refused, if it was
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
    def set_timeline(self, data: dict) -> None:
        """Adopt a detection result.

        The only place a worker result is turned into regions. Callers used to
        unpack it themselves, which silently produced garbage once entries
        stopped being bare boxes: `tuple()` on the new dict yields its keys.
        """
        self.timeline = [[regions.region_from_json(e) for e in f] for f in data["regions"]]
        self.excluded = set(data.get("excluded", []))

    def load_timeline(self) -> bool:
        self.timeline_note = ""
        if not os.path.isfile(self.paths.timeline):
            return False
        try:
            with open(self.paths.timeline, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self.timeline_note = f"cached timeline unreadable ({exc}); detect again"
            return False
        if not regions.timeline_is_current(data):
            # Refused rather than half-read: an older timeline still has boxes
            # and still loads, which is exactly why it used to go unnoticed.
            self.timeline_note = "cached timeline is from an older format; detect again"
            return False
        self.set_timeline(data)
        return True

    def save_timeline(self, data: dict) -> None:
        data = {**data, "version": regions.TIMELINE_VERSION,
                "excluded": sorted(self.excluded)}
        with open(self.paths.timeline, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def persist_exclusions(self) -> bool:
        """Write the current exclusions back into the cached timeline."""
        if not os.path.isfile(self.paths.timeline):
            return False
        with open(self.paths.timeline, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.save_timeline(data)
        return True

    def run_at(self, frame: int) -> Optional[int]:
        """The run id under a frame, preferring one that is not excluded."""
        items = self.timeline[frame] if frame < len(self.timeline) else []
        ids = [r.run for r in items if r.run >= 0]
        if not ids:
            return None
        live = [i for i in ids if i not in self.excluded]
        return live[0] if live else ids[0]

    def toggle_run(self, run_id: int) -> bool:
        """Exclude or restore a run. Returns True if it is now excluded."""
        if run_id in self.excluded:
            self.excluded.discard(run_id)
            return False
        self.excluded.add(run_id)
        return True

    def visible(self, frame: int) -> List[regions.Region]:
        items = self.timeline[frame] if frame < len(self.timeline) else []
        return [r for r in items if r.run not in self.excluded]

    def text_runs(self) -> List[Tuple[int, int]]:
        """Contiguous spans that carry text, for a timeline strip."""
        frames = [i for i in range(len(self.timeline)) if self.visible(i)]
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

    def run_frames(self, run_id: int) -> List[int]:
        return [i for i, items in enumerate(self.timeline)
                if any(r.run == run_id for r in items)]

    def best_run_mask(self, region: regions.Region) -> Optional[np.ndarray]:
        """The strongest segmentation anywhere in this run, for the whole run.

        Hi-SAM is a text-stroke model, and on anything that is not quite text -
        a logo, a stylised mark - its confidence swings hard with the
        background. On the DC ident the same static logo segments cleanly on
        one frame and almost vanishes forty frames later, while the two real
        captions beside it are perfect throughout. Segmenting each frame
        separately re-rolls that every time, which is what makes the mask
        flicker.

        The run has already been shown to be stationary (it passed the motion
        filter) and its box is canonical across the run, so one mask is valid
        for all of it - and using the best one is strictly better than using an
        arbitrary one. It also makes the composited glyphs perfectly steady,
        which per-frame segmentation cannot be.
        """
        if region.run < 0:
            return None
        key = (region.run, region.box)
        cached = self._run_masks.get(key)
        if cached is not None:
            return cached

        frames = self.run_frames(region.run)
        if not frames:
            return None
        n = max(1, min(self.run_mask_samples, len(frames)))
        picks = [frames[int(round(i * (len(frames) - 1) / max(n - 1, 1)))] for i in range(n)]

        best, best_score = None, -1.0
        for f in dict.fromkeys(picks):
            mask = self.mask_for(f, region.box, run_mask=False)
            score = float((mask > 0.5).sum())
            if score > best_score:
                best, best_score = mask, score
        if best is not None:
            self._run_masks[key] = best
        return best

    def mask_for(self, frame: int, box: Box, rgb: Optional[np.ndarray] = None,
                 run_mask: bool = True) -> np.ndarray:
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
        return all(
            (frame, r.box) in self._mem or os.path.isfile(self.masks.path_for(frame, r.box))
            for r in self.visible(frame)
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
        items = self.visible(frame)

        prob_rgb = np.zeros((self.rgb_info.height, self.rgb_info.width), np.float32)
        for region in items:
            x0, y0, x1, y1 = region.box
            if x1 <= x0 or y1 <= y0:
                continue
            if not segment and not self.is_cached(frame):
                continue
            mask = None
            if self.run_mask:
                mask = self.best_run_mask(region)
            if mask is None:
                mask = self.mask_for(frame, region.box, rgb)
            if self.gate:
                mask = regions.gate_by_detections(
                    mask, region, self.gate_dilate, self.gate_min_inside
                )
            np.maximum(prob_rgb[y0:y1, x0:x1], mask, out=prob_rgb[y0:y1, x0:x1])

        prob = self.alignment.warp(prob_rgb)
        depth_boxes = [self.alignment.map_box(r.box) for r in items]
        region_mask = comp.stack_region_masks((dw, dh), depth_boxes) if cfg.heal else None
        after = comp.composite_frame(before, prob, cfg, depth_boxes, region_mask, self.max_value)

        lo = float(min(before.min(), after.min()))
        hi = float(max(before.max(), after.max()))
        return FramePanels(frame, rgb, items, before, after, prob, (lo, hi))

    def clear_cache(self, masks: bool = True, timeline: bool = False) -> int:
        """Throw away what this clip has cached. Returns the bytes reclaimed.

        The two halves are not worth the same. Masks are the bulk of it and are
        pure GPU output: deleting them costs about 0.2s a frame to rebuild, and
        rebuilds identically. The timeline is small but holds the detection
        settings that produced it and any runs rejected by hand, which is work
        that cannot be recreated by running something again - so it goes only
        when asked for explicitly.
        """
        freed = 0
        if masks:
            freed += self.masks.clear()
            self._mem.clear()
        if timeline and os.path.isfile(self.paths.timeline):
            freed += os.path.getsize(self.paths.timeline)
            os.remove(self.paths.timeline)
            self.timeline, self.excluded = [], set()
        return freed

    def cache_size(self) -> int:
        total = self.masks.size()
        if os.path.isfile(self.paths.timeline):
            total += os.path.getsize(self.paths.timeline)
        return total

    def close(self) -> None:
        self.rgb_reader.close()
        self.depth_reader.close()


def to_display(plane: np.ndarray, value_range: Tuple[float, float]) -> np.ndarray:
    """Normalise a depth plane to 8-bit for display over a shared window."""
    lo, hi = value_range
    span = max(hi - lo, 1.0)
    return np.clip((plane.astype(np.float32) - lo) / span * 255.0, 0, 255).astype(np.uint8)
