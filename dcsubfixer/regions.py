"""Turning per-frame OCR boxes into stable, temporally coherent regions.

Raw detection boxes jitter by a few pixels from frame to frame even when the
on-screen text is identical. That jitter matters twice over: it breaks the
mask cache (forcing needless Hi-SAM runs) and it makes the composited glyphs
shimmer. Boxes are therefore merged, snapped to a grid, and then linked into
tracks across time so isolated false positives can be dropped and one-frame
dropouts filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive on x1/y1)

# Shortest time a real caption or credit stays on screen, near enough. Anything
# briefer than this cannot be read and is treated as a false detection.
MIN_TRACK_SECONDS = 0.3


def resolve_min_track(cfg: "RegionConfig", fps: float) -> int:
    if cfg.min_track is not None:
        return max(1, int(cfg.min_track))
    return max(2, int(round(MIN_TRACK_SECONDS * fps)))


@dataclass
class RegionConfig:
    pad: int = 12          # context pixels added around text before segmentation
    grid: int = 16         # snap box edges to this grid, to stabilise across frames
    merge_gap: int = 24    # boxes closer than this get merged into one region
    min_height: int = 8    # ignore detections thinner than this many pixels
    min_area: int = 120
    roi: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1 as 0..1 fractions
    # Drop text present in fewer than this many frames. On-screen text has to
    # stay up long enough to read, so a short-lived detection is nearly always
    # a false positive on scene texture. Resolved from the frame rate when left
    # at None; see MIN_TRACK_SECONDS.
    min_track: Optional[int] = None
    max_gap: int = 3       # bridge dropouts up to this many frames long
    sticky_iou: float = 0.8  # hold a box steady while it overlaps its run this much


def polys_to_boxes(polys: Sequence[np.ndarray]) -> List[Box]:
    boxes = []
    for poly in polys:
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        boxes.append((int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))))
    return boxes


def filter_boxes(boxes: Sequence[Box], frame_size: Tuple[int, int], cfg: RegionConfig) -> List[Box]:
    """Drop boxes that are too small or fall outside the region of interest."""
    w, h = frame_size
    if cfg.roi is not None:
        rx0, ry0, rx1, ry1 = cfg.roi
        roi = (rx0 * w, ry0 * h, rx1 * w, ry1 * h)
    else:
        roi = None

    kept = []
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bh < cfg.min_height or bw * bh < cfg.min_area:
            continue
        if roi is not None:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue
        kept.append((x0, y0, x1, y1))
    return kept


def _overlaps(a: Box, b: Box, gap: int) -> bool:
    return not (
        a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]
    )


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge_boxes(boxes: Sequence[Box], gap: int) -> List[Box]:
    """Union boxes that sit within `gap` pixels of each other."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: List[Box] = []
        for box in merged:
            for i, existing in enumerate(out):
                if _overlaps(existing, box, gap):
                    out[i] = _union(existing, box)
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


def stabilise(box: Box, frame_size: Tuple[int, int], cfg: RegionConfig) -> Box:
    """Pad a box and snap it to a grid, clamped to the frame."""
    w, h = frame_size
    g = max(1, cfg.grid)
    x0 = ((box[0] - cfg.pad) // g) * g
    y0 = ((box[1] - cfg.pad) // g) * g
    x1 = -(-(box[2] + cfg.pad) // g) * g
    y1 = -(-(box[3] + cfg.pad) // g) * g
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


def frame_regions(
    polys: Sequence[np.ndarray], frame_size: Tuple[int, int], cfg: RegionConfig
) -> List[Box]:
    boxes = filter_boxes(polys_to_boxes(polys), frame_size, cfg)
    if not boxes:
        return []
    merged = merge_boxes(boxes, cfg.merge_gap)
    snapped = [stabilise(b, frame_size, cfg) for b in merged]
    # Padding can push neighbours into overlap; fold those together again.
    return merge_boxes(snapped, 0)


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    boxes: Dict[int, Box] = field(default_factory=dict)  # frame index -> box
    last_frame: int = -1

    @property
    def span(self) -> int:
        return len(self.boxes)


def build_tracks(per_frame: Sequence[List[Box]], iou_thresh: float = 0.3) -> List[Track]:
    """Link regions across frames into tracks by box overlap."""
    tracks: List[Track] = []
    for idx, boxes in enumerate(per_frame):
        available = [t for t in tracks if idx - t.last_frame <= 12]
        for box in boxes:
            best, best_iou = None, iou_thresh
            for track in available:
                score = iou(track.boxes[track.last_frame], box)
                if score > best_iou:
                    best, best_iou = track, score
            if best is None:
                best = Track()
                tracks.append(best)
            elif best.last_frame == idx:
                # Already matched this frame; start a fresh track instead of
                # overwriting, so two regions never collapse into one.
                best = Track()
                tracks.append(best)
            best.boxes[idx] = box
            best.last_frame = idx
    return tracks


def _run_union(track: "Track", frames: Sequence[int]) -> Box:
    box = track.boxes[frames[0]]
    for f in frames[1:]:
        box = _union(box, track.boxes[f])
    return box


def smooth_timeline(
    per_frame: Sequence[List[Box]],
    cfg: RegionConfig,
    iou_thresh: float = 0.3,
    min_track: Optional[int] = None,
) -> List[List[Box]]:
    """Drop short-lived detections and bridge brief dropouts within a track."""
    if min_track is None:
        min_track = cfg.min_track if cfg.min_track is not None else 2
    tracks = build_tracks(per_frame, iou_thresh)
    out: List[List[Box]] = [[] for _ in per_frame]
    n = len(per_frame)

    for track in tracks:
        if track.span < min_track:
            continue
        frames = sorted(track.boxes)

        # Hold the box still while it keeps covering the same text. Snapping to
        # a grid is not enough on its own: a detection whose edge sits near a
        # grid line flips between two snapped values from frame to frame, which
        # both jitters the composited glyphs and, because the box is the mask
        # cache key, forces the same unchanged text to be segmented again.
        #
        # Each run of sufficiently-overlapping boxes collapses to their union,
        # so the run gets exactly one box and no frame ends up with a box
        # smaller than its own detection. Text that genuinely moves or resizes
        # breaks the run and starts a new one. A run that does merge two
        # different captions is still safe: the mask cache compares pixels, not
        # just boxes, so the change is caught there.
        runs: List[List[int]] = []
        for f in frames:
            if runs and iou(_run_union(track, runs[-1]), track.boxes[f]) >= cfg.sticky_iou:
                runs[-1].append(f)
            else:
                runs.append([f])
        for run in runs:
            canonical = _run_union(track, run)
            for f in run:
                track.boxes[f] = canonical

        filled = dict(track.boxes)
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if 0 < gap <= cfg.max_gap:
                # Hold the earlier box across the dropout: text that flickers out
                # for a frame or two is nearly always still on screen.
                for f in range(a + 1, b):
                    filled[f] = track.boxes[a]
        for f, box in filled.items():
            if 0 <= f < n:
                out[f].append(box)

    return [merge_boxes(boxes, 0) if boxes else [] for boxes in out]
