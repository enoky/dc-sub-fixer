"""Turning per-frame OCR boxes into stable, temporally coherent regions.

Raw detection boxes jitter by a few pixels from frame to frame even when the
on-screen text is identical. That jitter matters twice over: it breaks the
mask cache (forcing needless Hi-SAM runs) and it makes the composited glyphs
shimmer. Boxes are therefore merged, snapped to a grid, and then linked into
tracks across time so isolated false positives can be dropped and one-frame
dropouts filled.

A region keeps the detections it was built from, not just its own outline. The
outline is a rectangle around one or more lines of text, and a rectangle drawn
around two lines of different widths encloses corners that hold neither. Hi-SAM
segments every glyph in what it is given, so anything text-like sitting in
those corners - a label on a prop, a sign in the background - comes back in the
mask looking exactly like a caption. Keeping the detections lets the mask be
filtered back down to where text was actually found; see `gate_by_detections`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence,
                    Tuple, Union)

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive on x1/y1)


class Detection(NamedTuple):
    """One text box from the detector, with what we know about it."""

    box: Box
    score: float = 1.0
    tilt: float = 0.0  # degrees off horizontal, 0..90


class Region(NamedTuple):
    """A crop to segment, plus the detections that justify it."""

    box: Box
    dets: Tuple[Box, ...] = ()
    score: float = 1.0  # the best detection in it, for track-level judging


def poly_tilt(poly: np.ndarray) -> float:
    """How far off horizontal a detection sits, in degrees.

    Taken from the quad's own text edge rather than cv2.minAreaRect, whose
    angle convention flips between OpenCV versions and reports 90 degrees for
    perfectly level boxes depending on which corner it calls first.

    PP-OCR returns four points clockwise from the top left, so the first edge
    runs along the text baseline.
    """
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(p) == 4:
        dx, dy = p[1] - p[0]
    else:
        edges = np.roll(p, -1, axis=0) - p
        dx, dy = edges[int(np.argmax((edges ** 2).sum(axis=1)))]
    angle = float(np.degrees(np.arctan2(float(dy), float(dx))))
    return abs(((angle + 90.0) % 180.0) - 90.0)


RegionLike = Union[Region, Box]

# Shortest time a real caption or credit stays on screen, near enough. Anything
# briefer than this cannot be read and is treated as a false detection.
#
# Half a second sits between the two things it has to separate. Real captions
# are up for a second or more, so this keeps a factor of two in hand; scene
# texture that a detector momentarily reads as text persists for far less. It
# is the main defence against a false positive, which unlike a missed caption
# actively damages a region of the depth map that was fine.
MIN_TRACK_SECONDS = 0.5


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
    # Captions and credits are set level. Scene text - a sign, a label, a
    # newspaper being held - almost never is, so anything appreciably off
    # horizontal is rejected outright.
    max_tilt: float = 4.0
    # Reject a run whose box moves more than this fraction of its own text
    # height between frames, typically. Captions are pinned to the frame;
    # scene text moves with whatever carries it. Slow movement reads as zero
    # because boxes are grid-snapped first, so a scrolling credit roll still
    # passes - raise this or set it to None if a fast roll is being cut.
    max_motion: Optional[float] = 0.02
    # Reading the text back is the one check that does not depend on how the
    # footage was shot, but it needs the video, so it lives in the worker; see
    # smooth_timeline's track_filter. 0 disables it.
    rec_score: float = 0.55
    rec_min_chars: int = 2
    # A track is kept when its *best* frame clears this. Judging each frame
    # alone forces the threshold high enough to survive a caption's weakest
    # moment, which then cuts the fades at either end of every real one.
    track_score: float = 0.60
    roi: Optional[Tuple[float, float, float, float]] = None  # x0,y0,x1,y1 as 0..1 fractions
    # Drop text present in fewer than this many frames. On-screen text has to
    # stay up long enough to read, so a short-lived detection is nearly always
    # a false positive on scene texture. Resolved from the frame rate when left
    # at None; see MIN_TRACK_SECONDS.
    min_track: Optional[int] = None
    max_gap: int = 3       # bridge dropouts up to this many frames long
    sticky_iou: float = 0.8  # hold a box steady while it overlaps its run this much


def as_region(item: RegionLike) -> Region:
    """Accept a bare box where a region is expected, gating to the box itself."""
    if isinstance(item, Region):
        return item
    box = tuple(item)  # type: ignore[arg-type]
    return Region(box, (box,))


def polys_to_boxes(polys: Sequence[np.ndarray]) -> List[Box]:
    boxes = []
    for poly in polys:
        p = np.asarray(poly, np.float32).reshape(-1, 2)
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        boxes.append((int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))))
    return boxes


def detections_from_polys(
    polys: Sequence[np.ndarray], scores: Optional[Sequence[float]] = None
) -> List[Detection]:
    """Pair each detector quad with its score and its angle off horizontal."""
    boxes = polys_to_boxes(polys)
    if scores is None:
        scores = [1.0] * len(boxes)
    return [
        Detection(box, float(score), poly_tilt(poly))
        for box, poly, score in zip(boxes, polys, scores)
    ]


def filter_detections(
    dets: Sequence[Detection], frame_size: Tuple[int, int], cfg: RegionConfig
) -> List[Detection]:
    """Drop detections that are too small, off-ROI, or not level."""
    kept = []
    for det in dets:
        if cfg.max_tilt is not None and det.tilt > cfg.max_tilt:
            continue
        if filter_boxes([det.box], frame_size, cfg):
            kept.append(det)
    return kept


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


def _dedupe(boxes: Iterable[Box]) -> Tuple[Box, ...]:
    """Drop exact duplicates while keeping order.

    Deliberately not a merge: two detections that overlap are usually two lines
    of the same caption, and unioning them would recreate the very rectangle
    whose empty corners the detections exist to exclude.
    """
    seen, out = set(), []
    for b in boxes:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return tuple(out)


def merge_boxes(boxes: Sequence[Box], gap: int) -> List[Box]:
    """Union boxes that sit within `gap` pixels of each other."""
    return [r.box for r in merge_regions([as_region(b) for b in boxes], gap)]


def merge_regions(items: Sequence[RegionLike], gap: int) -> List[Region]:
    """Union regions whose boxes sit within `gap` pixels, keeping every detection."""
    merged = [as_region(i) for i in items]
    changed = True
    while changed:
        changed = False
        out: List[Region] = []
        for region in merged:
            for i, existing in enumerate(out):
                if _overlaps(existing.box, region.box, gap):
                    out[i] = Region(
                        _union(existing.box, region.box),
                        _dedupe(existing.dets + region.dets),
                        max(existing.score, region.score),
                    )
                    changed = True
                    break
            else:
                out.append(region)
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
    polys: Sequence[np.ndarray],
    frame_size: Tuple[int, int],
    cfg: RegionConfig,
    scores: Optional[Sequence[float]] = None,
) -> List[Region]:
    dets = filter_detections(detections_from_polys(polys, scores), frame_size, cfg)
    if not dets:
        return []
    merged = merge_regions([Region(d.box, (d.box,), d.score) for d in dets], cfg.merge_gap)
    snapped = [Region(stabilise(r.box, frame_size, cfg), r.dets, r.score) for r in merged]
    # Padding can push neighbours into overlap; fold those together again.
    return merge_regions(snapped, 0)


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
    regions: Dict[int, Region] = field(default_factory=dict)  # frame index -> region
    last_frame: int = -1

    @property
    def span(self) -> int:
        return len(self.regions)

    @property
    def peak_score(self) -> float:
        """The best detection confidence this track ever reached."""
        return max((r.score for r in self.regions.values()), default=0.0)

    @property
    def motion(self) -> float:
        """Typical movement between frames, as a fraction of the text height.

        The *median* step, not the total travel. A caption that never moves
        still shows a large total spread now and then, because the detector
        occasionally merges a neighbouring line and jumps the box for a frame
        or two; on the credit clip that put stationary credits at up to 2.0
        text heights of spread, overlapping the scene text completely. The
        median ignores those isolated jumps and reads 0.000 for every real
        credit there, against 0.037 and up for everything handheld.

        Must be read before smooth_timeline canonicalises a run's boxes to
        their union, which erases the very movement being measured.
        """
        frames = sorted(self.regions)
        if len(frames) < 2:
            return 0.0
        boxes = np.array([self.regions[f].box for f in frames], dtype=float)
        height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        if height <= 0:
            return 0.0

        # Take the part of the change that both edges share, which is the
        # translation; a box that only grows or shrinks contributes nothing.
        # Measuring the centre instead would call a still caption mobile
        # whenever its box flickers by one grid step, since that shifts the
        # centre by half a step - and at typical caption sizes that lands
        # squarely in the range real scene text occupies.
        dx = np.minimum(np.abs(np.diff(boxes[:, 0])), np.abs(np.diff(boxes[:, 2])))
        dy = np.minimum(np.abs(np.diff(boxes[:, 1])), np.abs(np.diff(boxes[:, 3])))
        return float(np.median(np.hypot(dx, dy)) / height)


def build_tracks(per_frame: Sequence[List[RegionLike]], iou_thresh: float = 0.3) -> List[Track]:
    """Link regions across frames into tracks by box overlap."""
    tracks: List[Track] = []
    for idx, items in enumerate(per_frame):
        available = [t for t in tracks if idx - t.last_frame <= 12]
        for item in items:
            region = as_region(item)
            best, best_iou = None, iou_thresh
            for track in available:
                score = iou(track.regions[track.last_frame].box, region.box)
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
            best.regions[idx] = region
            best.last_frame = idx
    return tracks


def _run_union(track: "Track", frames: Sequence[int]) -> Box:
    box = track.regions[frames[0]].box
    for f in frames[1:]:
        box = _union(box, track.regions[f].box)
    return box


def smooth_timeline(
    per_frame: Sequence[List[RegionLike]],
    cfg: RegionConfig,
    iou_thresh: float = 0.3,
    min_track: Optional[int] = None,
    track_filter: Optional[Callable[["Track"], bool]] = None,
) -> List[List[Region]]:
    """Drop short-lived detections and bridge brief dropouts within a track.

    `track_filter` gets the last word on a track that has passed the cheap
    checks, and is where anything needing the video itself belongs - reading
    the text back, say. It is called once per surviving track rather than once
    per frame, which is what makes an expensive check affordable.
    """
    if min_track is None:
        min_track = cfg.min_track if cfg.min_track is not None else 2
    tracks = build_tracks(per_frame, iou_thresh)
    out: List[List[Region]] = [[] for _ in per_frame]
    n = len(per_frame)

    for track in tracks:
        if track.span < min_track:
            continue
        # Judge the track by its best moment, not each frame on its own. A
        # caption fades in and out, so its first and last frames score no
        # better than scene texture; a per-frame threshold high enough to
        # reject the texture therefore eats the fades off every real caption.
        # Whether something is text does not change frame to frame, so the
        # verdict is made once and every frame of the track inherits it.
        if cfg.track_score and track.peak_score < cfg.track_score:
            continue
        # Captions are fixed to the frame; scene text rides on whatever is
        # carrying it. Checked here, before the boxes are canonicalised below.
        if cfg.max_motion is not None and track.motion > cfg.max_motion:
            continue
        # Last, because it is the only check that costs anything.
        if track_filter is not None and not track_filter(track):
            continue
        frames = sorted(track.regions)

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
        #
        # Only the outline is canonicalised. Each frame keeps its own
        # detections, because those are what the mask is filtered against and
        # borrowing a neighbour's would loosen the filter for no gain.
        runs: List[List[int]] = []
        for f in frames:
            if runs and iou(_run_union(track, runs[-1]), track.regions[f].box) >= cfg.sticky_iou:
                runs[-1].append(f)
            else:
                runs.append([f])
        for run in runs:
            canonical = _run_union(track, run)
            for f in run:
                track.regions[f] = Region(canonical, track.regions[f].dets)

        filled = dict(track.regions)
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if 0 < gap <= cfg.max_gap:
                # Hold the earlier region across the dropout: text that flickers
                # out for a frame or two is nearly always still on screen. Its
                # detections come along, so the bridged frames stay filterable.
                for f in range(a + 1, b):
                    filled[f] = track.regions[a]
        for f, region in filled.items():
            if 0 <= f < n:
                out[f].append(region)

    return [merge_regions(items, 0) if items else [] for items in out]


# ---------------------------------------------------------------- serialising
def region_to_json(region: Region) -> dict:
    return {
        "box": list(region.box),
        "dets": [list(d) for d in region.dets],
        "score": round(float(region.score), 4),
    }


def region_from_json(entry) -> Region:
    """Read a region back, tolerating timelines written before detections were kept."""
    if isinstance(entry, dict):
        box = tuple(entry["box"])
        dets = tuple(tuple(d) for d in entry.get("dets") or ())
        return Region(box, dets or (box,), float(entry.get("score", 1.0)))
    # A bare box from an older cache: gate to the whole region, i.e. no filtering.
    box = tuple(entry)
    return Region(box, (box,))


def timeline_has_detections(raw: Sequence[Sequence]) -> bool:
    """Whether a loaded timeline carries real detections, or just outlines."""
    for frame in raw:
        for entry in frame:
            return isinstance(entry, dict) and bool(entry.get("dets"))
    return True


# -------------------------------------------------------------------- gating
def gate_by_detections(
    prob: np.ndarray,
    region: Region,
    dilate: int = 8,
    min_inside: float = 0.5,
) -> np.ndarray:
    """Drop stroke blobs that are not where the detector found text.

    `prob` covers `region.box`; the detections are in frame coordinates.

    Whole connected components are kept or dropped, rather than the mask being
    clipped pixelwise to the detections. A caption's outline routinely clips
    the corner of some unrelated piece of scene text, and clipping pixelwise
    leaves that fragment behind looking like a broken glyph. A blob is the
    natural unit: either it is a glyph the detector found, or it is not.
    """
    if not region.dets or min_inside <= 0.0:
        return prob

    h, w = prob.shape
    x0, y0 = region.box[0], region.box[1]
    gate = np.zeros((h, w), np.uint8)
    for dx0, dy0, dx1, dy1 in region.dets:
        cv2.rectangle(gate, (dx0 - x0, dy0 - y0), (dx1 - x0 - 1, dy1 - y0 - 1), 1, -1)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        gate = cv2.dilate(gate, kernel)

    binary = (prob > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return prob

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    inside = np.bincount(labels[gate > 0].ravel(), minlength=count).astype(np.float64)
    keep = inside / np.maximum(areas, 1.0) >= min_inside
    keep[0] = False  # background
    return prob * keep[labels]
