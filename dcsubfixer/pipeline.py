"""End-to-end pipeline: detect text in RGB, segment glyphs, repaint depth.

The work is split into two passes over the clip. The first runs PP-OCRv6
detection on every frame and builds a temporally smoothed timeline of text
regions; it is cheap and tells us up front which frames need attention. The
second pass runs Hi-SAM only where text actually is, caching masks for as long
as the underlying pixels stay put — a subtitle held for two seconds costs one
segmentation, not sixty.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import composite as comp
from . import geometry, hisam, ocr, regions, video
from .regions import Box, Region


@dataclass
class PipelineConfig:
    rgb_path: str
    depth_path: str
    out_path: str
    models_dir: str = "models"
    model_type: str = "vit_l"
    hisam_checkpoint: Optional[str] = None
    device: str = "cuda"
    align: str = "auto"
    ocr_stride: int = 1
    cache_tolerance: float = 2.0
    max_frames: Optional[int] = None
    timeline_path: Optional[str] = None
    # Filtering the mask back down to where the detector found text. See
    # regions.gate_by_detections.
    exclude_runs: Tuple[int, ...] = ()
    gate: bool = True
    gate_dilate: int = 8
    gate_min_inside: float = 0.5
    debug_dir: Optional[str] = None
    debug_limit: int = 12
    quality: str = "lossless"
    preset: str = "slow"
    codec: str = "libx264"
    detector: ocr.DetectorConfig = field(default_factory=ocr.DetectorConfig)
    region: regions.RegionConfig = field(default_factory=regions.RegionConfig)
    segmenter: hisam.SegmenterConfig = field(default_factory=hisam.SegmenterConfig)
    composite: comp.CompositeConfig = field(default_factory=comp.CompositeConfig)


class MaskCache:
    """Reuses stroke masks while the text inside a region stays unchanged.

    Keyed on the (grid-snapped) region box. The freshness check deliberately
    ignores most of the crop and compares only the pixels on and immediately
    around the glyphs of the cached mask: a subtitle is usually held over
    moving footage, and comparing the whole crop would report a change on every
    frame and re-segment text that never moved. What matters is whether the
    *glyphs* changed.

    Besides saving nearly all of the Hi-SAM cost on static text, handing back
    the identical mask is what makes the composited glyphs perfectly stable
    from frame to frame.
    """

    def __init__(self, tolerance: float = 2.0, ttl: int = 8, margin: int = 4) -> None:
        self.tolerance = tolerance
        self.ttl = ttl
        self.margin = margin
        self._entries: Dict[Box, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _gray(crop: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

    def _compare_zone(self, mask: np.ndarray) -> np.ndarray:
        """Glyph pixels plus a margin: where a text change would show up."""
        k = self.margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate((mask > 0.5).astype(np.uint8), kernel).astype(bool)

    def get(self, box: Box, crop: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        entry = self._entries.get(box)
        if entry is None:
            self.misses += 1
            return None
        prev_gray, zone, mask, _ = entry
        gray = self._gray(crop)
        if gray.shape != prev_gray.shape:
            self.misses += 1
            return None
        now = gray[zone] if zone.any() else gray.ravel()
        was = prev_gray[zone] if zone.any() else prev_gray.ravel()
        # Compare shapes, not absolute levels: a fade, an exposure change or a
        # lighting shift moves every pixel together without touching the text,
        # and should not invalidate the mask. Removing each sample's median
        # cancels that common offset while leaving glyph changes intact.
        now = now.astype(np.float32) - float(np.median(now))
        was = was.astype(np.float32) - float(np.median(was))
        score = float(np.abs(now - was).mean())
        if score > self.tolerance:
            self.misses += 1
            return None
        self._entries[box] = (prev_gray, zone, mask, frame_idx)
        self.hits += 1
        return mask

    def put(self, box: Box, crop: np.ndarray, mask: np.ndarray, frame_idx: int) -> None:
        self._entries[box] = (self._gray(crop).copy(), self._compare_zone(mask), mask, frame_idx)
        stale = [k for k, entry in self._entries.items() if frame_idx - entry[-1] > self.ttl]
        for key in stale:
            del self._entries[key]


def detect_pass(cfg: PipelineConfig) -> List[List[Region]]:
    """Pass 1 - OCR the clip and return a smoothed timeline of text regions.

    Runs in a subprocess: Paddle cannot share a process with torch on Windows
    (see _paddle_env). Results are cached to --timeline when given, so that
    re-runs which only change compositing settings skip detection entirely.
    """
    result = None
    if cfg.timeline_path and os.path.isfile(cfg.timeline_path):
        with open(cfg.timeline_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if regions.timeline_is_current(cached):
            result = cached
            print(f"  reusing cached timeline from {cfg.timeline_path}")
        else:
            print(f"  cached timeline at {cfg.timeline_path} is from an older "
                  f"format; detecting again")
    if result is None:
        request = {
            "rgb_path": os.path.abspath(cfg.rgb_path),
            "detector": asdict(cfg.detector),
            "region": asdict(cfg.region),
            "ocr_stride": cfg.ocr_stride,
            "max_frames": cfg.max_frames,
        }
        result = _run_detect_worker(request)
        if cfg.timeline_path:
            os.makedirs(os.path.dirname(os.path.abspath(cfg.timeline_path)), exist_ok=True)
            with open(cfg.timeline_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh)

    raw = result["regions"]
    timeline = [[regions.region_from_json(e) for e in frame] for frame in raw]

    # Runs rejected by eye, either in the tuning window (stored in the
    # timeline) or on the command line.
    excluded = set(result.get("excluded", [])) | set(cfg.exclude_runs)
    if excluded:
        before = sum(1 for f in timeline if f)
        timeline = [[r for r in f if r.run not in excluded] for f in timeline]
        after = sum(1 for f in timeline if f)
        print(f"  excluding run(s) {sorted(excluded)}: {before - after} fewer text frames")
    print(
        f"  text detected in {result['raw_text_frames']}/{result['n_frames']} frames "
        f"-> {result['text_frames']} after temporal smoothing "
        f"(dropping tracks shorter than {result.get('min_track', '?')} frames)"
    )
    return timeline


def _run_detect_worker(request: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="dcsubfixer-") as tmp:
        req_path = os.path.join(tmp, "request.json")
        res_path = os.path.join(tmp, "result.json")
        with open(req_path, "w", encoding="utf-8") as fh:
            json.dump(request, fh)

        cmd = [sys.executable, "-m", "dcsubfixer.detect_worker", req_path, res_path]
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if proc.returncode != 0:
            raise RuntimeError(
                f"detection subprocess failed with exit code {proc.returncode}. "
                "Its output is above."
            )
        with open(res_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def render_pass(
    cfg: PipelineConfig,
    timeline: List[List[Region]],
    rgb_info: video.VideoInfo,
    depth_info: video.VideoInfo,
    alignment: geometry.Alignment,
) -> None:
    """Pass 2 - segment glyphs where text lives and repaint the depth frames."""
    model = hisam.build_hisam(
        *hisam.resolve_checkpoints(cfg.models_dir, cfg.model_type, cfg.hisam_checkpoint),
        model_type=cfg.model_type,
        device=cfg.device,
    )
    segmenter = hisam.StrokeSegmenter(model, device=cfg.device, config=cfg.segmenter)
    cache = MaskCache(tolerance=cfg.cache_tolerance)

    if cfg.debug_dir:
        os.makedirs(cfg.debug_dir, exist_ok=True)
    debug_written = 0

    pix_fmt = video.depth_format(depth_info)
    dh, dw = depth_info.height, depth_info.width
    max_value = (1 << depth_info.bit_depth) - 1

    rgb_frames = video.read_frames(cfg.rgb_path)
    depth_frames = video.read_depth(cfg.depth_path, pix_fmt)
    prob_buffer = np.zeros((rgb_info.height, rgb_info.width), dtype=np.float32)

    n_text_frames = sum(1 for b in timeline if b)
    total = len(timeline)
    segmentations = 0

    with video.DepthWriter(
        cfg.out_path,
        dw,
        dh,
        depth_info.fps,
        pix_fmt,
        codec=cfg.codec,
        quality=cfg.quality,
        preset=cfg.preset,
    ) as writer:
        bar = tqdm(total=total, desc="pass 2/2  segment+composite", unit="f")
        idx = -1
        for idx, (rgb, planar) in enumerate(zip(rgb_frames, depth_frames)):
            if idx >= total:
                break
            frame_regions = timeline[idx]
            if not frame_regions:
                # Written back untouched, so it re-encodes bit-for-bit.
                writer.write(planar)
                bar.update(1)
                continue
            depth = video.luma(planar, dh, dw)

            prob_buffer[:] = 0.0
            for region in frame_regions:
                x0, y0, x1, y1 = region.box
                crop = rgb[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                mask = cache.get(region.box, crop, idx)
                if mask is None:
                    mask = segmenter.segment(crop)
                    segmentations += 1
                    cache.put(region.box, crop, mask, idx)
                # Cached raw, filtered on use, so the gate stays adjustable
                # without re-running Hi-SAM.
                if cfg.gate:
                    mask = regions.gate_by_detections(
                        mask, region, cfg.gate_dilate, cfg.gate_min_inside
                    )
                np.maximum(prob_buffer[y0:y1, x0:x1], mask, out=prob_buffer[y0:y1, x0:x1])

            depth_prob = alignment.warp(prob_buffer)
            depth_boxes = [alignment.map_box(r.box) for r in frame_regions]
            region_mask = None
            if cfg.composite.heal:
                region_mask = comp.stack_region_masks((dw, dh), depth_boxes)

            out = comp.composite_frame(
                depth, depth_prob, cfg.composite, depth_boxes, region_mask, max_value
            )
            # Only luma is replaced; the chroma planes ride along untouched.
            planar = planar.copy()
            planar[:dh, :dw] = out
            writer.write(planar)

            if cfg.debug_dir and debug_written < cfg.debug_limit:
                _write_debug(
                    cfg.debug_dir, idx, rgb, frame_regions, depth, out, depth_prob, max_value
                )
                debug_written += 1

            bar.update(1)

        # A depth video longer than the RGB clip keeps its tail untouched.
        for planar in depth_frames:
            writer.write(planar)
            idx += 1
        bar.close()

    hit_rate = cache.hits / max(1, cache.hits + cache.misses)
    print(
        f"  segmented {segmentations} regions across {n_text_frames} text frames "
        f"(mask cache hit rate {hit_rate:.0%})"
    )


def _write_debug(
    debug_dir: str,
    idx: int,
    rgb: np.ndarray,
    frame_regions: List[Region],
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    prob: np.ndarray,
    max_value: int,
) -> None:
    annotated = rgb.copy()
    for x0, y0, x1, y1 in (r.box for r in frame_regions):
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
    h = depth_before.shape[0]
    scale = h / annotated.shape[0]
    annotated = cv2.resize(annotated, (int(annotated.shape[1] * scale), h))

    # Depth maps often use only part of the available range, so stretch the
    # two panels over a shared window rather than assuming full scale.
    lo = float(min(depth_before.min(), depth_after.min()))
    hi = float(max(depth_before.max(), depth_after.max()))
    span = max(hi - lo, 1.0)

    def as_rgb8(plane: np.ndarray) -> np.ndarray:
        norm = np.clip((plane.astype(np.float32) - lo) / span * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    mask_vis = cv2.cvtColor((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    panel = np.concatenate(
        [annotated, as_rgb8(depth_before), mask_vis, as_rgb8(depth_after)], axis=1
    )
    cv2.imwrite(os.path.join(debug_dir, f"frame_{idx:06d}.png"), panel[:, :, ::-1])


def _setup(cfg: PipelineConfig):
    rgb_info = video.probe(cfg.rgb_path)
    depth_info = video.probe(cfg.depth_path)
    print(f"RGB   : {rgb_info}  {cfg.rgb_path}")
    print(f"depth : {depth_info}  {cfg.depth_path}")

    if cfg.align == "auto":
        samples = geometry.sample_frames(cfg.depth_path, count=6)
        alignment, note = geometry.detect_alignment(
            (rgb_info.width, rgb_info.height), (depth_info.width, depth_info.height), samples
        )
        print(f"align : {alignment.describe()}\n        {note}")
    else:
        alignment = geometry.build_alignment(
            (rgb_info.width, rgb_info.height), (depth_info.width, depth_info.height), cfg.align
        )
        print(f"align : {alignment.describe()} (forced)")

    pix_fmt = video.depth_format(depth_info)
    if pix_fmt != depth_info.pix_fmt:
        print(
            f"  note: {depth_info.pix_fmt} cannot be passed through directly; "
            f"working in {pix_fmt}. Untouched pixels may shift slightly."
        )
    first = next(iter(video.read_depth(cfg.depth_path, pix_fmt)))
    if not video.chroma_is_neutral(first, depth_info.height, depth_info.bit_depth):
        print(
            "  warning: the depth video carries colour. Only luma is edited, so any "
            "colour is preserved but the text depth is read from luma alone."
        )

    if rgb_info.n_frames and depth_info.n_frames and rgb_info.n_frames != depth_info.n_frames:
        print(
            f"  note: frame counts differ ({rgb_info.n_frames} RGB vs {depth_info.n_frames} depth); "
            "processing the overlap and passing any depth tail through untouched"
        )
    return rgb_info, depth_info, alignment


def probe_only(cfg: PipelineConfig) -> None:
    """Report geometry without touching either model."""
    _setup(cfg)


def run(cfg: PipelineConfig) -> None:
    rgb_info, depth_info, alignment = _setup(cfg)

    timeline = detect_pass(cfg)
    if not any(timeline):
        print("No text detected anywhere in the clip; nothing to composite.")
    render_pass(cfg, timeline, rgb_info, depth_info, alignment)
    print(f"wrote : {cfg.out_path}")
