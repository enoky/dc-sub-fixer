"""OCR detection pass, run as its own process.

Paddle and PyTorch bundle incompatible cuDNN 9 builds and Windows resolves DLLs
by base name process-wide, so the two cannot coexist (see _paddle_env). The
pipeline is already split into a detect pass and a segment pass, so the detect
pass simply runs here, in a process that never imports torch, and hands back a
JSON timeline of text regions.

Invoked as:  python -m dcsubfixer.detect_worker <request.json> <result.json>

With --progress it also writes `PROGRESS <done> <total>` lines to stdout, which
the GUI parses to drive a progress bar. The flag is opt-in so the CLI's stdout
stays clean.
"""

from __future__ import annotations

import json
import sys
from typing import List

from tqdm import tqdm

from . import ocr, regions, video
from .regions import Region


def detect(request: dict, progress: bool = False) -> dict:
    rgb_path = request["rgb_path"]
    det_cfg = ocr.DetectorConfig(**request["detector"])
    reg_cfg = regions.RegionConfig(**request["region"])
    if reg_cfg.roi is not None:
        reg_cfg.roi = tuple(reg_cfg.roi)
    stride = max(1, int(request.get("ocr_stride", 1)))
    max_frames = request.get("max_frames")

    info = video.probe(rgb_path)
    frame_size = (info.width, info.height)
    total = min(info.n_frames, max_frames) if (info.n_frames and max_frames) else (max_frames or info.n_frames)

    detector = ocr.TextDetector(det_cfg)

    per_frame: List[List[Region]] = []
    batch, batch_idx = [], []
    last: List[Region] = []

    def flush() -> None:
        nonlocal last
        if not batch:
            return
        for idx, (polys, scores) in zip(batch_idx, detector.detect_batch(batch)):
            found = regions.frame_regions(polys, frame_size, reg_cfg, scores)
            while len(per_frame) < idx:
                per_frame.append(list(last))  # frames skipped by --ocr-stride
            per_frame.append(found)
            last = found
        batch.clear()
        batch_idx.clear()

    n_seen = 0
    with tqdm(total=total or None, desc="pass 1/2  detect", unit="f", file=sys.stderr) as bar:
        for idx, frame in enumerate(video.read_frames(rgb_path)):
            if max_frames and idx >= max_frames:
                break
            n_seen = idx + 1
            if idx % stride == 0:
                batch.append(frame)
                batch_idx.append(idx)
                if len(batch) >= det_cfg.batch_size:
                    flush()
            bar.update(1)
            if progress and total and idx % 8 == 0:
                print(f"PROGRESS {idx + 1} {total}", flush=True)
        flush()
    if progress:
        print(f"PROGRESS {n_seen} {total or n_seen}", flush=True)

    while len(per_frame) < n_seen:
        per_frame.append(list(last))

    min_track = regions.resolve_min_track(reg_cfg, float(info.fps))
    smoothed = regions.smooth_timeline(per_frame, reg_cfg, min_track=min_track)
    return {
        "rgb_path": rgb_path,
        "width": info.width,
        "height": info.height,
        "n_frames": len(smoothed),
        "raw_text_frames": sum(1 for b in per_frame if b),
        "text_frames": sum(1 for b in smoothed if b),
        "min_track": min_track,
        # Each region carries the detections it was built from, so the mask can
        # later be filtered back down to where text was actually found.
        "regions": [[regions.region_to_json(r) for r in items] for items in smoothed],
    }


def main(argv: List[str]) -> int:
    args = [a for a in argv[1:] if a != "--progress"]
    progress = "--progress" in argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(args[0], "r", encoding="utf-8") as fh:
        request = json.load(fh)
    result = detect(request, progress=progress)
    with open(args[1], "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
