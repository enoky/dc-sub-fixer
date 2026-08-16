"""Command line entry point for dc-sub-fixer."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

from . import composite as comp
from . import geometry, hisam, ocr, pipeline, regions


def _roi(value: str) -> Tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated fractions: x0,y0,x1,y1")
    if not all(0.0 <= p <= 1.0 for p in parts):
        raise argparse.ArgumentTypeError("ROI values are fractions of the frame, so must be in 0..1")
    if parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise argparse.ArgumentTypeError("ROI must satisfy x0<x1 and y0<y1")
    return (parts[0], parts[1], parts[2], parts[3])


def _quality(value: str) -> str:
    if value.lower() == "lossless":
        return "lossless"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--quality takes 'lossless' or a CRF integer")
    if not 0 <= n <= 51:
        raise argparse.ArgumentTypeError("CRF must be between 0 and 51")
    return str(n)


def _text_value(value: str) -> str:
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--text-value takes 'auto' or an integer 0-255")
    if not 0 <= n <= 255:
        raise argparse.ArgumentTypeError("--text-value must be between 0 and 255")
    return str(n)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dc-sub-fixer",
        description=(
            "Restore sharp text glyphs in a DepthCrafter depth map. Detects text in the "
            "RGB source with PP-OCRv6, cuts pixel-level glyph masks with Hi-SAM, and "
            "composites them onto the matching depth frames."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("rgb", help="RGB source clip")
    p.add_argument("depth", help="DepthCrafter depth map video for the same clip")
    p.add_argument("output", nargs="?", help="output depth video (required unless --probe)")
    p.add_argument("--gui", action="store_true",
                   help="open the tuning window instead of rendering; the two video "
                        "arguments are optional when it is given")

    g = p.add_argument_group("models")
    g.add_argument("--models-dir", default="models", help="directory holding the .pth checkpoints")
    g.add_argument("--model-type", default="vit_l", choices=["vit_b", "vit_l", "vit_h"])
    g.add_argument("--hisam-checkpoint", default=None, help="override the sam_tss_*.pth path")
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("geometry")
    g.add_argument(
        "--align",
        default="auto",
        choices=["auto", *geometry.MODES],
        help="how the RGB frame maps onto the depth frame",
    )
    g.add_argument(
        "--roi",
        type=_roi,
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="only consider text whose centre falls in this fraction of the frame, "
        "e.g. 0,0.7,1,1 for subtitles in the bottom third",
    )

    g = p.add_argument_group("text detection (PP-OCRv6)")
    g.add_argument("--det-model", default=ocr.DEFAULT_DET_MODEL)
    g.add_argument("--det-limit-side-len", type=int, default=1280,
                   help="frames are scaled so the long side is at most this before detection")
    g.add_argument("--det-thresh", type=float, default=0.3)
    g.add_argument("--det-floor", type=float, default=0.3,
                   help="confidence below which the detector discards a box outright. "
                        "Kept low on purpose: weak frames of a real caption are rescued "
                        "by --track-score, which cannot see what was never detected")
    g.add_argument("--track-score", type=float, default=0.60,
                   help="a run of text is kept when its *best* frame reaches this. "
                        "Judging frames individually forces the bar high enough to "
                        "survive a caption's weakest moment, which then cuts the fades "
                        "off every real one")
    g.add_argument("--max-motion", type=float, default=0.02,
                   help="reject a run whose box typically moves more than this "
                        "fraction of its own text height per frame. Captions are "
                        "pinned to the frame; scene text is not. Raise it if a fast "
                        "credit roll is being cut")
    g.add_argument("--max-tilt", type=float, default=4.0,
                   help="reject detections more than this many degrees off horizontal. "
                        "Captions are set level; scene text rarely is")
    g.add_argument("--det-unclip-ratio", type=float, default=1.8)
    g.add_argument("--det-batch-size", type=int, default=8)
    g.add_argument("--ocr-stride", type=int, default=1,
                   help="run detection every Nth frame and hold the result between")

    g = p.add_argument_group("regions and temporal consistency")
    g.add_argument("--pad", type=int, default=12, help="context pixels kept around text")
    g.add_argument("--grid", type=int, default=16, help="snap region edges to this grid")
    g.add_argument("--merge-gap", type=int, default=24, help="merge regions closer than this")
    g.add_argument("--min-height", type=int, default=8, help="ignore text shorter than this")
    g.add_argument("--min-track", type=int, default=None,
                   help=f"drop text present in fewer than this many frames "
                        f"(default: {regions.MIN_TRACK_SECONDS}s worth, from the frame rate)")
    g.add_argument("--max-gap", type=int, default=3,
                   help="bridge detection dropouts up to this many frames long")
    g.add_argument("--sticky-iou", type=float, default=0.8,
                   help="hold a region box steady while it overlaps its run this much")
    g.add_argument("--no-gate", dest="gate", action="store_false",
                   help="keep every glyph Hi-SAM finds inside a region, including any "
                        "scene text that happens to fall in the same rectangle")
    g.add_argument("--gate-dilate", type=int, default=8,
                   help="pixels of slack around each detection when filtering the mask")
    g.add_argument("--gate-min-inside", type=float, default=0.5,
                   help="fraction of a stroke blob that must lie on a detection for it "
                        "to be kept (0 disables filtering)")
    g.add_argument("--cache-tolerance", type=float, default=2.0,
                   help="mean pixel difference below which a region counts as unchanged "
                        "and its mask is reused")

    g = p.add_argument_group("glyph segmentation (Hi-SAM)")
    g.add_argument("--tile", type=int, default=1024,
                   help="max region size fed to the encoder; larger regions are tiled")
    g.add_argument("--overlap", type=float, default=0.25, help="tile overlap fraction")
    g.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])

    g = p.add_argument_group("compositing")
    g.add_argument("--text-value", type=_text_value, default="auto",
                   help="grey level for glyphs: 'auto' reads it back from the depth map, "
                        "or give 0-255")
    g.add_argument("--mask-low", type=float, default=0.35,
                   help="stroke probability mapped to fully transparent")
    g.add_argument("--mask-high", type=float, default=0.65,
                   help="stroke probability mapped to fully opaque")
    g.add_argument("--binary", action="store_true", help="hard-edged mask instead of feathered")
    g.add_argument("--dilate", type=int, default=0,
                   help="thicken glyphs by N pixels, or thin them with a negative value")
    g.add_argument("--opacity", type=float, default=1.0, help="blend strength of the glyphs")
    g.add_argument("--heal", action="store_true",
                   help="inpaint DepthCrafter's jagged halo around the text before "
                        "repainting the glyphs")
    g.add_argument("--heal-radius", type=int, default=6)

    g = p.add_argument_group("output")
    g.add_argument("--codec", default="libx264", choices=["libx264", "libx265", "ffv1"])
    g.add_argument("--quality", type=_quality, default="lossless",
                   help="'lossless' (default) or a CRF number. The depth map is data for a "
                        "downstream 3D stage, and a lossy re-encode disturbs every pixel in "
                        "the frame, including those this tool never touches")
    g.add_argument("--preset", default="slow")

    g = p.add_argument_group("misc")
    g.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    g.add_argument("--timeline", default=None,
                   help="cache detection results here; reused on later runs")
    g.add_argument("--debug-dir", default=None,
                   help="write side-by-side RGB / depth / mask / result panels here")
    g.add_argument("--debug-limit", type=int, default=12)
    g.add_argument("--probe", action="store_true",
                   help="print video geometry and the detected alignment, then exit")

    return p


def config_from_args(args: argparse.Namespace) -> pipeline.PipelineConfig:
    return pipeline.PipelineConfig(
        rgb_path=args.rgb,
        depth_path=args.depth,
        out_path=args.output,
        models_dir=args.models_dir,
        model_type=args.model_type,
        hisam_checkpoint=args.hisam_checkpoint,
        device=args.device,
        align=args.align,
        ocr_stride=args.ocr_stride,
        cache_tolerance=args.cache_tolerance,
        gate=args.gate,
        gate_dilate=args.gate_dilate,
        gate_min_inside=args.gate_min_inside,
        max_frames=args.max_frames,
        timeline_path=args.timeline,
        debug_dir=args.debug_dir,
        debug_limit=args.debug_limit,
        codec=args.codec,
        quality=args.quality,
        preset=args.preset,
        detector=ocr.DetectorConfig(
            model_name=args.det_model,
            device="gpu:0" if args.device.startswith("cuda") else "cpu",
            limit_side_len=args.det_limit_side_len,
            thresh=args.det_thresh,
            box_thresh=args.det_floor,
            unclip_ratio=args.det_unclip_ratio,
            batch_size=args.det_batch_size,
        ),
        region=regions.RegionConfig(
            pad=args.pad,
            grid=args.grid,
            merge_gap=args.merge_gap,
            min_height=args.min_height,
            max_tilt=args.max_tilt,
            max_motion=args.max_motion,
            track_score=args.track_score,
            roi=args.roi,
            min_track=args.min_track,
            max_gap=args.max_gap,
            sticky_iou=args.sticky_iou,
        ),
        segmenter=hisam.SegmenterConfig(
            tile=args.tile,
            overlap=args.overlap,
            precision=args.precision,
        ),
        composite=comp.CompositeConfig(
            text_value=args.text_value,
            mask_low=args.mask_low,
            mask_high=args.mask_high,
            binary=args.binary,
            dilate=args.dilate,
            heal=args.heal,
            heal_radius=args.heal_radius,
            opacity=args.opacity,
        ),
    )


def main(argv: Optional[list] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--gui" in raw:
        from .gui import main as gui_main

        rest = [a for a in raw if a != "--gui"]
        return gui_main([sys.argv[0]] + rest)

    parser = build_parser()
    args = parser.parse_args(argv)

    for path, label in ((args.rgb, "RGB"), (args.depth, "depth")):
        if not os.path.isfile(path):
            parser.error(f"{label} video not found: {path}")
    if not args.probe and not args.output:
        parser.error("an output path is required unless --probe is given")

    cfg = config_from_args(args)

    if args.probe:
        pipeline.probe_only(cfg)
        return 0

    pipeline.run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
