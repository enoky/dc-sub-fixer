# dc-sub-fixer

DepthCrafter produces temporally consistent depth maps, but it mangles text.
Subtitles and title cards come back as jagged, blurred blobs, because the depth
model works at a lower resolution than the source and text is exactly the kind
of thin high-frequency detail that does not survive that.

This tool repairs them. It finds the text in the **RGB source**, where the
glyphs are still pristine, cuts a pixel-level stroke mask, and composites that
mask back onto the depth frames at the depth the text is supposed to sit at.

```
RGB frame ──> PP-OCRv6 detect ──> text regions ──> Hi-SAM stroke masks
                                                          │
depth frame <── composite <── align to depth resolution <──┘
```

## Install

Needs a CUDA GPU. Developed against an RTX 5080 (SM 12.0) on CUDA 13, Python
3.12, Windows.

```bash
git clone https://github.com/ymy-k/Hi-SAM.git third_party/Hi-SAM
```

Then follow the install block at the top of [requirements.txt](requirements.txt) —
the torch and paddle wheels come from their own package indexes, so the order
matters.

Put the checkpoints in `models/`:

| file | what it is | from |
| --- | --- | --- |
| `sam_tss_l_textseg.pth` | Hi-SAM text stroke segmentation, ViT-L | [Hi-SAM](https://github.com/ymy-k/Hi-SAM) |
| `sam_vit_l_0b3195.pth` | base SAM ViT-L backbone | [segment-anything](https://github.com/facebookresearch/segment-anything) |

Both are required: the Hi-SAM checkpoint holds only the trained adapters, modal
aligner and mask decoder, and the frozen backbone weights are merged in from the
SAM checkpoint at load time.

PP-OCRv6 downloads itself on first run.

## Use

```bash
python -m dcsubfixer clip_rgb.mp4 clip_depth.mp4 clip_depth_fixed.mp4
```

Check how the two videos line up without running any models:

```bash
python -m dcsubfixer clip_rgb.mp4 clip_depth.mp4 --probe
```

Tune against real footage by writing inspection panels — RGB with detected
regions, depth before, the glyph mask, depth after — as PNGs:

```bash
python -m dcsubfixer clip_rgb.mp4 clip_depth.mp4 out.mp4 --debug-dir ./debug --max-frames 60
```

Detection results can be cached, so re-runs that only change compositing
settings skip straight to the second pass:

```bash
python -m dcsubfixer clip_rgb.mp4 clip_depth.mp4 out.mp4 --timeline ./timeline.json
```

`python -m dcsubfixer --help` lists every option.

## How it handles the hard parts

**Resolution and aspect ratio.** The RGB and depth videos rarely share a shape.
`--align auto` (the default) compares aspect ratios; if they match it maps the
frames directly, and if they differ it looks for letterbox bars in the depth
video and picks a fitted mapping when the bar thickness agrees. It prints what
it chose and why. Override with `--align stretch|fit|fill` when the guess is
wrong — check it with `--probe` first.

**Stroke detail.** Text regions are cropped from the RGB at native resolution
and fed to Hi-SAM individually, so a subtitle is segmented at full detail rather
than being downscaled with the whole frame. Regions wider than the encoder's
1024px input are covered by overlapping tiles blended with a tapered weight.

**Temporal consistency.** Detections are collected over the whole clip first,
then linked into tracks. Isolated one-frame detections are dropped
(`--min-track`), brief dropouts are bridged (`--max-gap`), and each run of
overlapping boxes collapses to a single stable box (`--sticky-iou`) so the
region does not shimmer between two grid positions. Because a held subtitle then
gets one identical mask for its whole duration, the composited glyphs are exactly
stable rather than merely similar.

**Precision.** Depth frames are read, edited and written in the source's own
planar YUV format, and only the luma plane is touched. This is not fussiness:
DepthCrafter writes 10-bit, and routing that through 8-bit RGB discards three
quarters of its precision, while even a 16-bit grey round trip shifts pixels by
a whole 10-bit step through limited/full range conversion. Output is lossless
by default, so **frames containing no text come back bit-identical** to the
source. On the demo clip, 72 of 129 frames are exactly unchanged and the other
57 differ only where glyphs were painted. Pass `--quality 20` (or any CRF) if
you would rather have a small file than an exact one.

**Efficiency.** Frames with no text are copied through untouched. Where text is
present, a mask cache keyed on the region box skips re-segmenting text that has
not changed. Its freshness check compares only the pixels on and around the
cached glyphs, and ignores a uniform brightness offset, so a caption held over
moving footage stays a cache hit.

Note that a credit which *fades* in or out legitimately misses the cache on
every frame: the glyphs really are changing opacity, and reusing a
fully-opaque mask would paint text at full strength over a caption that is
barely visible. Expect a low hit rate on fades and a high one on held text.

**Avoiding damage.** The defaults lean toward doing nothing rather than doing
something wrong, because the two failure modes are not symmetric: a missed
caption leaves the depth exactly as DepthCrafter made it, while a false
positive paints over a region that was fine. Two independent filters do the
work — a detection confidence floor (`--det-box-thresh`), and a persistence
requirement (`--min-track`, defaulting to half a second's worth of frames,
since text has to stay up long enough to read). On the demo clip either one
alone removes every false positive without losing a true frame, so the pair
has margin: an aircraft wall panel that the detector read as text for seven
straight frames is cut by persistence at the default, and by confidence at
`--det-box-thresh 0.7`.

**Choosing the grey level.** `--text-value auto` (the default) reads the level
back out of the depth map, so the repaired text keeps the depth DepthCrafter
assigned it. It samples an extreme percentile rather than the median, because
the smearing that damaged the glyphs also mixed background depth into them and
would drag a median toward the background. Whether the text is brighter or
darker than its surroundings is measured, not assumed, so disparity maps (near =
bright) and true depth maps (near = dark) both work. Pass `--text-value 0..255`
to force a fixed level instead.

**The leftover halo.** Painting clean glyphs does not remove the blobby mess
DepthCrafter left *around* them. `--heal` inpaints that halo from the
surrounding depth before the glyphs go down. It is off by default because it
changes pixels beyond the glyphs themselves — try it with `--debug-dir` and
judge on your own footage.

## Tuning

Start here — this was the best-looking combination on the demo clip, noticeably
crisper than the defaults:

```bash
python -m dcsubfixer rgb.mp4 depth.mp4 out.mp4 --heal --mask-low 0.45 --mask-high 0.6
```

| symptom | try |
| --- | --- |
| text missed entirely | lower `--det-box-thresh`, raise `--det-limit-side-len` |
| non-text picked up (panels, textures, faces) | raise `--det-box-thresh` toward 0.8, raise `--min-track`, or `--roi 0,0.7,1,1` for subtitles only |
| glyphs still look soft | `--heal`, and narrow the window to `--mask-low 0.45 --mask-high 0.6` |
| smeared halo remains around the text | `--heal`, raise `--heal-radius` |
| glyphs too thin or too fat | `--dilate 1` / `--dilate -1` |
| edges aliased or crawling | widen `--mask-low` / `--mask-high` |
| text flickers on and off | raise `--max-gap`, lower `--min-track` |
| short or fast-moving text dropped | `--min-track 1` |
| too slow | raise `--ocr-stride`, lower `--det-limit-side-len` |

## Notes and limits

- Paddle and PyTorch bundle incompatible cuDNN 9 builds, and Windows resolves
  DLLs by base name across the whole process. They therefore cannot share one
  process: text detection runs in a subprocess
  ([detect_worker.py](dcsubfixer/detect_worker.py)) that never imports torch.
  This is invisible in normal use.
- Text that moves more than roughly a third of its own width per frame falls
  below the tracker's overlap threshold and is dropped by `--min-track`.
  Subtitles and scrolling credits are far slower than that; pass `--min-track 1`
  if you hit it.
- `--min-track` also drops text shown for less than a third of a second. That
  is the intended trade (see "Avoiding damage"), but it does mean very brief
  flash frames need `--min-track 1`.
- Only the luma plane is edited. Chroma rides along untouched, so a colourised
  depth map survives, but the text depth is read from luma alone.
- How much this helps depends on how badly DepthCrafter mangled the text in the
  first place. Where it produced a soft, smeared approximation the glyph edges
  come back cleanly; it cannot recover text the depth model dropped entirely,
  and it does not flatten a whole text band that DepthCrafter placed at the
  wrong depth — it repairs glyph *shape*, not the region's overall depth.
- Audio is not carried across — the output is a depth video, not a deliverable
  cut.

Run the tests with `python -m pytest tests`. They cover the geometry, region
tracking and compositing logic, and need neither a GPU nor the checkpoints.

## Licence

MIT, see [LICENSE](LICENSE). Hi-SAM, SAM and PaddleOCR carry their own licences.
This is a personal, non-commercial project.
