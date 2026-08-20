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

There is a CLI for batch work and a [tuning window](#the-tuning-window) for
choosing settings by eye.

## Install

Needs a CUDA GPU. Developed against an RTX 5080 (SM 12.0) on CUDA 13, Python
3.12, Windows.

On Windows, run **`install.bat`** (or `install.ps1` directly). It reads the CUDA
version your driver supports, picks the matching torch and paddle wheel
channels, builds `.venv`, clones Hi-SAM, fetches **both** checkpoints, and
verifies the result — including running a convolution through Paddle in a
subprocess, because a wrong cuDNN DLL path stays invisible until one is
attempted.

Checkpoints are downloaded to a `.part` file and only put in place once their
size and SHA-256 match. That is not ceremony: the Hi-SAM weights come from a
Google Drive mirror, and Drive answers a plain share link for a file that size
with an HTML "could not scan for viruses" page — 75 KB of markup that saves
perfectly happily under a `.pth` name and only fails much later, looking like a
corrupt checkpoint.

```
install.bat                     # detect everything
install.bat -Force              # rebuild .venv from scratch
install.bat -SkipModels         # skip the 1.2 GB checkpoint download
install.bat -TorchIndex https://download.pytorch.org/whl/cu126 `
            -PaddleIndex https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

If a download fails or fails verification, the script says which checkpoint and
where to fetch it by hand, and carries on rather than leaving a half-written
file behind.

To do it by hand instead, clone Hi-SAM into `third_party/Hi-SAM` and follow the
install block at the top of [requirements.txt](requirements.txt) — the torch and
paddle wheels come from their own package indexes, so the order matters.

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

## The tuning window

The compositing settings are the ones no default gets right for every clip, and
the only way to choose them is to look at a real frame. That is what the GUI is
for:

```bash
python -m dcsubfixer --gui
```

It takes the two paths optionally, so `python -m dcsubfixer --gui rgb.mp4 depth.mp4`
opens straight into a clip.

On Windows, **`dc-sub-fixer.bat`** does the same without activating anything
first — double-click it, make a desktop shortcut to it, or drop an RGB clip and
its depth map onto it to open that pair. It keeps a console window open behind
the GUI on purpose: if something goes wrong, that is where the error appears.

Four panes over one shared zoom — RGB with the detected regions, depth before,
the glyph mask, depth after — a timeline strip marking every run of text, and
compositing controls that update the visible frame as you drag them.

| key | |
| --- | --- |
| `←` `→` | step a frame |
| `N` | jump to the next run of text |
| `1`–`4` | blow one pane up to fill the window |
| `Space` | flip between before and after, at the same zoom |
| `X` | exclude the selected run of text, or restore it |
| `0` / `F` | show all four / fit |

The reason it feels immediate is that segmentation is cached to disk per frame
and region. Moving a slider re-composites from the stored mask — array
arithmetic, about 20ms — and no GPU work happens until you visit a frame nobody
has segmented yet, which costs about 0.2s. The cache lives under
`~/.dcsubfixer/cache/` keyed to the clip pair, so it survives restarts.

A stored timeline records the format it was written in. When that does not
match, it is refused and detection runs again rather than the old file being
half-understood — which is the failure this project kept hitting: a timeline
written before regions carried their detections still loaded, still had boxes,
and quietly turned mask filtering off.

Closing the window clears the cached masks for that clip, which is where
essentially all of its size is. They are pure GPU output and rebuild
identically, so the cost is only the time to re-segment a frame when you next
look at it. The detections and any runs you excluded are kept, because that
half is small and holds decisions that cannot be recreated by running something
again — tick the second box to discard those too, or untick the first to keep
everything.

Detection runs as a subprocess because it has to (Paddle and torch cannot share
a process). The final render runs as one by choice: **it shells out to the same
CLI you could type yourself**, so what you tuned is exactly what you get. The
window shows that command, ready to copy for batch use.

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

**Level text, and judging a run rather than a frame.** Two more filters act
before that. Captions and credits are set level, so anything more than
`--max-tilt` degrees off horizontal is rejected: on the credit clip, real
credits measure within 0.7 degrees while the scene text around them has a
median tilt of 6 and a 90th percentile of 45. The angle comes from the
detector's own quadrilateral, which the pipeline previously discarded on
arrival by reducing it to an upright box.

The other is about *when* confidence is judged. Per frame, credits and scene
text genuinely overlap — on that clip the credits' 10th percentile is 0.84 and
the worst piece of scene text reaches 0.92 — so a threshold strict enough to
reject the scene text also eats the fade at either end of every real caption.
Whether something is text does not change from frame to frame, so the question
is asked once per *run*: a run is kept when its best frame clears
`--track-score`, and all of its frames inherit that verdict. The detector's own
floor (`--det-floor`) stays low, since a frame that is never detected cannot be
rescued later.

That recovers 174 frames of faded credit that a per-frame 0.85 threshold cut,
while still rejecting the newspaper. It is also not a knife edge: every
surviving run peaks above 0.85, so anything from 0.6 to 0.85 gives the same
answer on that clip.

**Text that moves.** A caption is pinned to the frame; scene text rides on
whatever is carrying it. Each run's typical movement between frames, in units
of its own text height, is measured before the boxes are canonicalised — the
step that would otherwise flatten a run to one box and erase the evidence.

Two details matter. It is the *median* step, not the total travel: a still
caption's box jumps for a frame or two whenever the detector merges a
neighbouring line, which on the credit clip put stationary credits at up to 2.0
text heights of total spread, overlapping the scene text completely. And it
measures translation — the part of the change both edges share — rather than
the centre, because a still caption whose box flickers one grid step taller
moves its centre by half a step, which at caption sizes is indistinguishable
from real movement.

Measured that way, every credit on the clip reads exactly 0.000 and every piece
of handheld text 0.037 or more. A scrolling credit roll still passes: boxes are
grid-snapped first, so a drift of a few pixels a frame stays in the same cell
most frames and the median step is zero. Raise `--max-motion` if a fast roll is
being cut.

**Glyph weight and edge.** `--dilate` thickens or thins the glyphs and
`--feather` softens their edge; both default to 0.70, chosen by eye on real
footage. Dilation is in depth-space pixels and is fractional on purpose:
morphology comes in whole pixels, one of those is two in the source, and on
thin text that is the difference between a hairline and a slab.

Feather is not the same knob as the `--mask-low`/`--mask-high` window. That
window reshapes how stroke *probability* becomes coverage, and does nothing at
all once `--binary` has thrown the soft values away; feather is a spatial blur,
so it softens the boundary whatever produced it.

**Reading the text back.** The filters above all infer text from how it looks
or behaves. The last one simply reads it: PP-OCRv6's recogniser runs on each
surviving run and it is kept only if the result is real text. On the credit clip
the runs come back as `PRESENTS` (1.00), `SUPERGIRL` (1.00), `MILLY ALCOCK`
(1.00), `EVE RIDLEY` (0.99) — and the two survivors the other filters had missed
read `UmazuA` (0.38) and a lone `目` (0.75). The first fails on score, the second
on `--rec-min-chars`, since a single character is not a caption however
confidently it is read.

This is the only check that does not care how the shot was made, which is why
it is worth a second model. It is affordable because it runs **once per run**,
on that run's most confident frame — a handful of recognitions for a whole clip,
not one per frame. It also reads the detections rather than the region outline,
whose empty corners would only add noise.

Real credits score 0.89 and up here against 0.38 for gibberish, so the 0.55
default sits in a wide gap; 0.9 starts cutting real ones, because a stylised
face gets misread (`MATTHIAS SCHOENAERTS` comes back as `FATCHIAS SLHO-NAERT`
at 0.89) while still being obviously a caption.

**Scene text inside a caption's outline.** Those two filters both act on whole
detections, and there is a third case they cannot see. A region is a rectangle
around one or more lines, and a rectangle drawn around two lines of different
widths encloses corners that hold neither — where a label on a prop or a sign
in the background may be sitting. Hi-SAM segments every glyph in what it is
given, correctly, and that intruder lands in the depth map looking like part of
the credit.

Nothing about detection can fix it: on the frame this was found, the intruder
was never detected at all, the two credit lines merge into one region at any
`--merge-gap` because their boxes overlap, and no `--roi` separates text from a
rectangle it sits inside.

So each region carries the detections it was built from, and the mask is
filtered back down to them — by connected component, not pixelwise, since
clipping a blob in half just leaves a fragment that looks like a broken glyph.
On that frame it removed all 683 intruding pixels and cost 3 of 18,668 real
ones. Tune with `--gate-dilate` (slack around each detection, for descenders
and tight boxes) and `--gate-min-inside`; `--no-gate` restores the old
behaviour.

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

**Text that is simply not a caption.** Some false positives are real, legible,
level, stationary, confidently-read text — a product label, a sign, a graphic
on a screen in shot. Every check above asks "is this text?" and the honest
answer is yes, so none of them helps.

For those, reject the run by eye. Each surviving run of text gets an id, drawn
on its box in the RGB pane; pick it from the list beside the transport and press
**X**. The choice is written into the timeline, so the CLI render honours it and
it survives a restart, and `--exclude-runs 0,2` does the same headlessly. On a
clip whose set is full of packaging, this took a scene of six runs down to the
one that was a credit.

## Tuning

Start here — this was the best-looking combination on the demo clip, noticeably
crisper than the defaults:

```bash
python -m dcsubfixer rgb.mp4 depth.mp4 out.mp4 --heal --mask-low 0.45 --mask-high 0.6
```

| symptom | try |
| --- | --- |
| text missed entirely | lower `--track-score`, then `--det-floor`; raise `--det-limit-side-len` |
| a caption's fade in/out is dropped | lower `--det-floor` — `--track-score` cannot rescue what was never detected |
| slanted or stylised titles rejected | raise `--max-tilt`, or 90 to disable |
| non-text picked up (panels, textures, faces) | raise `--track-score`, tighten `--max-tilt` or `--max-motion`, raise `--min-track`, or `--roi 0,0.7,1,1` |
| a fast credit roll is dropped | raise `--max-motion` |
| a stylised title is rejected | lower `--rec-score`, or 0 to skip recognition |
| scene text appearing beside a caption | already filtered; if some survives, raise `--gate-min-inside` or lower `--gate-dilate` |
| glyph edges or descenders clipped | raise `--gate-dilate`, or `--no-gate` |
| glyphs still look soft | `--heal`, and narrow the window to `--mask-low 0.45 --mask-high 0.6` |
| smeared halo remains around the text | `--heal`, raise `--heal-radius` |
| glyphs too thin or too fat | adjust `--dilate` (default 0.70). It is in *depth* pixels, so a whole one is two in the source; past about 1.0 the letters start to merge |
| glyph edges too crisp, or too soft | adjust `--feather` (default 0.70). A spatial blur, so unlike the mask window it works on `--binary` too |
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
