"""Tests for the non-model parts of the pipeline: geometry, regions, compositing.

Deliberately model-free so they run without a GPU or the checkpoints.
"""

import numpy as np
import pytest

from dcsubfixer import composite as comp
from dcsubfixer import geometry, regions, session


# ---------------------------------------------------------------- geometry


def test_stretch_maps_corners_onto_corners():
    a = geometry.build_alignment((1920, 1080), (1024, 576), "stretch")
    assert a.map_box((0, 0, 1920, 1080)) == (0, 0, 1024, 576)


def test_fit_letterboxes_and_centres():
    a = geometry.build_alignment((1920, 1080), (1024, 768), "fit")
    assert a.sx == pytest.approx(a.sy)
    x0, y0, x1, y1 = a.map_box((0, 0, 1920, 1080))
    assert (x0, x1) == (0, 1024)
    assert y0 == 96 and y1 == 672  # (768 - 1080 * 1024/1920) / 2 == 96


def test_fill_crops_the_overhang():
    a = geometry.build_alignment((1920, 1080), (1024, 768), "fill")
    # Scaled to cover, the source overflows horizontally and is clamped.
    assert a.sx == pytest.approx(768 / 1080)
    assert a.map_box((0, 0, 1920, 1080))[0] == 0


def test_map_box_never_returns_an_empty_box():
    a = geometry.build_alignment((1920, 1080), (64, 36), "stretch")
    x0, y0, x1, y1 = a.map_box((10, 10, 11, 11))
    assert x1 > x0 and y1 > y0


def test_warp_moves_a_mark_to_the_expected_place():
    a = geometry.build_alignment((400, 200), (200, 100), "stretch")
    src = np.zeros((200, 400), np.float32)
    src[100:120, 200:240] = 1.0
    out = a.warp(src)
    assert out.shape == (100, 200)
    ys, xs = np.nonzero(out > 0.5)
    # rows 100:120 and cols 200:240 at half scale -> rows 50:60, cols 100:120
    assert 50 <= ys.min() <= 51 and 59 <= ys.max() <= 60
    assert 100 <= xs.min() <= 101 and 119 <= xs.max() <= 120


def test_detect_alignment_prefers_stretch_when_aspect_matches():
    a, note = geometry.detect_alignment((1920, 1080), (1024, 576))
    assert a.mode == "stretch"
    assert "match" in note


def test_detect_alignment_finds_letterbox_bars():
    frames = []
    for _ in range(4):
        f = np.zeros((768, 1024, 3), np.uint8)
        f[96:672] = 180  # 16:9 content inside a 4:3 frame
        frames.append(f)
    a, note = geometry.detect_alignment((1920, 1080), (1024, 768), frames)
    assert a.mode == "fit"
    assert a.ty == pytest.approx(96.0)


def test_detect_alignment_falls_back_to_stretch_without_bars():
    frames = [np.full((768, 1024, 3), 120, np.uint8) for _ in range(4)]
    a, note = geometry.detect_alignment((1920, 1080), (1024, 768), frames)
    assert a.mode == "stretch"
    assert "--align" in note


# ----------------------------------------------------------------- regions


def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_nearby_boxes_merge_and_distant_ones_do_not():
    assert len(regions.merge_boxes([(0, 0, 50, 20), (60, 0, 100, 20)], gap=24)) == 1
    assert len(regions.merge_boxes([(0, 0, 50, 20), (400, 0, 450, 20)], gap=24)) == 2


def test_stabilise_snaps_outward_to_the_grid():
    cfg = regions.RegionConfig(pad=10, grid=16)
    assert regions.stabilise((100, 100, 200, 140), (1920, 1080), cfg) == (80, 80, 224, 160)


def test_stabilise_clamps_to_the_frame():
    cfg = regions.RegionConfig(pad=40, grid=16)
    assert regions.stabilise((5, 5, 100, 60), (320, 180), cfg) == (0, 0, 144, 112)


def test_roi_filters_by_box_centre():
    cfg = regions.RegionConfig(roi=(0.0, 0.7, 1.0, 1.0))
    boxes = [(100, 100, 300, 140), (100, 900, 300, 940)]
    assert regions.filter_boxes(boxes, (1920, 1080), cfg) == [(100, 900, 300, 940)]


def test_tiny_detections_are_dropped():
    cfg = regions.RegionConfig(min_height=8, min_area=120)
    assert regions.filter_boxes([(0, 0, 40, 4)], (1920, 1080), cfg) == []


def test_short_tracks_are_dropped():
    box = (100, 100, 200, 140)
    timeline = [[], [box], [], [], []]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2))
    assert all(not boxes for boxes in out)


def test_brief_dropouts_are_bridged():
    box = (100, 100, 200, 140)
    timeline = [[box], [box], [], [box], [box]]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2, max_gap=3))
    assert all(boxes for boxes in out), "frame 2 should have been filled in"


def test_long_dropouts_are_not_bridged():
    box = (100, 100, 200, 140)
    timeline = [[box], [box]] + [[]] * 8 + [[box], [box]]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2, max_gap=3))
    assert not any(out[4:8])


def test_jittering_boxes_collapse_to_one_stable_box():
    # The same caption detected one grid step taller on alternate frames.
    timeline = [[(400, 896, 1152, 992)], [(400, 896, 1152, 1008)]] * 6
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2))
    produced = {r.box for items in out for r in items}
    assert len(produced) == 1, f"expected one stable box, got {produced}"
    # The union, so no frame's own detection gets cropped.
    assert produced == {(400, 896, 1152, 1008)}


def test_canonical_box_does_not_borrow_a_neighbours_detections():
    """Only the outline is held steady; the filter stays per-frame."""
    a = regions.Region((0, 0, 100, 50), ((10, 10, 40, 40),))
    b = regions.Region((0, 0, 100, 56), ((60, 10, 90, 40),))
    out = regions.smooth_timeline([[a], [b]] * 4, regions.RegionConfig(min_track=2))
    assert {r.box for items in out for r in items} == {(0, 0, 100, 56)}
    assert out[0][0].dets == ((10, 10, 40, 40),)
    assert out[1][0].dets == ((60, 10, 90, 40),)


def test_slowly_scrolling_text_is_tracked_but_not_frozen():
    # 40px a frame on a 300px-wide box: overlapping enough to stay one track,
    # too much movement to be treated as the same still box.
    timeline = [[(40 * i, 100, 300 + 40 * i, 140)] for i in range(5)]
    out = regions.smooth_timeline(
        timeline, regions.RegionConfig(min_track=2, max_motion=None))
    assert all(boxes for boxes in out), "scrolling text should not be dropped"
    produced = {r.box for items in out for r in items}
    assert len(produced) > 1, "a moving box must not be frozen to one position"


def test_text_moving_faster_than_the_tracker_is_dropped():
    """Documents a real limit: motion below the track IoU threshold is lost.

    Subtitles and scrolling credits move far slower than this, but text that
    jumps most of its own width between frames breaks tracking and is then cut
    by min_track. Lower --min-track to 1 to keep such detections.
    """
    timeline = [[(200 * i, 100, 300 + 200 * i, 140)] for i in range(5)]
    dropped = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2))
    assert not any(dropped)
    kept = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=1))
    assert all(boxes for boxes in kept)


def test_min_track_defaults_to_half_a_second():
    cfg = regions.RegionConfig()
    assert regions.resolve_min_track(cfg, 23.976) == 12
    assert regions.resolve_min_track(cfg, 60.0) == 30
    # Never below 2, however slow the clip.
    assert regions.resolve_min_track(cfg, 1.0) == 2


def test_explicit_min_track_overrides_the_frame_rate():
    assert regions.resolve_min_track(regions.RegionConfig(min_track=1), 60.0) == 1


def test_brief_false_positives_are_dropped_by_persistence():
    """Scene texture briefly read as text must not reach the compositor."""
    box = (300, 200, 500, 260)
    timeline = [[] for _ in range(30)]
    for i in range(10, 14):  # four frames only
        timeline[i] = [box]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(), min_track=7)
    assert not any(out)


def test_frame_regions_merges_words_into_one_line():
    polys = [_poly(100, 900, 200, 940), _poly(210, 900, 320, 940)]
    out = regions.frame_regions(polys, (1920, 1080), regions.RegionConfig())
    assert len(out) == 1
    x0, y0, x1, y1 = out[0].box
    assert x0 <= 100 and x1 >= 320
    # Both words survive as separate detections, never unioned into the outline.
    assert len(out[0].dets) == 2
    assert out[0].box not in out[0].dets


# -------------------------------------------------------------- composite


def test_alpha_window_maps_probabilities_to_a_soft_edge():
    cfg = comp.CompositeConfig(mask_low=0.35, mask_high=0.65)
    prob = np.array([[0.0, 0.35, 0.5, 0.65, 1.0]], np.float32)
    alpha = comp.probability_to_alpha(prob, cfg)
    assert alpha[0, 0] == 0.0
    assert alpha[0, 2] == pytest.approx(0.5)
    assert alpha[0, 4] == 1.0


def test_binary_mode_is_hard_edged():
    cfg = comp.CompositeConfig(binary=True, mask_high=0.65)
    prob = np.array([[0.5, 0.9]], np.float32)
    assert list(comp.probability_to_alpha(prob, cfg)[0]) == [0.0, 1.0]


def test_auto_value_recovers_bright_text_over_dark_background():
    depth = np.full((40, 120), 100 * 257, np.uint16)
    alpha = np.zeros((40, 120), np.float32)
    alpha[18:22, 20:100] = 1.0
    depth[18:22, 20:100] = 220 * 257
    value = comp.resolve_text_value(depth, alpha, comp.CompositeConfig(), 65535)
    assert value > 200 * 257


def test_auto_value_recovers_dark_text_over_bright_background():
    """A true-depth map puts near objects dark; the sampling must follow suit."""
    depth = np.full((40, 120), 200 * 257, np.uint16)
    alpha = np.zeros((40, 120), np.float32)
    alpha[18:22, 20:100] = 1.0
    depth[18:22, 20:100] = 30 * 257
    value = comp.resolve_text_value(depth, alpha, comp.CompositeConfig(), 65535)
    assert value < 60 * 257


def test_explicit_text_value_is_scaled_to_the_frame_range():
    depth = np.full((10, 10), 100 * 257, np.uint16)
    alpha = np.ones((10, 10), np.float32)
    cfg = comp.CompositeConfig(text_value="255")
    assert comp.resolve_text_value(depth, alpha, cfg, 65535) == pytest.approx(65535)
    assert comp.resolve_text_value(depth, alpha, cfg, 255) == pytest.approx(255)


def test_composite_leaves_a_textless_frame_untouched():
    depth = np.full((40, 120), 100 * 257, np.uint16)
    prob = np.zeros((40, 120), np.float32)
    out = comp.composite_frame(depth, prob, comp.CompositeConfig())
    assert out is depth or np.array_equal(out, depth)


def test_composite_paints_only_inside_the_mask():
    depth = np.full((40, 120), 100 * 257, np.uint16)
    prob = np.zeros((40, 120), np.float32)
    prob[10:14, 30:60] = 1.0
    out = comp.composite_frame(depth, prob, comp.CompositeConfig(text_value="255"))
    assert (out[10:14, 30:60] == 65535).all()
    assert (out[0:5] == 100 * 257).all()


def test_composite_preserves_16_bit_precision_outside_the_mask():
    """The whole point of the luma path: untouched pixels stay bit-exact."""
    rng = np.random.RandomState(0)
    depth = rng.randint(0, 65535, (40, 120), dtype=np.uint16)
    prob = np.zeros((40, 120), np.float32)
    prob[10:14, 30:60] = 1.0
    out = comp.composite_frame(depth, prob, comp.CompositeConfig(text_value="255"))
    assert out.dtype == np.uint16
    untouched = np.ones((40, 120), bool)
    untouched[10:14, 30:60] = False
    assert np.array_equal(out[untouched], depth[untouched])


# --------------------------------------------------- tilt and track scoring


def _quad(x0, y0, x1, y1, degrees=0.0):
    """A detector-style quad, clockwise from the top left, rotated about centre."""
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    c = pts.mean(axis=0)
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    return (pts - c) @ rot.T + c


def test_tilt_of_a_level_quad_is_zero():
    assert regions.poly_tilt(_quad(10, 10, 200, 50)) == pytest.approx(0.0, abs=1e-3)


def test_tilt_matches_the_rotation_applied():
    for deg in (3.0, 12.0, 30.0):
        assert regions.poly_tilt(_quad(10, 10, 200, 50, deg)) == pytest.approx(deg, abs=0.1)


def test_tilt_is_symmetric_about_horizontal():
    assert regions.poly_tilt(_quad(10, 10, 200, 50, -8.0)) == pytest.approx(8.0, abs=0.1)


def test_tilt_of_vertical_text_is_ninety():
    """Text turned on its side, i.e. the quad's own edge is vertical."""
    assert regions.poly_tilt(_quad(10, 10, 200, 50, 90.0)) == pytest.approx(90.0, abs=1e-3)


def test_tilt_reads_the_text_direction_not_the_box_shape():
    """A tall narrow box of level text is level; only the quad order says so.

    This is why the measure comes from the quad rather than its bounding box:
    a single wide letter and a column of stacked text share a bounding box.
    """
    tall_but_level = _quad(10, 10, 50, 200)  # first edge runs left to right
    assert regions.poly_tilt(tall_but_level) == pytest.approx(0.0, abs=1e-3)


def test_tilt_never_exceeds_ninety():
    """The measure folds to 0..90, so a near-vertical quad cannot read as small."""
    for deg in (0, 45, 89, 91, 135, 179, 200, 271):
        assert 0.0 <= regions.poly_tilt(_quad(10, 10, 200, 50, deg)) <= 90.0 + 1e-6


def test_tilted_detections_are_rejected():
    cfg = regions.RegionConfig(max_tilt=4.0)
    polys = [_quad(100, 100, 400, 150), _quad(100, 300, 400, 350, 15.0)]
    out = regions.frame_regions(polys, (1920, 1080), cfg, scores=[0.9, 0.9])
    assert len(out) == 1
    assert out[0].box[1] < 200, "the level detection should be the survivor"


def test_tilt_filter_can_be_disabled():
    cfg = regions.RegionConfig(max_tilt=90.0, merge_gap=0)
    polys = [_quad(100, 100, 400, 150), _quad(100, 300, 400, 350, 15.0)]
    assert len(regions.frame_regions(polys, (1920, 1080), cfg, scores=[0.9, 0.9])) == 2


def test_region_score_is_the_best_of_its_detections():
    polys = [_quad(100, 900, 200, 940), _quad(210, 900, 320, 940)]
    out = regions.frame_regions(polys, (1920, 1080), regions.RegionConfig(),
                                scores=[0.42, 0.88])
    assert out[0].score == pytest.approx(0.88)


def test_a_track_is_judged_by_its_best_frame_not_its_worst():
    """A caption fades in, so its first frames score no better than texture."""
    box = (400, 800, 900, 880)
    weak = regions.Region(box, (box,), 0.35)
    strong = regions.Region(box, (box,), 0.93)
    timeline = [[weak], [weak], [strong], [strong], [weak], [weak]]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2,
                                                                track_score=0.6))
    assert all(items for items in out), "the fade frames must survive with the peak"


def test_a_track_that_never_scores_well_is_dropped():
    box = (400, 800, 900, 880)
    weak = regions.Region(box, (box,), 0.42)
    timeline = [[weak] for _ in range(20)]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2,
                                                                track_score=0.6))
    assert not any(out)


def test_track_scoring_can_be_disabled():
    box = (400, 800, 900, 880)
    weak = regions.Region(box, (box,), 0.1)
    timeline = [[weak] for _ in range(20)]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2,
                                                                track_score=0.0))
    assert all(out)


def test_two_tracks_are_judged_separately():
    """One strong caption must not carry an unrelated weak detection with it."""
    good = (100, 100, 400, 160)
    bad = (1400, 600, 1700, 660)
    timeline = [[regions.Region(good, (good,), 0.92), regions.Region(bad, (bad,), 0.33)]
                for _ in range(12)]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2,
                                                                track_score=0.6))
    kept = {r.box for items in out for r in items}
    assert kept == {good}


def _moving_timeline(step, n=20, box=(400, 300, 800, 380), score=0.95):
    """A track whose box shifts by `step` pixels each frame."""
    out = []
    for i in range(n):
        b = (box[0] + i * step, box[1], box[2] + i * step, box[3])
        out.append([regions.Region(b, (b,), score)])
    return out


def test_a_stationary_caption_has_no_motion():
    tracks = regions.build_tracks(_moving_timeline(0))
    assert tracks[0].motion == pytest.approx(0.0)


def test_motion_is_measured_against_the_text_height():
    """Same pixel speed, taller text, lower relative motion."""
    short = regions.build_tracks(_moving_timeline(8, box=(400, 300, 800, 340)))[0]
    tall = regions.build_tracks(_moving_timeline(8, box=(400, 300, 800, 420)))[0]
    assert short.motion > tall.motion


def test_a_box_that_only_changes_size_is_not_moving():
    """A still caption whose box flickers one grid step taller.

    Its centre moves half a step, which at caption sizes reads the same as
    real scene text; its edges do not both move, so translation reads zero.
    """
    a, b = (400, 896, 1152, 992), (400, 896, 1152, 1008)
    timeline = [[regions.Region(a, (a,), 0.95)], [regions.Region(b, (b,), 0.95)]] * 6
    track = max(regions.build_tracks(timeline), key=lambda t: t.span)
    assert track.motion == pytest.approx(0.0)


def test_a_scroll_slower_than_the_grid_reads_as_still():
    """Why a credit roll survives: boxes are grid-snapped before this runs.

    A couple of pixels a frame lands in the same grid cell most frames, so the
    median step is zero even though the text is drifting.
    """
    grid, speed = 16, 2
    timeline = []
    for i in range(24):
        y = 300 + (i * speed // grid) * grid
        box = (400, y, 800, y + 80)
        timeline.append([regions.Region(box, (box,), 0.95)])
    track = max(regions.build_tracks(timeline), key=lambda t: t.span)
    assert track.motion == pytest.approx(0.0)
    assert any(regions.smooth_timeline(
        timeline, regions.RegionConfig(min_track=2, max_motion=0.02)))


def test_motion_ignores_an_isolated_jump():
    """The reason it is a median: a stray merge moves the box for one frame.

    Total travel would call this caption mobile; it is not.
    """
    box = (400, 300, 800, 380)
    far = (400, 700, 800, 780)
    timeline = [[regions.Region(box, (box,), 0.95)] for _ in range(12)]
    timeline[6] = [regions.Region(far, (far,), 0.95)]
    track = max(regions.build_tracks(timeline), key=lambda t: t.span)
    assert track.motion == pytest.approx(0.0)


def test_moving_text_is_dropped_and_stationary_text_is_kept():
    cfg = regions.RegionConfig(min_track=2, max_motion=0.02)
    assert not any(regions.smooth_timeline(_moving_timeline(12), cfg))
    assert all(regions.smooth_timeline(_moving_timeline(0), cfg))


def test_motion_filter_can_be_disabled():
    cfg = regions.RegionConfig(min_track=2, max_motion=None)
    assert any(regions.smooth_timeline(_moving_timeline(12), cfg))


def test_motion_is_read_before_boxes_are_canonicalised():
    """smooth_timeline replaces a run's boxes with their union.

    If motion were measured after that, every track would look stationary and
    the filter would never fire.
    """
    cfg = regions.RegionConfig(min_track=2, max_motion=0.02, sticky_iou=0.0)
    # sticky_iou 0 forces every frame into one run, i.e. maximum flattening.
    assert not any(regions.smooth_timeline(_moving_timeline(12), cfg))


# ------------------------------------------------------ mask gating


def _mask_with(shape, blobs):
    """A probability map with a filled rectangle per blob, in region coords."""
    m = np.zeros(shape, np.float32)
    for x0, y0, x1, y1 in blobs:
        m[y0:y1, x0:x1] = 1.0
    return m


def test_gate_drops_a_blob_that_sits_outside_every_detection():
    # Region spans (0,0)-(200,100). Text was found on the right; something
    # text-like also sits in the empty corner on the left.
    region = regions.Region((0, 0, 200, 100), ((100, 10, 180, 40),))
    prob = _mask_with((100, 200), [(110, 15, 170, 35), (10, 15, 60, 35)])
    out = regions.gate_by_detections(prob, region, dilate=4, min_inside=0.5)
    assert out[15:35, 110:170].min() == 1.0, "the detected text must survive intact"
    assert out[15:35, 10:60].max() == 0.0, "the intruder must be gone"


def test_gate_removes_a_blob_only_clipping_a_detection():
    """The case a pixelwise clip gets wrong, leaving a fragment behind."""
    region = regions.Region((0, 0, 200, 100), ((0, 50, 200, 90),))
    # A blob mostly above the detection, dipping a few rows into it.
    prob = _mask_with((100, 200), [(20, 30, 60, 56)])
    out = regions.gate_by_detections(prob, region, dilate=0, min_inside=0.5)
    assert out.max() == 0.0, "a mostly-outside blob should go entirely"


def test_gate_keeps_glyphs_that_overhang_their_detection():
    """Detector boxes are tight; a descender poking out must not be cut."""
    region = regions.Region((0, 0, 200, 100), ((40, 40, 160, 60),))
    prob = _mask_with((100, 200), [(50, 44, 150, 66)])  # overhangs below
    out = regions.gate_by_detections(prob, region, dilate=8, min_inside=0.5)
    assert out[44:66, 50:150].min() == 1.0


def test_gate_uses_frame_coordinates_for_detections():
    """Detections are absolute; the mask is relative to the region's origin."""
    region = regions.Region((500, 300, 700, 400), ((600, 310, 680, 340),))
    prob = _mask_with((100, 200), [(110, 15, 170, 35)])  # region-local, inside the det
    out = regions.gate_by_detections(prob, region, dilate=4, min_inside=0.5)
    assert out.max() == 1.0, "the detection was not translated into region space"


def test_gate_is_a_no_op_without_detections():
    region = regions.Region((0, 0, 200, 100), ())
    prob = _mask_with((100, 200), [(10, 10, 60, 40)])
    assert np.array_equal(regions.gate_by_detections(prob, region), prob)


def test_gate_is_a_no_op_at_zero_min_inside():
    region = regions.Region((0, 0, 200, 100), ((100, 10, 180, 40),))
    prob = _mask_with((100, 200), [(10, 10, 60, 40)])
    out = regions.gate_by_detections(prob, region, min_inside=0.0)
    assert np.array_equal(out, prob)


def test_legacy_timeline_entry_gates_to_the_whole_region():
    """An old cache has no detections; it must not filter everything away."""
    region = regions.region_from_json([0, 0, 200, 100])
    assert region.dets == ((0, 0, 200, 100),)
    prob = _mask_with((100, 200), [(10, 10, 60, 40)])
    assert np.array_equal(regions.gate_by_detections(prob, region), prob)


def test_region_json_round_trip():
    region = regions.Region((0, 0, 200, 100), ((10, 10, 40, 40), (60, 10, 90, 40)))
    assert regions.region_from_json(regions.region_to_json(region)) == region


def test_timeline_detection_check_spots_an_old_cache():
    assert not regions.timeline_has_detections([[[0, 0, 10, 10]]])
    assert regions.timeline_has_detections(
        [[{"box": [0, 0, 10, 10], "dets": [[1, 1, 5, 5]]}]]
    )


def _worker_result(frames):
    """A detection result shaped exactly as detect_worker writes it."""
    return {
        "n_frames": len(frames),
        "raw_text_frames": sum(1 for f in frames if f),
        "text_frames": sum(1 for f in frames if f),
        "min_track": 12,
        "regions": [[regions.region_to_json(r) for r in f] for f in frames],
    }


def test_worker_result_parses_into_regions():
    """The GUI once unpacked this by hand and got the dict's keys instead."""
    region = regions.Region((0, 0, 200, 100), ((10, 10, 40, 40),))
    parsed = [
        [regions.region_from_json(e) for e in frame]
        for frame in _worker_result([[], [region]])["regions"]
    ]
    assert parsed[0] == []
    assert parsed[1] == [region]
    assert parsed[1][0].box == (0, 0, 200, 100)
    assert parsed[1][0].dets == ((10, 10, 40, 40),)


def test_region_survives_a_json_text_round_trip():
    """Through real serialisation, since lists come back where tuples went in."""
    import json as _json

    region = regions.Region((704, 288, 1216, 464), ((730, 362, 1188, 444),))
    data = _json.loads(_json.dumps(_worker_result([[region]])))
    back = regions.region_from_json(data["regions"][0][0])
    assert back == region
    assert isinstance(back.box, tuple) and isinstance(back.dets[0], tuple)


def test_mask_store_round_trips_a_probability_map(tmp_path):
    store = session.MaskStore(str(tmp_path))
    prob = np.linspace(0, 1, 64 * 32, dtype=np.float32).reshape(32, 64)
    box = (16, 32, 80, 64)
    assert store.get(7, box) is None
    store.put(7, box, prob)
    back = store.get(7, box)
    assert back is not None and back.shape == prob.shape
    # 8-bit quantisation, far finer than the alpha window that consumes it.
    assert np.abs(back - prob).max() < 1.0 / 255 + 1e-6


def test_mask_store_keys_on_frame_and_box(tmp_path):
    store = session.MaskStore(str(tmp_path))
    prob = np.ones((8, 8), np.float32)
    store.put(1, (0, 0, 8, 8), prob)
    assert store.get(2, (0, 0, 8, 8)) is None, "a different frame must not collide"
    assert store.get(1, (0, 0, 8, 9)) is None, "a different box must not collide"


def test_cache_dir_is_stable_and_pair_specific():
    a = session.default_cache_dir("clip.mp4", "clip_depth.mp4")
    assert a == session.default_cache_dir("clip.mp4", "clip_depth.mp4")
    assert a != session.default_cache_dir("other.mp4", "clip_depth.mp4")


def test_display_normalisation_uses_the_shared_window():
    plane = np.array([[100, 200, 300]], np.uint16)
    out = session.to_display(plane, (100.0, 300.0))
    assert list(out[0]) == [0, 127, 255]


def test_display_normalisation_survives_a_flat_frame():
    plane = np.full((4, 4), 500, np.uint16)
    out = session.to_display(plane, (500.0, 500.0))
    assert out.dtype == np.uint8 and out.max() == 0


def test_each_region_resolves_its_own_depth():
    """A title and a subtitle at different depths must not average together."""
    depth = np.full((200, 200), 60 * 257, np.uint16)
    prob = np.zeros((200, 200), np.float32)
    depth[20:24, 20:80] = 120 * 257
    prob[20:24, 20:80] = 1.0
    depth[160:164, 20:80] = 240 * 257
    prob[160:164, 20:80] = 1.0

    boxes = [(0, 0, 200, 100), (0, 100, 200, 200)]
    out = comp.composite_frame(depth, prob, comp.CompositeConfig(), boxes)
    assert abs(int(out[21, 40]) - 120 * 257) <= 8 * 257
    assert abs(int(out[161, 40]) - 240 * 257) <= 8 * 257
