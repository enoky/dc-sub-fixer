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
    produced = {tuple(b) for boxes in out for b in boxes}
    assert len(produced) == 1, f"expected one stable box, got {produced}"
    # The union, so no frame's own detection gets cropped.
    assert produced == {(400, 896, 1152, 1008)}


def test_slowly_scrolling_text_is_tracked_but_not_frozen():
    # 40px a frame on a 300px-wide box: overlapping enough to stay one track,
    # too much movement to be treated as the same still box.
    timeline = [[(40 * i, 100, 300 + 40 * i, 140)] for i in range(5)]
    out = regions.smooth_timeline(timeline, regions.RegionConfig(min_track=2))
    assert all(boxes for boxes in out), "scrolling text should not be dropped"
    produced = {tuple(b) for boxes in out for b in boxes}
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
    x0, y0, x1, y1 = out[0]
    assert x0 <= 100 and x1 >= 320


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
