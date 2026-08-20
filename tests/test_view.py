"""The comparison view's fitting, driven offscreen.

Kept apart from test_core so the model-free suite stays free of Qt as well;
these skip cleanly wherever PySide6 is not installed.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dcsubfixer.gui import ComparisonView, fit_transform, recentred_offset  # noqa: E402

FW, FH = 960, 384


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pane_fit(width, height, solo=False):
    """What the fit should be for a view of this size."""
    cell = (width, height) if solo else (width / 2.0, height / 2.0)
    return min(cell[0] / FW, cell[1] / FH) * 0.98


# -- the arithmetic, without a widget --------------------------------------
def test_fit_transform_centres_and_leaves_a_margin():
    scale, ox, oy = fit_transform((100, 50), (200, 200))
    assert scale == pytest.approx(2.0 * 0.98)
    assert ox == pytest.approx((200 - 100 * scale) / 2.0)
    assert oy == pytest.approx((200 - 50 * scale) / 2.0)


def test_fit_transform_is_limited_by_the_tighter_axis():
    """A wide frame in a tall pane is held by width, and the reverse."""
    wide, _, _ = fit_transform((100, 10), (200, 200), margin=1.0)
    tall, _, _ = fit_transform((10, 100), (200, 200), margin=1.0)
    assert wide == pytest.approx(2.0)
    assert tall == pytest.approx(2.0)


def test_recentred_offset_holds_the_pane_centre():
    """Whatever sat at the middle of the pane is still at the middle of it."""
    off, before, after = (30.0, -12.0), (500.0, 300.0), (700.0, 450.0)
    ox, oy = recentred_offset(off, before, after)
    scale = 3.7  # any zoom: the point kept is the same one
    assert (before[0] / 2 - off[0]) / scale == pytest.approx((after[0] / 2 - ox) / scale)
    assert (before[1] / 2 - off[1]) / scale == pytest.approx((after[1] / 2 - oy) / scale)


def test_recentred_offset_does_nothing_when_the_pane_does_not_move():
    assert recentred_offset((30.0, -12.0), (500.0, 300.0), (500.0, 300.0)) == (30.0, -12.0)


# -- the widget ------------------------------------------------------------
def fresh_view(width=1000, height=600):
    view = ComparisonView()
    view.resize(width, height)
    view.set_images([np.zeros((FH, FW), np.uint8)] * 4, (FW, FH))
    view.grab()  # the paint that consumes the pending fit
    return view


def test_the_first_paint_fits_the_frame_into_a_pane(app):
    view = fresh_view()
    assert view.scale == pytest.approx(pane_fit(1000, 600))
    assert view.offset.x() == pytest.approx((500 - FW * view.scale) / 2.0)


@pytest.mark.parametrize("size", [(1400, 900), (700, 420), (400, 260), (1920, 1080)])
def test_resizing_the_view_refits_the_previews(app, size):
    """The regression this was written for: panes used to keep the old scale."""
    view = fresh_view()
    view.resize(*size)
    view.grab()
    assert view.scale == pytest.approx(pane_fit(*size))
    assert view.offset.x() == pytest.approx((size[0] / 2 - FW * view.scale) / 2.0)
    assert view.offset.y() == pytest.approx((size[1] / 2 - FH * view.scale) / 2.0)


def test_the_view_can_shrink_below_the_old_minimum(app):
    view = fresh_view()
    assert view.minimumWidth() <= 320 and view.minimumHeight() <= 200


def test_soloing_a_pane_refits_it_to_the_whole_view(app):
    view = fresh_view()
    view.set_solo(1)
    view.grab()
    assert view.scale == pytest.approx(pane_fit(1000, 600, solo=True))
    view.set_solo(None)
    view.grab()
    assert view.scale == pytest.approx(pane_fit(1000, 600))


def spin(view, at, up=True):
    view.wheelEvent(QWheelEvent(
        QPointF(*at), view.mapToGlobal(QPoint(*at)), QPoint(0, 0),
        QPoint(0, 120 if up else -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))


def test_a_zoom_survives_a_resize_and_keeps_its_subject_centred(app):
    """Enlarging the window while zoomed in should reveal more, not jump."""
    view = fresh_view()
    spin(view, (250, 150))
    spin(view, (250, 150))
    zoomed = view.scale
    assert zoomed > pane_fit(1000, 600)
    before = (QPointF(250, 150) - view.offset) / view.scale

    view.resize(1400, 900)
    view.grab()
    assert view.scale == pytest.approx(zoomed), "the user's zoom is not ours to reset"
    after = (QPointF(350, 225) - view.offset) / view.scale
    assert after.x() == pytest.approx(before.x())
    assert after.y() == pytest.approx(before.y())


def test_fitting_again_hands_the_framing_back(app):
    """What the F key does: resizes follow the window once more."""
    view = fresh_view()
    spin(view, (250, 150))
    view.fit()
    assert view.scale == pytest.approx(pane_fit(1000, 600))
    view.resize(900, 560)
    view.grab()
    assert view.scale == pytest.approx(pane_fit(900, 560))


def test_a_pan_also_makes_the_framing_the_users(app):
    view = fresh_view()
    press = type("E", (), {"button": lambda s: Qt.LeftButton,
                           "position": lambda s: QPointF(100, 100)})()
    view.mousePressEvent(press)
    view.mouseMoveEvent(type("E", (), {"position": lambda s: QPointF(140, 130)})())
    view.mouseReleaseEvent(press)
    panned = view.offset.x()
    view.resize(1200, 700)
    view.grab()
    assert view.scale == pytest.approx(pane_fit(1000, 600)), "pan is framing too"
    assert view.offset.x() == pytest.approx(panned + (600 - 500) / 2.0)
