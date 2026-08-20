"""A tuning GUI for dc-sub-fixer.

The point of this window is judgement, not automation: the compositing
settings are the ones no default can get right for every clip, and the only
way to choose them is to look at a real frame. So everything is arranged
around one loop - jump to a frame with text, move a slider, see the result.

That loop is only pleasant because segmentation is cached to disk (see
session.py). Moving a slider re-composites from a stored mask, which is plain
array arithmetic and takes about twenty milliseconds; no GPU work happens
until you visit a frame nobody has segmented yet.

Two jobs stay in subprocesses. Detection must, because Paddle cannot share a
process with torch on Windows. The final render does by choice: it shells out
to the same CLI you could run yourself, so what you tune is exactly what you
get, and the command is shown so you can reproduce it without the GUI.

Run with:  python -m dcsubfixer.gui
"""

from __future__ import annotations

import faulthandler
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from . import composite as comp
from . import ocr, regions, session

PANE_TITLES = ("RGB source", "depth (before)", "glyph mask", "depth (fixed)")


# --------------------------------------------------------------------------
# comparison view
# --------------------------------------------------------------------------
class ComparisonView(QWidget):
    """Four panes over one shared pan/zoom.

    Painted directly rather than assembled from scroll areas: every pane shows
    the same region of the same coordinate space, so a single transform is
    both simpler and exactly synchronised, which matters when the whole task
    is comparing one pane against another.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self._images: List[Optional[QImage]] = [None] * 4
        self._frame_size = (1, 1)
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._drag: Optional[QPointF] = None
        self._fit_pending = True
        self.solo: Optional[int] = None
        self.setFocusPolicy(Qt.StrongFocus)

    def set_images(self, images: List[Optional[np.ndarray]], frame_size: Tuple[int, int]) -> None:
        self._images = [_to_qimage(im) for im in images]
        if frame_size != self._frame_size:
            self._frame_size = frame_size
            self._fit_pending = True
        self.update()

    def _cells(self) -> List[QRectF]:
        """A 2x2 grid, or one full-size pane when soloing.

        Stacking the panes in a column was tried for scope footage and is no
        better: with four panes in a roughly square viewport the limit is the
        area each one gets, not the arrangement, and both layouts land on the
        same image size. Detail comes from soloing a pane instead.
        """
        if self.solo is not None:
            return [QRectF(0, 0, self.width(), self.height())]
        w, h = self.width() / 2.0, self.height() / 2.0
        return [QRectF(c * w, r * h, w, h) for r in range(2) for c in range(2)]

    def fit(self) -> None:
        fw, fh = self._frame_size
        cell = self._cells()[0]
        if fw <= 0 or fh <= 0:
            return
        self.scale = min(cell.width() / fw, cell.height() / fh) * 0.98
        self.offset = QPointF(
            (cell.width() - fw * self.scale) / 2.0, (cell.height() - fh * self.scale) / 2.0
        )
        self._fit_pending = False
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._fit_pending:
            self.fit()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 26))
        cells = self._cells()
        shown = [self.solo] if self.solo is not None else [0, 1, 2, 3]

        for cell, idx in zip(cells, shown):
            img = self._images[idx] if idx is not None else None
            p.save()
            p.setClipRect(cell)
            p.translate(cell.topLeft())
            if img is not None:
                p.translate(self.offset)
                p.scale(self.scale, self.scale)
                p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 3.0)
                p.drawImage(0, 0, img)
            p.restore()

            p.save()
            p.setPen(QPen(QColor(70, 70, 78)))
            p.drawRect(cell.adjusted(0, 0, -1, -1))
            p.setPen(QPen(QColor(180, 220, 180)))
            p.drawText(cell.adjusted(8, 6, -8, -8), Qt.AlignTop | Qt.AlignLeft, PANE_TITLES[idx])
            p.restore()
        p.end()

    # -- interaction ----------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        # Zoom about the cursor, in the coordinates of whichever pane it is
        # over, so the point under the pointer stays put in every pane.
        cells = self._cells()
        pos = event.position()
        cell = next((c for c in cells if c.contains(pos)), cells[0])
        local = pos - cell.topLeft()
        before = (local - self.offset) / self.scale
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale = max(0.05, min(64.0, self.scale * factor))
        self.offset = local - before * self.scale
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            self.offset += event.position() - self._drag
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Double click zooms one pane to fill the view, and back again."""
        if self.solo is not None:
            self.set_solo(None)
        else:
            for idx, cell in enumerate(self._cells()):
                if cell.contains(event.position()):
                    self.set_solo(idx)
                    break

    def set_solo(self, index: Optional[int]) -> None:
        """Show one pane full size, keeping the current zoom and pan."""
        if index == self.solo:
            return
        self.solo = index
        # Only refit when leaving a zoomed-in inspection, so that flipping
        # between before and after does not disturb the framing.
        if index is None:
            self._fit_pending = True
        self.update()

    def flip(self) -> None:
        """Swap between the before and after panes, the A/B comparison.

        Held at the same zoom and position, so the only thing that changes
        between the two views is the pixels being judged.
        """
        if self.solo == 3:
            self.set_solo(1)
        else:
            self.set_solo(3)


def _to_qimage(arr: Optional[np.ndarray]) -> Optional[QImage]:
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return img.copy()  # detach from the numpy buffer


# --------------------------------------------------------------------------
# timeline strip
# --------------------------------------------------------------------------
class TimelineStrip(QWidget):
    """Frame ruler showing where text lives and what has been segmented."""

    seek = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(34)
        self.n_frames = 0
        self.runs: List[Tuple[int, int]] = []
        self.excluded_runs: List[Tuple[int, int]] = []
        self.cached: set = set()
        self.current = 0

    def configure(self, n_frames: int, runs: List[Tuple[int, int]]) -> None:
        self.n_frames = max(1, n_frames)
        self.runs = runs
        self.update()

    def set_current(self, frame: int) -> None:
        self.current = frame
        self.update()

    def _x(self, frame: int) -> float:
        return frame / max(self.n_frames - 1, 1) * (self.width() - 1)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 36))
        h = self.height()
        for a, b in self.runs:
            x0, x1 = self._x(a), self._x(b)
            p.fillRect(QRectF(x0, 6, max(x1 - x0, 1.5), h - 18), QColor(72, 132, 200))
        # Rejected runs stay visible, so it is obvious what was removed and
        # where, rather than the bar simply disappearing.
        for a, b in self.excluded_runs:
            x0, x1 = self._x(a), self._x(b)
            p.fillRect(QRectF(x0, 6, max(x1 - x0, 1.5), h - 18), QColor(120, 60, 60))
        for f in self.cached:
            p.fillRect(QRectF(self._x(f), h - 10, 1.5, 5), QColor(120, 200, 120))
        p.setPen(QPen(QColor(240, 220, 120), 2))
        x = self._x(self.current)
        p.drawLine(int(x), 0, int(x), h)
        p.end()

    def _emit_from(self, pos) -> None:
        frac = min(max(pos.x() / max(self.width() - 1, 1), 0.0), 1.0)
        self.seek.emit(int(round(frac * (self.n_frames - 1))))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._emit_from(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton:
            self._emit_from(event.position())


# --------------------------------------------------------------------------
# background workers
# --------------------------------------------------------------------------
class RenderWorker(QObject):
    """Renders one frame off the UI thread; segmentation can take ~0.2s.

    The kick is a signal rather than a direct call or a QTimer, because both of
    those run in whichever thread asked. A signal emitted from the UI thread to
    an object living in the render thread is delivered as a queued call, which
    is what actually gets the work off the UI thread.

    Only the most recent request is kept. Dragging a slider produces a stream of
    them, and rendering every intermediate value would put the view further and
    further behind the controls.
    """

    done = Signal(object, float)
    failed = Signal(str)
    _kick = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.session: Optional[session.TuningSession] = None
        self._pending: Optional[Tuple[int, comp.CompositeConfig, bool]] = None
        self._lock = threading.Lock()
        self._kick.connect(self._run)

    def request(self, frame: int, cfg: comp.CompositeConfig, segment: bool) -> None:
        with self._lock:
            self._pending = (frame, cfg, segment)
        self._kick.emit()

    def _run(self) -> None:
        with self._lock:
            job, self._pending = self._pending, None
        if job is None or self.session is None:
            return
        frame, cfg, segment = job
        try:
            t = time.time()
            panels = self.session.render(frame, cfg, segment=segment)
            self.done.emit(panels, (time.time() - t) * 1000.0)
        except Exception as exc:  # surfaced in the status bar, never silent
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


class ProcessWorker(QObject):
    """Runs a subprocess, forwarding PROGRESS lines and the exit status."""

    progress = Signal(int, int)
    line = Signal(str)
    finished = Signal(int)

    def __init__(self, cmd: List[str], cwd: Optional[str] = None) -> None:
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    def run(self) -> None:
        pattern = re.compile(r"PROGRESS (\d+) (\d+)")
        try:
            self._proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for raw in self._proc.stdout:
                text = raw.rstrip()
                m = pattern.search(text)
                if m:
                    self.progress.emit(int(m.group(1)), int(m.group(2)))
                elif text:
                    self.line.emit(text)
            self._proc.wait()
            self.finished.emit(-1 if self._cancelled else self._proc.returncode)
        except Exception as exc:
            self.line.emit(f"{type(exc).__name__}: {exc}")
            self.finished.emit(1)

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# --------------------------------------------------------------------------
# main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("dc-sub-fixer")
        self.resize(1500, 950)

        self.session: Optional[session.TuningSession] = None
        self.rgb_path = ""
        self.depth_path = ""
        self.frame = 0
        self._proc_thread: Optional[QThread] = None
        self._proc_worker: Optional[ProcessWorker] = None
        self._proc_done_cb = None

        self._build_ui()
        self._build_worker()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(lambda: self._request_render(segment=False))

    # -- construction ---------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)

        # file row
        files = QHBoxLayout()
        self.btn_rgb = QPushButton("Open RGB…")
        self.btn_depth = QPushButton("Open depth…")
        self.btn_rgb.clicked.connect(lambda: self._pick("rgb"))
        self.btn_depth.clicked.connect(lambda: self._pick("depth"))
        self.lbl_files = QLabel("no clip loaded")
        self.lbl_files.setStyleSheet("color: #9ab;")
        files.addWidget(self.btn_rgb)
        files.addWidget(self.btn_depth)
        files.addWidget(self.lbl_files, 1)
        outer.addLayout(files)

        split = QSplitter(Qt.Horizontal)
        self.view = ComparisonView()
        split.addWidget(self.view)
        split.addWidget(self._build_side_panel())
        split.setStretchFactor(0, 1)
        split.setSizes([1050, 430])
        outer.addWidget(split, 1)

        # transport
        self.strip = TimelineStrip()
        self.strip.seek.connect(self.goto_frame)
        outer.addWidget(self.strip)

        transport = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.goto_frame)
        self.lbl_frame = QLabel("- / -")
        self.lbl_frame.setMinimumWidth(120)
        for text, delta in (("|◀", -10 ** 9), ("◀", -1), ("▶", 1), ("▶|", 10 ** 9)):
            b = QPushButton(text)
            b.setFixedWidth(38)
            b.clicked.connect(lambda _=False, d=delta: self.goto_frame(self.frame + d))
            transport.addWidget(b)
        self.btn_next_text = QPushButton("next text (N)")
        self.btn_next_text.clicked.connect(self._next_text_run)
        transport.addWidget(self.btn_next_text)
        self.cmb_run = QComboBox()
        self.cmb_run.setMinimumWidth(150)
        self.cmb_run.setToolTip("Runs of text on this frame; ids match the labels "
                                "drawn on the boxes in the RGB pane")
        self.cmb_run.currentIndexChanged.connect(lambda _: self._sync_exclude_button())
        transport.addWidget(self.cmb_run)
        self.btn_exclude = QPushButton("exclude (X)")
        self.btn_exclude.setToolTip(
            "Reject the selected run. Some scene text is real, level and still, "
            "and no measurement can tell it from a caption.")
        self.btn_exclude.clicked.connect(self._toggle_run)
        transport.addWidget(self.btn_exclude)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.lbl_frame)
        outer.addLayout(transport)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        self.setCentralWidget(root)
        self.status = self.statusBar()
        self.status.showMessage("Open an RGB clip and its depth map to begin.")

        shortcuts = [
            (Qt.Key_Left, lambda: self.goto_frame(self.frame - 1)),
            (Qt.Key_Right, lambda: self.goto_frame(self.frame + 1)),
            (Qt.Key_F, self.view.fit),
            (Qt.Key_Space, self.view.flip),
            (Qt.Key_0, lambda: self.view.set_solo(None)),
            # Not Tab: Qt spends that on focus navigation before a shortcut sees it.
            (Qt.Key_N, self._next_text_run),
            (Qt.Key_X, self._toggle_run),
        ]
        for i in range(4):
            shortcuts.append((getattr(Qt, f"Key_{i + 1}"), lambda n=i: self.view.set_solo(n)))
        for key, slot in shortcuts:
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(slot)
            self.addAction(act)

        hint = QLabel(
            "  ←/→ frame · N next text · X exclude a run · 1-4 solo a pane · Space flips before/after"
            " · 0 all · F fit · double-click a pane"
        )
        hint.setStyleSheet("color: #778;")
        outer.addWidget(hint)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)

        # detection
        det = QGroupBox("1 · Detect text")
        dl = QFormLayout(det)
        self.spin_box_thresh = QDoubleSpinBox()
        self.spin_box_thresh.setRange(0.05, 0.95)
        self.spin_box_thresh.setSingleStep(0.05)
        self.spin_box_thresh.setValue(0.6)
        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(0, 300)
        self.spin_min_track.setSpecialValueText("auto (0.5s)")
        self.spin_min_track.setValue(0)
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["auto", "stretch", "fit", "fill"])
        self.cmb_align.currentTextChanged.connect(self._align_changed)
        dl.addRow("track conf.", self.spin_box_thresh)
        dl.addRow("min frames", self.spin_min_track)
        dl.addRow("alignment", self.cmb_align)
        self.btn_detect = QPushButton("Run detection")
        self.btn_detect.clicked.connect(self._run_detection)
        self.btn_detect.setEnabled(False)
        dl.addRow(self.btn_detect)
        self.lbl_detect = QLabel("not run")
        self.lbl_detect.setWordWrap(True)
        self.lbl_detect.setStyleSheet("color: #9ab;")
        dl.addRow(self.lbl_detect)
        box.addWidget(det)

        # compositing
        cgb = QGroupBox("2 · Compositing (live)")
        cl = QFormLayout(cgb)
        self.chk_auto_value = QCheckBox("read depth from the map")
        self.chk_auto_value.setChecked(True)
        self.chk_auto_value.stateChanged.connect(self._settings_changed)
        self.sld_value = _slider(0, 255, 255, self._settings_changed)
        self.sld_low = _slider(0, 100, 45, self._settings_changed)
        self.sld_high = _slider(0, 100, 60, self._settings_changed)
        self.sld_dilate = _slider(-150, 150, 70, self._settings_changed)
        self.sld_feather = _slider(0, 40, 14, self._settings_changed)
        self.sld_opacity = _slider(0, 100, 100, self._settings_changed)
        self.chk_binary = QCheckBox("hard edges")
        self.chk_binary.stateChanged.connect(self._settings_changed)
        self.chk_heal = QCheckBox("inpaint the halo")
        self.chk_heal.setChecked(True)
        self.chk_heal.stateChanged.connect(self._settings_changed)
        self.sld_heal_r = _slider(1, 24, 6, self._settings_changed)

        cl.addRow("text value", self.chk_auto_value)
        cl.addRow("  fixed level", self.sld_value)
        cl.addRow("mask low", self.sld_low)
        cl.addRow("mask high", self.sld_high)
        cl.addRow("dilate", self.sld_dilate)
        cl.addRow("feather", self.sld_feather)
        cl.addRow("opacity", self.sld_opacity)
        cl.addRow("", self.chk_binary)
        self.chk_run_mask = QCheckBox("one mask per run of text")
        self.chk_run_mask.setChecked(True)
        self.chk_run_mask.setToolTip(
            "Take each run's strongest mask and use it throughout, rather than "
            "segmenting every frame. Stops logos flickering.")
        self.chk_run_mask.stateChanged.connect(self._run_mask_changed)
        cl.addRow("", self.chk_run_mask)
        cl.addRow("heal", self.chk_heal)
        cl.addRow("  radius", self.sld_heal_r)
        self.lbl_settings = QLabel()
        self.lbl_settings.setStyleSheet("color: #9ab;")
        cl.addRow(self.lbl_settings)
        box.addWidget(cgb)

        # render
        rgb_ = QGroupBox("3 · Render")
        rl = QVBoxLayout(rgb_)
        q = QHBoxLayout()
        q.addWidget(QLabel("quality"))
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["lossless", "12", "18", "23"])
        q.addWidget(self.cmb_quality, 1)
        rl.addLayout(q)
        self.btn_render = QPushButton("Render full clip…")
        self.btn_render.clicked.connect(self._run_render)
        self.btn_render.setEnabled(False)
        rl.addWidget(self.btn_render)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_process)
        self.btn_cancel.setEnabled(False)
        rl.addWidget(self.btn_cancel)
        self.chk_clear_masks = QCheckBox("clear cached masks on close")
        self.chk_clear_masks.setChecked(True)
        self.chk_clear_masks.setToolTip(
            "Masks are pure GPU output and rebuild identically, about 0.2s a "
            "frame. Detections and excluded runs are kept either way.")
        rl.addWidget(self.chk_clear_masks)
        self.chk_clear_timeline = QCheckBox("…and the detections too")
        self.chk_clear_timeline.setToolTip(
            "Also discards the detection pass and any runs you excluded, so the "
            "next open starts from nothing.")
        rl.addWidget(self.chk_clear_timeline)
        rl.addWidget(QLabel("equivalent command:"))
        self.txt_cmd = QPlainTextEdit()
        self.txt_cmd.setReadOnly(True)
        self.txt_cmd.setMaximumHeight(110)
        self.txt_cmd.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        rl.addWidget(self.txt_cmd)
        box.addWidget(rgb_)

        box.addStretch(1)
        return panel

    def _build_worker(self) -> None:
        self._render_thread = QThread(self)
        self._render_worker = RenderWorker()
        self._render_worker.moveToThread(self._render_thread)
        self._render_worker.done.connect(self._on_rendered)
        # A bound method, for the same reason as in _start_process: a lambda
        # here would run in the render thread and touch the status bar there.
        self._render_worker.failed.connect(self._on_render_failed)
        self._render_thread.start()

    def _on_render_failed(self, message: str) -> None:
        self.status.showMessage(f"render failed — {message.splitlines()[0]}")
        print(message, file=sys.stderr)

    # -- settings -------------------------------------------------------
    def composite_config(self) -> comp.CompositeConfig:
        low = self.sld_low.value() / 100.0
        high = max(self.sld_high.value() / 100.0, low + 0.01)
        return comp.CompositeConfig(
            text_value="auto" if self.chk_auto_value.isChecked() else str(self.sld_value.value()),
            mask_low=low,
            mask_high=high,
            binary=self.chk_binary.isChecked(),
            dilate=self.sld_dilate.value() / 100.0,
            feather=self.sld_feather.value() / 20.0,
            heal=self.chk_heal.isChecked(),
            heal_radius=self.sld_heal_r.value(),
            opacity=self.sld_opacity.value() / 100.0,
        )

    def _settings_changed(self, *_) -> None:
        self.sld_value.setEnabled(not self.chk_auto_value.isChecked())
        self.sld_heal_r.setEnabled(self.chk_heal.isChecked())
        cfg = self.composite_config()
        self.lbl_settings.setText(
            f"window {cfg.mask_low:.2f}–{cfg.mask_high:.2f}"
            + (f", dilate {cfg.dilate:+.2f}px" if cfg.dilate else "")
            + (f", feather {cfg.feather:.2f}" if cfg.feather else "")
            + (f", opacity {cfg.opacity:.0%}" if cfg.opacity != 1 else "")
        )
        self._update_command()
        if self.session is not None:
            self._debounce.start()

    def _run_mask_changed(self, *_) -> None:
        if self.session is None:
            return
        self.session.run_mask = self.chk_run_mask.isChecked()
        self._update_command()
        self._request_render(segment=True)

    def _align_changed(self, text: str) -> None:
        if self.session is None:
            return
        self.session.set_alignment(text)
        self.status.showMessage(f"alignment — {self.session.align_note}")
        self._update_command()
        self._request_render(segment=False)

    # -- files ----------------------------------------------------------
    def _pick(self, which: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choose the {which} video", "", "Video (*.mp4 *.mkv *.mov *.avi);;All files (*)"
        )
        if not path:
            return
        if which == "rgb":
            self.rgb_path = path
            if not self.depth_path:
                guess = _guess_depth(path)
                if guess:
                    self.depth_path = guess
        else:
            self.depth_path = path
        if self.rgb_path and self.depth_path:
            self._open_session()
        else:
            self.lbl_files.setText(f"RGB: {self.rgb_path or '—'}   depth: {self.depth_path or '—'}")

    def _open_session(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
            self.session = session.TuningSession(
                self.rgb_path,
                self.depth_path,
                models_dir=_models_dir(),
                align=self.cmb_align.currentText(),
                region=self._region_config(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Could not open", f"{type(exc).__name__}: {exc}")
            self.session = None
            return

        s = self.session
        self._render_worker.session = s
        self.lbl_files.setText(
            f"RGB {s.rgb_info.width}×{s.rgb_info.height} {s.rgb_info.bit_depth}-bit · "
            f"depth {s.depth_info.width}×{s.depth_info.height} {s.depth_info.bit_depth}-bit · "
            f"{s.n_frames} frames"
        )
        self.slider.setRange(0, max(s.n_frames - 1, 0))
        self.strip.configure(s.n_frames, [])
        self.btn_detect.setEnabled(True)
        self._sync_exclude_button()
        self.status.showMessage(f"alignment — {s.align_note}")

        if s.load_timeline():
            self.lbl_detect.setText(
                f"loaded cached timeline · {sum(1 for b in s.timeline if b)} frames with text"
            )
            self._after_detection()
        elif s.timeline_note:
            self.lbl_detect.setText(s.timeline_note)
        self.goto_frame(0)
        self._settings_changed()

    def _region_config(self) -> regions.RegionConfig:
        mt = self.spin_min_track.value()
        return regions.RegionConfig(min_track=mt if mt > 0 else None)

    # -- navigation -----------------------------------------------------
    def goto_frame(self, frame: int) -> None:
        if self.session is None:
            return
        frame = max(0, min(int(frame), max(self.session.n_frames - 1, 0)))
        self.frame = frame
        if self.slider.value() != frame:
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
        self.strip.set_current(frame)
        self._sync_run_list()
        has_text = bool(self.session.timeline[frame]) if frame < len(self.session.timeline) else False
        self.lbl_frame.setText(f"{frame} / {max(self.session.n_frames - 1, 0)}" + ("  ✎" if has_text else ""))
        self._request_render(segment=True)

    def _next_text_run(self) -> None:
        if self.session is None:
            return
        for a, _b in self.session.text_runs():
            if a > self.frame:
                self.goto_frame(a)
                return
        runs = self.session.text_runs()
        if runs:
            self.goto_frame(runs[0][0])

    def _runs_here(self) -> List[int]:
        s = self.session
        if s is None or self.frame >= len(s.timeline):
            return []
        return sorted({r.run for r in s.timeline[self.frame] if r.run >= 0})

    def _selected_run(self) -> Optional[int]:
        data = self.cmb_run.currentData()
        return int(data) if data is not None else None

    def _sync_run_list(self) -> None:
        """Offer the runs on this frame, keeping the selection where possible."""
        s = self.session
        want = self._runs_here()
        keep = self._selected_run()
        self.cmb_run.blockSignals(True)
        self.cmb_run.clear()
        for rid in want:
            span = sum(1 for f in s.timeline if any(r.run == rid for r in f))
            mark = " (excluded)" if rid in s.excluded else ""
            self.cmb_run.addItem(f"run {rid} · {span}f{mark}", rid)
        if keep in want:
            self.cmb_run.setCurrentIndex(want.index(keep))
        self.cmb_run.blockSignals(False)
        self._sync_exclude_button()

    def _sync_exclude_button(self) -> None:
        rid = self._selected_run()
        s = self.session
        on = rid is not None and s is not None
        self.btn_exclude.setEnabled(on)
        if on and rid in s.excluded:
            self.btn_exclude.setText("restore (X)")
        else:
            self.btn_exclude.setText("exclude (X)")

    def _toggle_run(self) -> None:
        """Reject or restore the run selected for this frame."""
        s = self.session
        if s is None:
            return
        run_id = self._selected_run()
        if run_id is None:
            self.status.showMessage("no run of text on this frame to exclude")
            return
        excluded = s.toggle_run(run_id)
        s.persist_exclusions()
        self._refresh_runs()
        self.goto_frame(self.frame)
        self.status.showMessage(
            f"run {run_id} {'excluded' if excluded else 'restored'} "
            f"({len(s.excluded)} excluded in total)"
        )

    def _request_render(self, segment: bool) -> None:
        if self.session is None:
            return
        self._render_worker.request(self.frame, self.composite_config(), segment)

    def _on_rendered(self, panels: session.FramePanels, ms: float) -> None:
        s = self.session
        if s is None:
            return
        dh, dw = s.depth_info.height, s.depth_info.width
        rgb = cv2.resize(panels.rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        for region in panels.regions:
            a = s.alignment.map_box(region.box)
            cv2.rectangle(rgb, (a[0], a[1]), (a[2], a[3]), (90, 230, 90), 1)
            if region.run >= 0:
                cv2.putText(rgb, str(region.run), (a[0] + 2, max(10, a[1] - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 230, 90), 1)

        images = [
            rgb,
            session.to_display(panels.depth_before, panels.value_range),
            (np.clip(panels.prob, 0, 1) * 255).astype(np.uint8),
            session.to_display(panels.depth_after, panels.value_range),
        ]
        self.view.set_images(images, (dw, dh))
        if panels.boxes:
            self.strip.cached.add(panels.index)
        self.strip.update()
        self.status.showMessage(
            f"frame {panels.index} · {len(panels.boxes)} region(s) · {ms:.0f} ms"
            + ("" if panels.boxes else " · no text here")
        )

    # -- detection ------------------------------------------------------
    def _run_detection(self) -> None:
        if self.session is None or self._proc_thread is not None:
            return
        s = self.session
        s.region = self._region_config()
        # The spin box is the *track* bar; the detector keeps its low floor so
        # weak frames of a real caption still reach the tracker.
        s.region.track_score = self.spin_box_thresh.value()
        det = ocr.DetectorConfig()
        request = {
            "rgb_path": os.path.abspath(self.rgb_path),
            "detector": asdict(det),
            "region": asdict(s.region),
            "ocr_stride": 1,
            "max_frames": None,
        }
        self._req_file = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(request, self._req_file)
        self._req_file.close()
        self._res_path = self._req_file.name.replace(".json", ".result.json")

        cmd = [sys.executable, "-m", "dcsubfixer.detect_worker",
               self._req_file.name, self._res_path, "--progress"]
        self._start_process(cmd, "detecting text", self._on_detection_done)

    def _on_detection_done(self, code: int) -> None:
        if code != 0:
            self.lbl_detect.setText(f"detection failed (exit {code})")
            return
        try:
            with open(self._res_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as exc:
            self.lbl_detect.setText(f"could not read result — {exc}")
            return
        s = self.session
        s.set_timeline(data)
        s.save_timeline(data)
        self.lbl_detect.setText(
            f"text in {data['text_frames']} of {data['n_frames']} frames · "
            f"tracks shorter than {data['min_track']} frames dropped"
        )
        for p in (self._req_file.name, self._res_path):
            try:
                os.remove(p)
            except OSError:
                pass
        self._after_detection()

    def _refresh_runs(self) -> None:
        s = self.session
        self.strip.excluded_runs = _spans(
            [i for i in range(len(s.timeline)) if s.timeline[i] and not s.visible(i)]
        )
        self.strip.configure(s.n_frames, s.text_runs())
        self._sync_run_list()

    def _after_detection(self) -> None:
        s = self.session
        runs = s.text_runs()
        self._refresh_runs()
        self.btn_render.setEnabled(bool(runs))
        self._update_command()
        if runs:
            self.goto_frame(runs[0][0])

    # -- render ---------------------------------------------------------
    def _run_render(self) -> None:
        if self.session is None or self._proc_thread is not None:
            return
        default = os.path.splitext(self.depth_path)[0] + "_fixed.mp4"
        out, _ = QFileDialog.getSaveFileName(self, "Save corrected depth video", default,
                                             "MP4 (*.mp4);;All files (*)")
        if not out:
            return
        self._start_process(self._render_command(out), "rendering", self._on_render_done)

    def _on_render_done(self, code: int) -> None:
        if code == 0:
            self.status.showMessage("render complete")
        elif code == -1:
            self.status.showMessage("render cancelled")
        else:
            self.status.showMessage(f"render failed (exit {code})")

    def _render_command(self, out_path: str = "OUTPUT.mp4") -> List[str]:
        cfg = self.composite_config()
        s = self.session
        cmd = [sys.executable, "-m", "dcsubfixer", self.rgb_path, self.depth_path, out_path,
               "--mask-low", f"{cfg.mask_low:.2f}", "--mask-high", f"{cfg.mask_high:.2f}",
               "--quality", self.cmb_quality.currentText(),
               "--track-score", f"{self.spin_box_thresh.value():.2f}",
               "--align", self.cmb_align.currentText()]
        if cfg.heal:
            cmd += ["--heal", "--heal-radius", str(cfg.heal_radius)]
        if cfg.binary:
            cmd += ["--binary"]
        if not self.chk_run_mask.isChecked():
            cmd += ["--no-run-mask"]
        if cfg.dilate:
            cmd += ["--dilate", f"{cfg.dilate:.2f}"]
        if cfg.feather:
            cmd += ["--feather", f"{cfg.feather:.2f}"]
        if cfg.opacity != 1.0:
            cmd += ["--opacity", f"{cfg.opacity:.2f}"]
        if cfg.text_value != "auto":
            cmd += ["--text-value", cfg.text_value]
        if self.spin_min_track.value() > 0:
            cmd += ["--min-track", str(self.spin_min_track.value())]
        if s is not None and os.path.isfile(s.paths.timeline):
            cmd += ["--timeline", s.paths.timeline]
        if s is not None and s.excluded:
            cmd += ["--exclude-runs", ",".join(str(i) for i in sorted(s.excluded))]
        return cmd

    def _update_command(self) -> None:
        if not (self.rgb_path and self.depth_path):
            return
        parts = self._render_command()
        shown = ["python -m dcsubfixer"] + [_q(p) for p in parts[3:]]
        self.txt_cmd.setPlainText(" ".join(shown))

    # -- subprocess plumbing --------------------------------------------
    def _start_process(self, cmd: List[str], label: str, on_done) -> None:
        """Run a subprocess, reporting into the window as it goes.

        Every signal below lands on a bound method of this window rather than
        on a lambda or a closure. That is not style: Qt picks a direct or a
        queued connection from the receiver's thread, and it can only work that
        out for a slot belonging to a QObject. A plain callable gets a direct
        connection and therefore runs in the worker thread, which means the
        widgets it touches are being mutated off the GUI thread - an access
        violation that takes the whole process down.
        """
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.btn_detect.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.showMessage(f"{label}…")

        thread = QThread(self)
        worker = ProcessWorker(cmd, cwd=_project_root())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.line.connect(self._on_process_line)
        worker.finished.connect(self._on_process_finished)

        self._proc_done_cb = on_done
        self._proc_thread = thread
        self._proc_worker = worker
        thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _on_process_line(self, text: str) -> None:
        self.status.showMessage(text[:160])

    def _on_process_finished(self, code: int) -> None:
        thread, self._proc_thread = self._proc_thread, None
        self._proc_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        self.progress.setVisible(False)
        self.btn_detect.setEnabled(self.session is not None)
        self.btn_render.setEnabled(bool(self.session and self.session.text_runs()))
        self.btn_cancel.setEnabled(False)
        cb, self._proc_done_cb = self._proc_done_cb, None
        if cb is not None:
            cb(code)

    def _cancel_process(self) -> None:
        if self._proc_worker is not None:
            self._proc_worker.cancel()

    # -- teardown -------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel_process()
        self._render_thread.quit()
        self._render_thread.wait(2000)
        if self.session is not None:
            self._clear_cache_on_exit()
            self.session.close()
        super().closeEvent(event)

    def _clear_cache_on_exit(self) -> None:
        """Reclaim the clip's cache, to whatever depth was asked for."""
        masks = self.chk_clear_masks.isChecked()
        timeline = self.chk_clear_timeline.isChecked()
        if not (masks or timeline):
            return
        try:
            freed = self.session.clear_cache(masks=masks, timeline=timeline)
        except OSError as exc:
            # Never let tidying up stop the window from closing.
            print(f"could not clear the cache: {exc}", file=sys.stderr)
            return
        what = "masks and detections" if timeline else "masks"
        self.status.showMessage(f"cleared cached {what} — {_bytes(freed)}")
        print(f"cleared cached {what} ({_bytes(freed)}) from "
              f"{self.session.paths.cache_dir}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bytes(n: int) -> str:
    """A size a person can read, rather than 0.0 MB for everything small."""
    for unit, size in (("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} bytes"


def _spans(frames: List[int]) -> List[Tuple[int, int]]:
    """Contiguous runs from a sorted list of frame indices."""
    out: List[Tuple[int, int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def _slider(lo: int, hi: int, value: int, on_change) -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(value)
    s.valueChanged.connect(on_change)
    return s


def _q(text: str) -> str:
    return f'"{text}"' if " " in text else text


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _models_dir() -> str:
    env = os.environ.get("DCSUBFIXER_MODELS")
    if env:
        return env
    return os.path.join(_project_root(), "models")


def _guess_depth(rgb_path: str) -> Optional[str]:
    """Depth maps sit beside the source with a predictable suffix."""
    stem, ext = os.path.splitext(rgb_path)
    for candidate in (f"{stem}_depth{ext}", f"{stem}_depth.mp4"):
        if os.path.isfile(candidate):
            return candidate
    folder = os.path.dirname(rgb_path)
    name = os.path.basename(stem)
    for sub in ("depth", "../depth"):
        candidate = os.path.join(folder, sub, f"{name}_depth.mp4")
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    # A GUI that dies silently is impossible to diagnose, and a Qt slot that
    # raises would otherwise take the process down without a word.
    faulthandler.enable()

    def hook(exc_type, exc, tb) -> None:
        traceback.print_exception(exc_type, exc, tb)
        sys.stderr.flush()

    sys.excepthook = hook

    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    args = [a for a in (argv or sys.argv)[1:] if not a.startswith("-")]
    if len(args) >= 2:
        win.rgb_path, win.depth_path = args[0], args[1]
        win._open_session()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
