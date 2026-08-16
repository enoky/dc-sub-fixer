"""Video decode/encode helpers built on PyAV.

Depth frames are read, edited and written in the source's **own planar YUV
format**, and only the luma plane is touched. Anything else loses precision:

  * rgb24 throws away three quarters of a 10-bit depth map before compositing
    even starts;
  * gray16 keeps the bit depth but forces a limited-range (16-235) to
    full-range conversion in each direction, and the rounding at both ends
    moves pixels by a whole 10-bit step;
  * any RGB round trip also puts the YUV matrix in the path, which rounds.

Staying in the native format and encoding losslessly, frames this tool does
not modify come back bit-identical. That matters because a depth map is not a
picture to be looked at, it is data a downstream stereo stage samples.

RGB sources are read as 8-bit rgb24, which is all the detector and segmenter
need.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterator, List, Optional

import av
import numpy as np

# Planar YUV formats we can pass through untouched. In every one of these the
# luma plane is simply the first `height` rows of the array PyAV hands back.
PLANAR_OK = {
    "yuv420p", "yuv422p", "yuv444p",
    "yuv420p10le", "yuv422p10le", "yuv444p10le",
    "yuv420p12le", "yuv422p12le", "yuv444p12le",
    "gray", "gray10le", "gray12le", "gray16le",
}


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    n_frames: int  # 0 when the container does not report it
    pix_fmt: Optional[str]
    codec: Optional[str] = None

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def bit_depth(self) -> int:
        fmt = self.pix_fmt or ""
        for depth in (16, 14, 12, 10):
            if str(depth) in fmt:
                return depth
        return 8

    def __str__(self) -> str:
        n = self.n_frames if self.n_frames else "?"
        return (
            f"{self.width}x{self.height} @ {float(self.fps):.3f}fps, {n} frames, "
            f"{self.pix_fmt} ({self.bit_depth}-bit)"
        )


def probe(path: str) -> VideoInfo:
    with av.open(path) as container:
        stream = container.streams.video[0]
        n_frames = stream.frames or 0
        if not n_frames and stream.duration and stream.average_rate:
            n_frames = int(float(stream.duration * stream.time_base * stream.average_rate))
        return VideoInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=stream.average_rate or Fraction(24, 1),
            n_frames=n_frames,
            pix_fmt=stream.codec_context.pix_fmt,
            codec=stream.codec_context.name,
        )


def read_frames(path: str, fmt: str = "rgb24") -> Iterator[np.ndarray]:
    """Yield successive frames.

    `fmt="rgb24"` gives HxWx3 uint8; `fmt="gray16le"` gives HxW uint16 luma.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format=fmt)


def depth_format(info: "VideoInfo") -> str:
    """The pixel format to read and write a depth video in.

    The source's own format when it is one we can pass through, so untouched
    pixels never go through a conversion; otherwise the nearest planar format
    with the same bit depth.
    """
    if info.pix_fmt in PLANAR_OK:
        return info.pix_fmt
    return "yuv420p10le" if info.bit_depth > 8 else "yuv420p"


def read_depth(path: str, fmt: str) -> Iterator[np.ndarray]:
    """Yield whole planar frames, luma plane first.

    The returned array is the raw plane layout, e.g. (H*3/2, W) for 4:2:0. Use
    `luma()` to address the part that matters.
    """
    return read_frames(path, fmt=fmt)


def luma(planar: np.ndarray, height: int, width: int) -> np.ndarray:
    """A view of the luma plane inside a planar YUV frame."""
    return planar[:height, :width]


def chroma_is_neutral(planar: np.ndarray, height: int, bit_depth: int, tol: int = 2) -> bool:
    """Whether the colour planes are flat, i.e. the frame really is grey."""
    rest = planar[height:]
    if rest.size == 0:
        return True  # a gray* format has no chroma at all
    neutral = 1 << (bit_depth - 1)
    return bool(np.abs(rest.astype(np.int32) - neutral).max() <= tol)


def sample_depth(path: str, count: int = 6) -> List[np.ndarray]:
    """Grab a handful of depth frames spread through a video, as 8-bit grey."""
    info = probe(path)
    total = info.n_frames
    wanted = (
        {int(total * (i + 0.5) / count) for i in range(count)}
        if total
        else set(range(0, count * 30, 30))
    )
    out = []
    for idx, frame in enumerate(read_frames(path, fmt="gray")):
        if idx in wanted:
            out.append(frame)
            if len(out) >= count:
                break
        elif total and idx > max(wanted):
            break
    return out


class FrameReader:
    """Random access to video frames, by frame index.

    PyAV decodes forward, so a jump means seeking to the keyframe at or before
    the target and decoding up to it. Frame identity comes from the packet
    timestamp rather than a running count, because a seek lands wherever the
    keyframe is and counting from there drifts.
    """

    # Decoding forward is cheaper than a seek for small hops, which is the
    # common case when stepping through frames.
    FORWARD_LIMIT = 48

    def __init__(self, path: str, fmt: str = "rgb24", cache_size: int = 8) -> None:
        self.path = path
        self.fmt = fmt
        self.info = probe(path)
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._time_base = self._stream.time_base
        self._rate = self._stream.average_rate
        self._start = self._stream.start_time or 0
        self._decoder = None
        self._position = -1  # index of the last frame handed out
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size

    # -- index/timestamp conversion -------------------------------------
    def _index_of(self, frame) -> Optional[int]:
        pts = frame.pts if frame.pts is not None else frame.dts
        if pts is None:
            return None
        return int(round(float((pts - self._start) * self._time_base * self._rate)))

    def _pts_of(self, index: int) -> int:
        return int(index / self._rate / self._time_base) + self._start

    # -- cache ----------------------------------------------------------
    def _remember(self, index: int, arr: np.ndarray) -> None:
        if index in self._cache:
            return
        self._cache[index] = arr
        self._cache_order.append(index)
        while len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]

    # -- reading --------------------------------------------------------
    def _restart(self, index: int) -> None:
        self._container.seek(self._pts_of(index), stream=self._stream, backward=True)
        self._decoder = self._container.decode(self._stream)
        self._position = -1

    def frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        hit = self._cache.get(index)
        if hit is not None:
            return hit

        if self._decoder is None or index <= self._position or index - self._position > self.FORWARD_LIMIT:
            self._restart(index)

        last = None
        for frame in self._decoder:
            idx = self._index_of(frame)
            if idx is None:
                continue
            self._position = idx
            if idx > index:
                # Overshot: the seek landed past the target, back up further.
                break
            last = frame
            if idx == index:
                arr = frame.to_ndarray(format=self.fmt)
                self._remember(index, arr)
                return arr

        if last is not None and self._index_of(last) is not None:
            # Ran out of frames; the clip may be shorter than asked for.
            arr = last.to_ndarray(format=self.fmt)
            self._remember(index, arr)
            return arr
        raise IndexError(f"frame {index} not found in {self.path}")

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:
            pass

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DepthWriter:
    """Encodes whole planar YUV frames to a video file.

    Lossless by default, and in the source's own pixel format, so a frame that
    was not modified is reproduced exactly.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: Fraction,
        pix_fmt: str,
        codec: str = "libx264",
        quality: str = "lossless",
        preset: str = "slow",
    ) -> None:
        self.pix_fmt = pix_fmt
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt

        options = {"preset": preset}
        if quality == "lossless":
            # qp=0 is x264/x265's true lossless mode; crf=0 is not, at 10-bit.
            options["qp"] = "0"
            if codec == "libx265":
                options = {"preset": preset, "x265-params": "lossless=1"}
            elif codec == "ffv1":
                options = {}
        else:
            options["crf"] = str(int(quality))
        self.stream.options = options
        self._closed = False

    def write(self, planar: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(planar), format=self.pix_fmt)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()

    def __enter__(self) -> "DepthWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Encodes 8-bit RGB frames. Used by the test helpers, not the depth path."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: Fraction,
        codec: str = "libx264",
        crf: int = 12,
        preset: str = "slow",
        pix_fmt: str = "yuv420p",
    ) -> None:
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt
        options = {"preset": preset}
        if codec in ("libx264", "libx265"):
            options["crf"] = str(crf)
        self.stream.options = options
        self._closed = False

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
