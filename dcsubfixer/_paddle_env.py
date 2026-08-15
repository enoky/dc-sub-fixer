"""Making Paddle importable in a process, on Windows.

Two separate problems are handled here, both of which only bite on Windows.

1. Paddle's CUDA 13 DLL search path (see `apply`).
2. Paddle and PyTorch bundle different cuDNN 9.x builds. Windows resolves DLLs
   by base name process-wide, so whichever framework loads first wins and the
   other dies with WinError 127. paddlex imports modelscope, which imports
   torch, so merely importing paddleocr is enough to trigger it. `isolate` cuts
   that import chain (see `isolate`).

Because of (2) the OCR stage runs in its own process; see detect_worker.py.

Original DLL-path note:

paddlepaddle-gpu's Windows loader adds `site-packages/nvidia/<pkg>/bin` to the
DLL search path, but the CUDA 13 wheels put their DLLs one level deeper, in
`bin/x86_64`. cuDNN itself loads fine (it lives at the expected depth), then
fails to resolve cublas64_13.dll and the import dies with

    OSError: [WinError 127] ... Error loading "...cudnn_cnn64_9.dll"

only once a convolution is actually needed. Registering the deeper directories
before Paddle is imported fixes it. Harmless on other platforms and on
installs that do not have the problem.
"""

from __future__ import annotations

import glob
import os
import site
import sys
import types
from typing import List

_applied = False
# os.add_dll_directory returns a handle that de-registers the directory when it
# is garbage collected, so these have to outlive the call.
_handles: List[object] = []


def _candidate_site_dirs() -> List[str]:
    dirs = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        dirs.append(user_site)
    dirs.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def apply() -> List[str]:
    """Register nested nvidia DLL directories. Returns the paths added."""
    global _applied
    if _applied or sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []
    _applied = True

    added = []
    for site_dir in _candidate_site_dirs():
        for path in glob.glob(os.path.join(site_dir, "nvidia", "*", "bin", "x86_64")):
            if os.path.isdir(path):
                try:
                    _handles.append(os.add_dll_directory(path))
                except OSError:
                    continue
                added.append(path)
    return added


def isolate() -> None:
    """Keep torch out of this process so Paddle's cuDNN wins.

    paddlex imports modelscope unconditionally, and modelscope imports torch at
    module scope. modelscope is only ever used by paddlex as an alternative
    model *download* host, and the default host is huggingface, so standing in
    a placeholder module costs nothing and breaks the chain to torch.

    Call before importing paddleocr, and only in a process that has no other
    use for torch.
    """
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is already imported in this process; Paddle's cuDNN will "
            "fail to load. Run the OCR stage in a separate process."
        )
    if "modelscope" not in sys.modules:
        stub = types.ModuleType("modelscope")
        stub.__doc__ = "Placeholder installed by dc-sub-fixer; see _paddle_env.isolate()."
        # No __path__, so `import modelscope.hub.errors` raises ModuleNotFoundError,
        # which paddlex already catches and treats as "ModelScope unavailable".
        sys.modules["modelscope"] = stub

    # The hoster reachability probe costs several seconds and needs the network;
    # model weights are cached locally after the first run.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
