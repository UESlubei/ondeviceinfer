"""Opt-in profiling annotations shared by benchmarkable runtime paths.

Set ``ONDEVICEINFER_NVTX=1`` before importing :mod:`ondeviceinfer` to emit
NVTX ranges.  Keeping the switch at module scope makes the normal inference
path pay only a cheap branch when profiling is disabled.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

import torch


_NVTX_ENABLED = os.environ.get("ONDEVICEINFER_NVTX", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def nvtx_enabled() -> bool:
    """Return whether this process was started with NVTX annotations enabled."""
    return _NVTX_ENABLED


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Emit one CUDA NVTX range when profiling is enabled.

    ``torch.cuda.nvtx`` is deliberately never touched on CPU-only runs.  This
    keeps unit tests and the CPU emulation backend independent of CUDA/NVTX.
    """
    if not _NVTX_ENABLED:
        yield
        return

    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
