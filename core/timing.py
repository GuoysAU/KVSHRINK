"""Timing helpers shared by the generation path and the scoring paths."""

import time

import torch


def sync_cuda() -> None:
    """Block until queued CUDA work finishes; no-op without CUDA.

    Without it an asynchronously queued kernel is charged to whichever phase
    synchronizes next, not the one that launched it.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseTimer:
    """Measure wall-clock time with synchronized CUDA boundaries."""

    elapsed: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        sync_cuda()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        sync_cuda()
        self.elapsed = time.perf_counter() - self._start
