"""
Timing utilities for SwiftEdit

Provides lightweight wall-clock timers, context managers for scoped timing, and
an aggregator for named timers. Supports optional CUDA synchronization to ensure
accurate measurement of GPU-bound sections when torch is available.

Public API:
- Timer: start/stop/reset with elapsed retrieval (seconds or milliseconds)
- scoped_timer(name=None, sink=None, synchronize=False): context manager for timing code blocks
- MultiTimer: named timers with tic/toc and summary reporting
- cuda_sync(): synchronize CUDA stream if available
- perf_counter_ms(): high-resolution wall clock in milliseconds
- measure_time(fn, *args, synchronize=False, repeat=1, warmup=0, **kwargs): run a callable and measure

These utilities are designed to be dependency-light and robust when torch is not
installed. If torch is present and CUDA is available, enabling synchronize=True
will call torch.cuda.synchronize() before reading or after writing GPU work to
improve timing accuracy.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional torch import for CUDA synchronization
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


def cuda_sync() -> None:
    """Synchronize CUDA if torch is available and CUDA is initialized.

    Safe no-op if torch or CUDA are unavailable.
    """
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            # Ignore synchronization errors to keep timing non-fatal
            pass


def perf_counter_ms() -> float:
    """Return high-resolution wall clock in milliseconds."""
    return time.perf_counter() * 1000.0


class Timer:
    """Simple wall-clock timer with optional CUDA synchronization.

    Usage:
      t = Timer(synchronize=True)
      t.start()
      # ... work ...
      t.stop()
      print(t.elapsed_ms)

    Or with context manager via scoped_timer(...).
    """

    def __init__(self, synchronize: bool = False) -> None:
        self.synchronize = synchronize
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        self._running: bool = False

    def reset(self) -> None:
        self._t0 = None
        self._t1 = None
        self._running = False

    def start(self) -> None:
        if self.synchronize:
            cuda_sync()
        self._t0 = time.perf_counter()
        self._t1 = None
        self._running = True

    def stop(self) -> None:
        if self._t0 is None:
            # If not started, start now to avoid None state
            self._t0 = time.perf_counter()
        if self.synchronize:
            cuda_sync()
        self._t1 = time.perf_counter()
        self._running = False

    @property
    def elapsed(self) -> float:
        """Elapsed seconds between start and stop (or now if running)."""
        if self._t0 is None:
            return 0.0
        t1 = time.perf_counter() if self._running or self._t1 is None else self._t1
        return max(0.0, t1 - self._t0)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds."""
        return self.elapsed * 1000.0

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"Timer(state={state}, elapsed_ms={self.elapsed_ms:.3f}, sync={self.synchronize})"


class _ScopedTimer:
    """Context manager for scoped timing.

    Parameters:
    - name: Optional label to include in logs
    - sink: Optional callable to receive (name, elapsed_ms) upon exit
    - synchronize: Whether to synchronize CUDA on enter/exit
    """

    def __init__(self, name: Optional[str] = None, sink: Optional[Callable[[str, float], None]] = None, synchronize: bool = False) -> None:
        self.name = name or "timer"
        self.sink = sink
        self.timer = Timer(synchronize=synchronize)

    def __enter__(self) -> Timer:
        self.timer.start()
        return self.timer

    def __exit__(self, exc_type, exc, tb) -> None:
        self.timer.stop()
        if self.sink is not None:
            try:
                self.sink(self.name, self.timer.elapsed_ms)
            except Exception:
                # Logging sink failures should not be fatal
                pass


def scoped_timer(name: Optional[str] = None, sink: Optional[Callable[[str, float], None]] = None, synchronize: bool = False) -> _ScopedTimer:
    """Return a scoped timer context manager.

    Example:
      with scoped_timer("inversion", sink=print, synchronize=True) as t:
          eps = inv_net(z, c_y)
      # Automatically logs ("inversion", elapsed_ms)
    """
    return _ScopedTimer(name=name, sink=sink, synchronize=synchronize)


class MultiTimer:
    """Aggregator for multiple named timers.

    Provides tic(name)/toc(name) semantics and accumulates durations per name.
    Optional CUDA synchronization per timer operation.
    """

    def __init__(self, synchronize: bool = False) -> None:
        self.synchronize = synchronize
        self._timers: Dict[str, Timer] = {}
        self._history_ms: Dict[str, List[float]] = {}

    def tic(self, name: str) -> None:
        t = self._timers.get(name)
        if t is None:
            t = Timer(synchronize=self.synchronize)
            self._timers[name] = t
        t.start()

    def toc(self, name: str, record: bool = True) -> float:
        t = self._timers.get(name)
        if t is None:
            # Create and immediately stop to yield 0
            t = Timer(synchronize=self.synchronize)
            self._timers[name] = t
            t.start()
        t.stop()
        ms = t.elapsed_ms
        if record:
            self._history_ms.setdefault(name, []).append(ms)
        return ms

    def elapsed(self, name: str) -> float:
        t = self._timers.get(name)
        return 0.0 if t is None else t.elapsed_ms

    def summary(self, reset: bool = False) -> Dict[str, Dict[str, float]]:
        """Return summary stats per name: count, total_ms, avg_ms, last_ms."""
        out: Dict[str, Dict[str, float]] = {}
        for name, hist in self._history_ms.items():
            total = float(sum(hist)) if hist else 0.0
            count = float(len(hist))
            avg = (total / count) if count > 0 else 0.0
            last = float(hist[-1]) if hist else 0.0
            out[name] = {
                "count": count,
                "total_ms": total,
                "avg_ms": avg,
                "last_ms": last,
            }
        if reset:
            self._history_ms.clear()
        return out

    def clear(self) -> None:
        self._history_ms.clear()
        self._timers.clear()


def measure_time(
    fn: Callable[..., Any],
    *args: Any,
    synchronize: bool = False,
    repeat: int = 1,
    warmup: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Measure wall-clock runtime of a callable.

    Parameters:
    - fn: callable to execute
    - *args/**kwargs: forwarded to the callable
    - synchronize: whether to call cuda_sync() before/after timing
    - repeat: number of times to measure (returns list and average)
    - warmup: number of warmup runs (not measured)

    Returns a dict with keys: {"values_ms": List[float], "avg_ms": float, "repeats": int}
    """
    # Warmup runs (not measured)
    for _ in range(max(0, int(warmup))):
        try:
            fn(*args, **kwargs)
        except Exception:
            # Warmup errors are ignored to avoid failing measurement entirely
            pass

    values: List[float] = []
    for _ in range(max(1, int(repeat))):
        if synchronize:
            cuda_sync()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if synchronize:
            cuda_sync()
        t1 = time.perf_counter()
        values.append((t1 - t0) * 1000.0)
        # Optionally return last result in the payload (not used by default)
        last_result = result  # noqa: F841
    avg = float(sum(values)) / float(len(values)) if values else 0.0
    return {"values_ms": values, "avg_ms": avg, "repeats": len(values)}


__all__ = [
    "Timer",
    "scoped_timer",
    "MultiTimer",
    "cuda_sync",
    "perf_counter_ms",
    "measure_time",
]
