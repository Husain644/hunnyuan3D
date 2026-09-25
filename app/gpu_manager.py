"""VRAM budgeting and CUDA/CUDAGraph helpers tuned for a 14.6 GB T4.

T4 realities:
  * 16 GB physical, ~14.6 GB usable (driver + context reservation).
  * fp16 is the fastest dtype; T4 has no native bf16 tensor cores.
  * Max CUDA context from a single process: one compute stream.

The manager tracks the *serialized* pipeline stages (shape → texture →
baking). It never allows two heavy stages to be resident at once, so peak
VRAM stays under the usable ceiling even though the weights (~16-29 GB) far
exceed it.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from .config import T4_USABLE_GB

logger = logging.getLogger("hy3d.gpu")


def _torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError:  # pragma: no cover - CPU-only preview box
        return None


@dataclass
class GpuInfo:
    available: bool
    name: Optional[str] = None
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0


def gpu_info() -> GpuInfo:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return GpuInfo(available=False)
    try:
        torch.cuda.init()
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        if hasattr(torch.cuda, "mem_get_info"):
            free, _tot = torch.cuda.mem_get_info(idx)
            free_gb = free / (1024**3)
        else:
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            free_gb = total - reserved
        return GpuInfo(
            available=True,
            name=name,
            total_gb=total,
            free_gb=free_gb,
            used_gb=total - free_gb,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("GPU introspection failed: %s", exc)
        return GpuInfo(available=False)


class GpuManager:
    """Serializes heavy pipeline stages and enforces VRAM budgets.

    Pipeline stages acquire the manager through ``stage_lock(i)`` with the
    same key so a shape and texture stage can never overlap on the GPU.
    """

    def __init__(self, budgets: dict[str, float], total_gb: float = T4_USABLE_GB) -> None:
        self.budgets = budgets
        self.total_gb = total_gb
        self._lock = threading.Lock()
        self._resident: dict[str, str] = {}  # key -> stage label
        self._stage_locks: dict[str, threading.Lock] = {}

    def stage_lock(self, stage: str) -> threading.Lock:
        """Return a per-stage reentrant-free lock used to serialize loads."""
        with self._lock:
            lock = self._stage_locks.get(stage)
            if lock is None:
                lock = threading.Lock()
                self._stage_locks[stage] = lock
            return lock

    # -- load/unload bookkeeping -----------------------------------------
    def mark_resident(self, stage: str, label: str) -> None:
        with self._lock:
            self._resident[stage] = label

    def mark_unresolved(self, stage: str) -> None:
        with self._lock:
            self._resident.pop(stage, None)

    def resident_snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._resident)

    def would_fit(self, stage: str, extra_gb: float = 0.0) -> bool:
        """Heuristic: predicted budget must stay under the ceiling."""
        budget = self.budgets.get(stage, 0.0) + extra_gb
        # We never co-load two heavy stages, so comparing a single budget
        # against the usable ceiling is the correct proxy.
        proxy = gpu_info()
        if not proxy.available:
            return True  # CPU fallback path
        if proxy.total_gb <= 0:
            return True
        ceiling = min(self.total_gb, proxy.total_gb)
        return budget <= ceiling

    @staticmethod
    def empty_cache() -> None:
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def force_offload(model, device: str = "cpu") -> None:
        """Move every parameter/buffer to CPU and free CUDA cache.

        Uses ``module.to('cpu')`` which is deterministic and independent of
        accelerate hooks, so it works even for pipelines that don't ship
        ``enable_model_cpu_offload``.
        """
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            model.to(device)
        except AttributeError as exc:
            # Not a torch module (e.g. the vendored paint pipeline, which is a
            # plain class owning internal models). Offload is best-effort, so a
            # missing .to must never fail the job that already finished.
            logger.info("%s has no .to(%s); skipping offload (%s)", type(model).__name__, device, exc)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("Model .to(cpu) failed (%s); falling back to hooks", exc)
            offload = getattr(model, "enable_model_cpu_offload", None)
            if callable(offload):
                return offload()
            logger.warning("No offload hook either; leaving %s in place", type(model).__name__)
            return
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Forced offload of %s to %s", type(model).__name__, device)

    @staticmethod
    def release(model) -> None:
        """Free a torch module's parameters/buffers in place (GPU + CPU).

        Whereas ``force_offload`` moves every tensor to CPU (which has to
        *first* allocate host RAM equal to the model size, then release the
        CUDA copy — a moment of double residency that OOMs small-RAM boxes),
        ``release`` nulls the tensors directly and empties the CUDA cache, so
        no CPU copy is ever made. Exact for models we will never touch again.
        """
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            mods = list(model.modules()) if callable(getattr(model, "modules", None)) else []
        except Exception:  # noqa: BLE001
            mods = []
        if not mods:
            mods = [model]
        for m in mods:
            for container in (getattr(m, "_parameters", None),
                              getattr(m, "_buffers", None)):
                for k in list(container or {}):
                    if isinstance(container[k], torch.Tensor):
                        container[k] = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Released %s from GPU+CPU", type(model).__name__)

    @staticmethod
    def recommended_chunk() -> int:
        """VAE decode chunk hint: lower on small cards to cut peak VRAM."""
        info = gpu_info()
        if not info.available:
            return 4000
        if info.free_gb < 8:
            return 4000
        if info.free_gb < 12:
            return 8000
        return 16000


class GpuAdmission:
    """VRAM-aware concurrency guard for GPU-resident jobs.

    ``auto`` mode derives the cap from live VRAM: ``floor(total_gb / per_job)``
    so a 22 GB card admits 2 concurrent jobs (each ~9 GB), while a 14.6 GB T4
    stays at 1. A numeric hard cap from config pins the ceiling instead.

    Admission also re-checks *free* VRAM before admitting a second job so we
    never co-load more checkpoints than the card currently holds. Call as an
    async context manager; pairing works as ``async with admission: ...`` with
    ``__aexit__`` releasing the slot.
    """

    def __init__(self, jobs_vram_gb: float, hard_cap: Optional[int] = None) -> None:
        self.jobs_vram_gb = max(2.0, jobs_vram_gb)
        self.hard_cap = hard_cap
        self._active = 0
        self._mutex = threading.Lock()

    def cap(self) -> int:
        """Max concurrent GPU jobs right now (auto from VRAM or fixed)."""
        if self.hard_cap is not None:
            return max(1, self.hard_cap)
        info = gpu_info()
        if not info.available or info.total_gb <= 0:
            return 2  # CPU fallback: small bounded default
        return max(1, int(info.total_gb // self.jobs_vram_gb))

    def can_fit(self) -> bool:
        """Enough *free* VRAM for another checkpoint (slack 1 GB)."""
        info = gpu_info()
        if not info.available:
            return True
        return info.free_gb >= max(1.0, self.jobs_vram_gb - 1.0)

    async def acquire(self) -> None:
        while True:
            if self._try_acquire():
                return
            await asyncio.sleep(0.5)

    async def __aenter__(self) -> "GpuAdmission":
        await self.acquire()
        return self

    async def __aexit__(self, *_exc) -> None:
        self.release()

    def _try_acquire(self) -> bool:
        cap = self.cap()
        with self._mutex:
            if self._active >= cap:
                return False
            if self._active and not self.can_fit():
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._mutex:
            self._active = max(0, self._active - 1)


gpu_manager = GpuManager({})


def init_gpu_manager(budgets: dict[str, float]) -> None:
    global gpu_manager
    gpu_manager = GpuManager(budgets)