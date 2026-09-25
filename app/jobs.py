"""In-process async job queue with durable state on disk."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .schemas import JobStatus
from .storage import PayloadStore, Storage

logger = logging.getLogger("hy3d.jobs")

_JobFn = Callable[[str, dict], Awaitable[dict]]


def _field_defaults(stage: Optional[str] = None, message: str = "") -> dict:
    now = time.time()
    return {
        "status": JobStatus.QUEUED.value,
        "progress": 0.0,
        "stage": stage,
        "message": message,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "result_url": None,
    }


class _Job:
    def __init__(self, job_id: str, payload: dict) -> None:
        self.job_id = job_id
        self.payload = payload
        self.state = _field_defaults(payload.get("stage"), payload.get("message", "queued"))
        self.lock = threading.Lock()
        self.cancelled = threading.Event()

    def touch(self) -> None:
        self.state["updated_at"] = time.time()

    def set_status(self, status: JobStatus, stage: Optional[str] = None,
                   message: Optional[str] = None, progress: Optional[float] = None,
                   error: Optional[str] = None, result_url: Optional[str] = None) -> None:
        with self.lock:
            self.state["status"] = status.value
            if stage is not None:
                self.state["stage"] = stage
            if message is not None:
                self.state["message"] = message
            if progress is not None:
                self.state["progress"] = max(0.0, min(1.0, progress))
            if error is not None:
                self.state["error"] = error
            if result_url is not None:
                self.state["result_url"] = result_url
            self.touch()

    def snapshot(self, public: bool = True) -> dict:
        with self.lock:
            s = {"job_id": self.job_id, **dict(self.state)}
            return s


class JobManager:
    """Very small job orchestrator.

    * ``submit`` creates the job and schedules background execution.
    * A single worker drives the pipeline; concurrency is capped by
      ``max_active`` so overlapping heavy stages never double-resident on GPU.
    * Jobs beyond capacity wait in the asyncio queue and report QUEUED.
    * If ``records`` (a :class:`PayloadStore`) is given, QUEUED/RUNNING jobs
      are persisted to disk and re-queued after a server restart, so an
      unexpected crash no longer surfaces as an auto-cancelled job.
    """

    def __init__(self, storage: Storage, run_fn: _JobFn, max_active: int = 1,
                 records: Optional[PayloadStore] = None,
                 global_slot: Optional[asyncio.Semaphore] = None) -> None:
        self.storage = storage
        self.run_fn = run_fn
        self.max_active = max_active
        self.records = records
        self._slot = global_slot
        self._active = 0
        self._jobs: dict[str, _Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._lock = threading.Lock()

    def _persist(self, job: _Job) -> None:
        if self.records is not None:
            try:
                self.records.save(job.job_id, job.payload, job.state)
            except Exception:  # noqa: BLE001
                logger.warning("payload persist failed for %s", job.job_id)

    def recover(self) -> int:
        """Re-queue any persisted QUEUED/RUNNING jobs after a restart.

        Returns the number of jobs recovered (0 if no records store).
        Terminal records are discarded. Payload images are decoded from the
        base64 disk form back to bytes so ``run_fn`` sees identical input.
        """
        if self.records is None:
            return 0
        recovered = 0
        for job_id, payload, state in self.records.iter_records():
            if state.get("status") not in (JobStatus.QUEUED.value,
                                           JobStatus.RUNNING.value):
                self.records.discard(job_id)
                continue
            job = _Job(job_id, payload)
            job.state = dict(state)
            with self._lock:
                self._jobs[job_id] = job
            self._queue.put_nowait(job_id)
            recovered += 1
        if recovered:
            logger.info("recovered %d in-flight job(s) after restart", recovered)
        return recovered

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._workers and all(not t.done() for t in self._workers):
            return
        self._workers = [
            loop.create_task(self._worker())
            for _ in range(max(1, self.max_active))
        ]

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            if job.cancelled.is_set():
                job.set_status(JobStatus.CANCELLED, message="cancelled before start")
                self._queue.task_done()
                continue
            ctx = self._slot if self._slot is not None else nullcontext()
            async with ctx:
                with self._lock:
                    self._active += 1
                try:
                    await self._run(job)
                finally:
                    with self._lock:
                        self._active -= 1
                    self._queue.task_done()

    async def _run(self, job: _Job) -> None:
        job.set_status(JobStatus.RUNNING, message="shape stage starting")
        self._persist(job)
        try:
            result = await self.run_fn(job.job_id, job.payload)
            if job.cancelled.is_set():
                # cancelled (or paused) while the blocking stage ran: drop the
                # result, mark cancelled, and forget the job so it can't linger.
                job.set_status(JobStatus.CANCELLED, message="cancelled")
                self.storage.save_meta(job.job_id, job.snapshot())
                self.forget(job.job_id)
                return
            job.set_status(
                JobStatus.SUCCEEDED,
                stage="done",
                message="completed",
                progress=1.0,
                result_url=result.get("result_url"),
                error=result.get("error"),
            )
            self.storage.save_meta(job.job_id, job.snapshot())
            if self.records is not None:
                self.records.discard(job.job_id)
        except asyncio.CancelledError:
            job.set_status(JobStatus.CANCELLED, message="cancelled")
            if self.records is not None:
                self.records.discard(job.job_id)
            raise
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("Job %s failed:\n%s", job.job_id, tb)
            job.set_status(JobStatus.FAILED, stage="error", message="failed",
                           error=str(exc)[:2000])
            self.storage.save_meta(job.job_id, job.snapshot())
            if self.records is not None:
                self.records.discard(job.job_id)

    # -- public API ---------------------------------------------------------
    def submit(self, job_id: str, payload: dict) -> None:
        job = _Job(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        self.storage.save_meta(job_id, job.state)
        self._persist(job)
        self._queue.put_nowait(job_id)

    def get(self, job_id: str) -> Optional[_Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancelled.set()
        if job.state.get("status") in (JobStatus.QUEUED.value,
                                       JobStatus.RUNNING.value):
            job.set_status(JobStatus.CANCELLED, message="cancelled")
            self.storage.save_meta(job_id, job.snapshot())
        if self.records is not None:
            self.records.discard(job_id)
        return True

    def reset(self) -> int:
        """Cancel all queued/running jobs and forget every job record.

        Used from the in-page "reset" button to clear a wedged job or
        leftover task state without restarting the process.
        """
        with self._lock:
            ids = list(self._jobs)
        cleared = 0
        for job_id in ids:
            job = self.get(job_id)
            if job is None:
                continue
            status = job.state.get("status")
            if status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                job.cancelled.set()
                job.set_status(JobStatus.CANCELLED,
                               message="cancelled by reset")
            try:
                self.forget(job_id)
            except Exception:  # noqa: BLE001
                logger.warning("reset: forget %s failed", job_id)
            cleared += 1
        logger.info("Reset cleared %d job record(s)", cleared)
        return cleared

    def list_states(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._jobs.values())[-limit:]
        return [j.snapshot() for j in items]

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def queue_depth(self) -> int:
        return max(0, self._queue.qsize() - self._active)

    def forget(self, job_id: str) -> bool:
        """Drop a finished job from memory and storage (used on RAM stores)."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            try:
                self.storage.discard(job_id)
            except Exception:  # noqa: BLE001
                pass
        if self.records is not None:
            try:
                self.records.discard(job_id)
            except Exception:  # noqa: BLE001
                pass
        return job is not None

    def sweep_finished(self, ttl_seconds: float = 3600.0) -> int:
        """Remove succeeded/failed/cancelled jobs older than ``ttl`` seconds."""
        now = time.time()
        done = {"succeeded", "failed", "cancelled"}
        stale: list[str] = []
        for jid, j in list(self._jobs.items()):
            if j.state.get("status") in done and now - j.state.get("updated_at", 0) > ttl_seconds:
                stale.append(jid)
        for jid in stale:
            self.forget(jid)
        return len(stale)

    def shutdown(self) -> None:
        for _j in self._jobs.values():
            pass