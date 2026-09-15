"""In-process async job queue with durable state on disk."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .schemas import JobStatus
from .storage import Storage

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
    """

    def __init__(self, storage: Storage, run_fn: _JobFn, max_active: int = 1) -> None:
        self.storage = storage
        self.run_fn = run_fn
        self.max_active = max_active
        self._active = 0
        self._jobs: dict[str, _Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

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
        try:
            result = await self.run_fn(job.job_id, job.payload)
            job.set_status(
                JobStatus.SUCCEEDED,
                stage="done",
                message="completed",
                progress=1.0,
                result_url=result.get("result_url"),
                error=result.get("error"),
            )
            self.storage.save_meta(job.job_id, job.snapshot())
        except asyncio.CancelledError:
            job.set_status(JobStatus.CANCELLED, message="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("Job %s failed:\n%s", job.job_id, tb)
            job.set_status(JobStatus.FAILED, stage="error", message="failed",
                           error=str(exc)[:2000])
            self.storage.save_meta(job.job_id, job.snapshot())

    # -- public API ---------------------------------------------------------
    def submit(self, job_id: str, payload: dict) -> None:
        job = _Job(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        self.storage.save_meta(job_id, job.state)
        self._queue.put_nowait(job_id)

    def get(self, job_id: str) -> Optional[_Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancelled.set()
        if job.state.get("status") == JobStatus.QUEUED.value:
            job.set_status(JobStatus.CANCELLED, message="cancelled")
            self.storage.save_meta(job_id, job.snapshot())
        return True

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

    def shutdown(self) -> None:
        for _j in self._jobs.values():
            pass