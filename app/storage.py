"""GLB + thumbnails persistence."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("hy3d.storage")


def _safe_name(job_id: str) -> str:
    return "".join(c for c in job_id if c.isalnum() or c in "_-.") or "job"


class PayloadStore:
    """Disk-backed record of in-flight job payloads, for restart recovery.

    Only jobs that are still QUEUED/RUNNING are kept; the record is removed
    the moment a job reaches a terminal state. Payload images (bytes) are
    base64-encoded so the whole record is a single JSON file::

        recovery/{job_id}.json   -> {"payload": {...}, "state": {...}}
    """

    def __init__(self, dir: Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{_safe_name(job_id)}.json"

    @staticmethod
    def _encode(payload: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, bytes):
                out[k] = {"$bin": base64.b64encode(v).decode("ascii")}
            elif isinstance(v, dict):
                out[k] = PayloadStore._encode(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _decode(record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, dict) and set(v) == {"$bin"} and isinstance(v.get("$bin"), str):
                out[k] = base64.b64decode(v["$bin"])
            elif isinstance(v, dict):
                out[k] = PayloadStore._decode(v)
            else:
                out[k] = v
        return out

    def save(self, job_id: str, payload: dict[str, Any], state: dict[str, Any]) -> None:
        data = json.dumps(
            {"payload": self._encode(payload), "state": state},
            ensure_ascii=False,
        )
        with self._lock:
            tmp = self._path(job_id).with_suffix(".json.tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self._path(job_id))

    def load(self, job_id: str) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        p = self._path(job_id)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            return self._decode(rec.get("payload") or {}), rec.get("state") or {}
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt recovery record for %s", job_id)
            return None

    def iter_records(self) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
        with self._lock:
            names = [p.name for p in self.dir.glob("*.json")]
        for name in names:
            job_id = _safe_name(Path(name).stem)
            try:
                rec = json.loads((self.dir / name).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            yield job_id, self._decode(rec.get("payload") or {}), rec.get("state") or {}

    def discard(self, job_id: str) -> None:
        with self._lock:
            p = self._path(job_id)
            if p.exists():
                p.unlink(missing_ok=True)
            t = p.with_suffix(".json.tmp")
            if t.exists():
                t.unlink(missing_ok=True)


class Storage:
    """Manages the outputs directory: job metadata, GLBs, thumbnails.

    Layout::

        outputs/
          glb/{job_id}.glb
          jobs/{job_id}.json
          thumbs/{job_id}.png
    """

    def __init__(self, glb_dir: Path, jobs_dir: Path) -> None:
        self.glb_dir = Path(glb_dir)
        self.jobs_dir = Path(jobs_dir)
        self.thumbs_dir = self.jobs_dir.parent / "thumbs"
        for d in (self.glb_dir, self.jobs_dir, self.thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- files --------------------------------------------------------------
    def glb_path(self, job_id: str) -> Path:
        return self.glb_dir / f"{_safe_name(job_id)}.glb"

    def meta_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{_safe_name(job_id)}.json"

    def thumb_path(self, job_id: str) -> Path:
        return self.thumbs_dir / f"{_safe_name(job_id)}.png"

    # -- meta -----------------------------------------------------------------
    def save_meta(self, job_id: str, data: dict[str, Any]) -> None:
        p = self.meta_path(job_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def load_meta(self, job_id: str) -> Optional[dict[str, Any]]:
        p = self.meta_path(job_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt job meta for %s", job_id)
            return None

    # -- glb ------------------------------------------------------------------
    def store_glb(self, job_id: str, src: Path) -> Path:
        dst = self.glb_path(job_id)
        shutil.copyfile(src, dst)
        return dst

    def read_glb(self, job_id: str) -> Optional[bytes]:
        p = self.glb_path(job_id)
        if p.exists():
            return p.read_bytes()
        return None

    def glb_sha256(self, job_id: str) -> str:
        data = self.read_glb(job_id) or b""
        return hashlib.sha256(data).hexdigest()

    # -- thumbnail ---------------------------------------------------------------
    def store_thumbnail(self, job_id: str, image_bytes: bytes) -> Optional[Path]:
        p = self.thumb_path(job_id)
        try:
            p.write_bytes(image_bytes)
            return p
        except OSError as exc:  # pragma: no cover
            logger.warning("thumb write failed: %s", exc)
            return None


class MemoryStorage:
    """RAM-only duck-type of Storage: no files are ever written to disk.

    GLBs and metadata live in in-memory dicts; ``discard`` drops a job
    entirely (used by the public pipeline so nothing persists on the host).
    """

    def __init__(self) -> None:
        self._meta: dict[str, dict[str, Any]] = {}
        self._glb: dict[str, bytes] = {}
        self._thumbs: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def save_meta(self, job_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._meta[job_id] = dict(data)

    def load_meta(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            m = self._meta.get(job_id)
            return dict(m) if m is not None else None

    def store_glb(self, job_id: str, src: Path) -> Path:
        # src is a path from pipeline export; keep the bytes copy in RAM.
        with self._lock:
            self._glb[job_id] = Path(src).read_bytes()
        return Path("/") / "memory" / f"{job_id}.glb"

    def store_glb_bytes(self, job_id: str, data: bytes) -> None:
        with self._lock:
            self._glb[job_id] = data

    def read_glb(self, job_id: str) -> Optional[bytes]:
        with self._lock:
            return self._glb.get(job_id)

    def glb_sha256(self, job_id: str) -> str:
        data = self.read_glb(job_id) or b""
        return hashlib.sha256(data).hexdigest()

    def store_thumbnail(self, job_id: str, image_bytes: bytes) -> Optional[Path]:
        with self._lock:
            self._thumbs[job_id] = image_bytes
            return Path("/") / "memory" / f"{job_id}.png"

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._meta.pop(job_id, None)
            self._glb.pop(job_id, None)
            self._thumbs.pop(job_id, None)