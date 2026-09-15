"""GLB + thumbnails persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hy3d.storage")


def _safe_name(job_id: str) -> str:
    return "".join(c for c in job_id if c.isalnum() or c in "_-.") or "job"


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