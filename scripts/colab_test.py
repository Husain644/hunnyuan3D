#!/usr/bin/env python3
"""
colab_test.py — Automated end-to-end smoke test for Hunyuan3D 2.1 API.

Standalone Python (no external deps beyond the service's own requirements).
  - Spawns colab_run.sh if /health is not reachable.
  - POSTs a synthetic test image as JSON base64.
  - Polls until the job reaches a terminal status.
  - Downloads the GLB and validates it with trimesh.
  - Best-effort cancel test.
  - Prints PASS / FAIL.

Usage:
    python scripts/colab_test.py
    HY3D_TEST_TIMEOUT=600 python scripts/colab_test.py   # first-run may download ~7 GB
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
BASE = os.getenv("HY3D_TEST_BASE", "http://127.0.0.1:8080")
TIMEOUT = int(os.getenv("HY3D_TEST_TIMEOUT", "2400"))   # first-run model download
POLL_INTERVAL = 5

# ── helpers ──────────────────────────────────────────────────────────────
def _info(msg: str) -> None:
    print(f"[INFO]  {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"\nFAIL — {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _req(method: str, path: str, data: bytes | None = None,
         headers: dict | None = None) -> tuple[int, dict | bytes]:
    """Minimal urllib.request wrapper returning (status, body)."""
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, data=data, method=method,
                                headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ct = resp.headers.get("Content-Type", "")
            body = resp.read()
            if "json" in ct:
                return resp.status, json.loads(body)
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        raise RuntimeError(f"HTTP {method} {path} failed: {exc}") from exc


def _make_test_image_png_base64(size: int = 256) -> str:
    """Generate a solid-colour test PNG in memory."""
    from PIL import Image

    img = Image.new("RGB", (size, size), color=(128, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── server bootstrap ─────────────────────────────────────────────────────
def _ensure_server() -> None:
    try:
        status, _ = _req("GET", "/health")
        if status == 200:
            _info("Server already running")
            return
    except Exception:
        pass

    _info("Server not reachable; spawning colab_run.sh …")
    sh = SERVICE_ROOT / "scripts" / "colab_run.sh"
    if not sh.exists():
        _fail(f"{sh} not found")
    proc = subprocess.Popen(
        ["bash", str(sh)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(SERVICE_ROOT),
    )
    # Wait up to 120 s for the shell script to bring the server up.
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(3)
        try:
            status, _ = _req("GET", "/health")
            if status == 200:
                _info(f"Server up (spawned, pid={proc.pid})")
                return
        except Exception:
            pass
    _fail("colab_run.sh did not bring the server up within 120 s")


# ── tests ────────────────────────────────────────────────────────────────
def test_health() -> dict:
    _info("GET /health")
    status, body = _req("GET", "/health")
    if status != 200:
        _fail(f"/health returned {status}: {body}")
    _info(f"  gpu={body.get('gpu_name')}  vram={body.get('vram_gb')}GB")
    return body


def test_submit_and_wait() -> dict:
    _info("POST /v1/generate  (JSON base64, shape-only)")
    payload = json.dumps({
        "image": _make_test_image_png_base64(),
        "enable_texture": False,
        "octree_resolution": 256,
    }).encode()
    status, body = _req("POST", "/v1/generate", data=payload,
                        headers={"Content-Type": "application/json"})
    if status != 202:
        _fail(f"/v1/generate returned {status}: {body}")
    job_id = body.get("job_id")
    _info(f"  job_id={job_id}")

    _info(f"Polling /v1/jobs/{job_id} (timeout {TIMEOUT}s) …")
    deadline = time.time() + TIMEOUT
    last_status = ""
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        s, j = _req("GET", f"/v1/jobs/{job_id}")
        if s != 200:
            _fail(f"poll returned {s}: {j}")
        cur = j.get("status", "")
        if cur != last_status:
            _info(f"  {cur}  stage={j.get('stage')}  progress={j.get('progress')}")
            last_status = cur
        if cur == "succeeded":
            return j
        if cur in ("failed", "cancelled"):
            _fail(f"job {cur}: {j.get('error')}")
    _fail(f"job did not reach terminal status within {TIMEOUT}s")


def test_download_glb(job: dict) -> None:
    result_url = job.get("result_url")
    if not result_url:
        _fail("no result_url in job payload")
    _info(f"GET {result_url}")
    status, body = _req("GET", result_url)
    if status != 200:
        _fail(f"download returned {status}")
    if len(body) < 100:
        _fail(f"downloaded only {len(body)} bytes")
    _info(f"  downloaded {len(body)} bytes")

    # Validate with trimesh
    try:
        import trimesh

        mesh = trimesh.load(io.BytesIO(body), file_type="glb")
        verts = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if verts is None or faces is None:
            _fail("trimesh could not extract vertices/faces")
        _info(f"  trimesh OK — verts={len(verts)}  faces={len(faces)}")
    except ImportError:
        _info("  trimesh not installed; skipping mesh validation")
    except Exception as exc:
        _fail(f"trimesh validation failed: {exc}")


def test_cancel(job_id: str) -> None:
    _info(f"POST /v1/jobs/{job_id}/cancel  (best-effort)")
    status, body = _req("POST", f"/v1/jobs/{job_id}/cancel")
    _info(f"  status={status}  body={body!r:.120}")


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    _info(f"Service root : {SERVICE_ROOT}")
    _info(f"Base URL     : {BASE}")
    _info(f"Timeout      : {TIMEOUT}s")

    _ensure_server()
    test_health()
    job = test_submit_and_wait()
    test_download_glb(job)
    # Cancel test uses a fresh job to avoid race conditions
    _info("Submitting second job for cancel test …")
    payload = json.dumps({
        "image": _make_test_image_png_base64(),
        "enable_texture": False,
        "octree_resolution": 256,
    }).encode()
    s, j2 = _req("POST", "/v1/generate", data=payload,
                  headers={"Content-Type": "application/json"})
    if s == 202:
        test_cancel(j2.get("job_id", ""))

    print("\nPASS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
