"""Hunyuan3D 2.1 async API package.

Submodules can be imported independently (e.g. ``from app import config``)
without pulling in FastAPI/uvicorn.
"""
from __future__ import annotations


def __getattr__(name: str):
    if name in ("app", "server"):
        import importlib

        module = importlib.import_module(f"app.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "server"]