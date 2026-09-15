"""Thin adapters that let the pipeline treat third-party libraries like the
Hunyuan3D bindings (same duck-typed surface: ``to(device)`` + ``__call__``).

Kept out of :mod:`app.pipeline` so both the rembg library and hy3dgen
post-processors stay fully optional (imported lazily only when used).
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger("hy3d.pipeline")


class _RembgLibAdapter:
    """Wrap the standalone ``rembg`` pip library in the BackgroundRemover API.

    Used as a last-resort background remover when the vendored
    ``hy3dshape.rembg``/``hy3dgen.rembg`` modules are unavailable.
    """

    def __init__(self, remove_fn, onnx_session=None) -> None:
        self._remove = remove_fn
        self._sess = onnx_session
        self._device: Any = None

    @classmethod
    def _build(cls, rembg) -> "_RembgLibAdapter":
        # rembg>=2.0 exposes a preconfigured session via .new_session().
        session = None
        try:
            session = rembg.new_session()
        except Exception as exc:  # noqa: BLE001
            logger.debug("rembg.new_session() unavailable (%s); using defaults", exc)

        def _fn(image: Image.Image) -> Image.Image:
            png = image.convert("RGBA") if isinstance(image, Image.Image) else image
            return rembg.remove(png, session=session)

        return cls(_fn, session)

    def to(self, device: Any) -> "_RembgLibAdapter":
        self._device = device
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        out = self._remove(image)
        if isinstance(out, Image.Image):
            return out
        # rembg may return raw bytes / numpy arrays.
        from io import BytesIO

        return Image.open(BytesIO(out)).convert("RGBA")