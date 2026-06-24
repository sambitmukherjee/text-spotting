from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from invoice_grounding.models import GroundingConfig, InputLoadError, OCRWord, OCRError

LOGGER = logging.getLogger(__name__)


def load_image(image: str | Path | Image.Image | np.ndarray, *, preprocess: bool = False) -> Image.Image:
    try:
        if isinstance(image, (str, Path)):
            pil = Image.open(image)
        elif isinstance(image, Image.Image):
            pil = image
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image input type: {type(image).__name__}")
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        if preprocess:
            pil = ImageEnhance.Contrast(pil).enhance(1.15)
        return pil
    except Exception as exc:
        raise InputLoadError(f"Unable to load image: {exc}") from exc


def run_doctr_ocr(
    image: str | Path | Image.Image | np.ndarray,
    *,
    config: GroundingConfig | None = None,
) -> tuple[list[OCRWord], dict[str, Any], Image.Image]:
    config = config or GroundingConfig()
    pil = load_image(image, preprocess=config.preprocess)
    source = _doctr_source(image, pil)
    try:
        from doctr.io import DocumentFile
    except Exception as exc:  # pragma: no cover - depends on optional docTR install
        raise OCRError("python-doctr is not installed or could not be imported") from exc

    try:
        document = DocumentFile.from_images([str(source)])
        model = _get_ocr_predictor(config.device)
        result = model(document)
        exported = result.export()
    except Exception as exc:  # pragma: no cover - depends on optional docTR runtime
        raise OCRError(f"docTR OCR failed: {exc}") from exc
    finally:
        if isinstance(source, Path) and source.name.startswith("invoice-grounding-"):
            source.unlink(missing_ok=True)

    words = extract_words_from_doctr_export(exported, pil.width, pil.height)
    info = {
        "predictor": "ocr_predictor(pretrained=True)",
        "device": config.device,
        "page_count": len(exported.get("pages", [])) if isinstance(exported, dict) else 1,
    }
    LOGGER.debug("docTR produced %d words", len(words))
    return words, info, pil


@lru_cache(maxsize=2)
def _get_ocr_predictor(device: str) -> Any:
    try:
        from doctr.models import ocr_predictor
    except Exception as exc:  # pragma: no cover
        raise OCRError("Unable to import docTR OCR predictor") from exc
    try:
        model = ocr_predictor(pretrained=True)
        if device in {"cpu", "cuda"}:
            _move_model_to_device(model, device)
        elif device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    _move_model_to_device(model, "cuda")
            except Exception:
                LOGGER.debug("Could not inspect CUDA availability", exc_info=True)
        return model
    except Exception as exc:  # pragma: no cover
        raise OCRError(f"Unable to initialize docTR OCR predictor: {exc}") from exc


def _move_model_to_device(model: Any, device: str) -> None:
    for attr in ("det_predictor", "reco_predictor"):
        predictor = getattr(model, attr, None)
        inner = getattr(predictor, "model", None)
        if hasattr(inner, "to"):
            inner.to(device)


def _doctr_source(original: str | Path | Image.Image | np.ndarray, pil: Image.Image) -> Path | str:
    # Always OCR the EXIF-corrected RGB image so returned boxes and overlays share coordinates.
    handle = tempfile.NamedTemporaryFile(prefix="invoice-grounding-", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.close()
    pil.save(path)
    return path


def extract_words_from_doctr_export(exported: dict[str, Any], width: int, height: int) -> list[OCRWord]:
    pages = exported.get("pages", []) if isinstance(exported, dict) else []
    words: list[OCRWord] = []
    reading_order = 0
    for page_index, page in enumerate(pages):
        for block_index, block in enumerate(page.get("blocks", []) or []):
            for line_index, line in enumerate(block.get("lines", []) or []):
                for word_index, word in enumerate(line.get("words", []) or []):
                    geometry = word.get("geometry")
                    bbox_norm, polygon_norm = _parse_geometry(geometry)
                    bbox_pixels = _normalized_to_pixels(bbox_norm, width, height)
                    polygon_pixels = (
                        [_point_to_pixels(point, width, height) for point in polygon_norm]
                        if polygon_norm is not None
                        else None
                    )
                    words.append(
                        OCRWord(
                            id=f"p{page_index}-b{block_index}-l{line_index}-w{word_index}",
                            text=str(word.get("value", "")),
                            confidence=float(word.get("confidence") or 0.0),
                            page_index=page_index,
                            block_index=block_index,
                            line_index=line_index,
                            word_index=word_index,
                            reading_order=reading_order,
                            bbox_normalized=bbox_norm,
                            bbox_pixels=bbox_pixels,
                            polygon_normalized=polygon_norm,
                            polygon_pixels=polygon_pixels,
                        )
                    )
                    reading_order += 1
    return words


def _parse_geometry(geometry: Any) -> tuple[tuple[float, float, float, float], list[tuple[float, float]] | None]:
    if geometry is None:
        raise OCRError("Malformed docTR word geometry: missing geometry")
    points = _geometry_points(geometry)
    if len(points) < 2:
        raise OCRError(f"Malformed docTR word geometry: {geometry!r}")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    bbox = (_clamp01(min(x_values)), _clamp01(min(y_values)), _clamp01(max(x_values)), _clamp01(max(y_values)))
    polygon = points if len(points) > 2 else None
    return bbox, polygon


def _geometry_points(geometry: Any) -> list[tuple[float, float]]:
    if isinstance(geometry, tuple):
        geometry = list(geometry)
    points: list[tuple[float, float]] = []
    if isinstance(geometry, list):
        for item in geometry:
            if isinstance(item, tuple):
                item = list(item)
            if isinstance(item, list) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
    return points


def _normalized_to_pixels(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box
    return (
        int(round(x_min * width)),
        int(round(y_min * height)),
        int(round(x_max * width)),
        int(round(y_max * height)),
    )


def _point_to_pixels(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return int(round(point[0] * width)), int(round(point[1] * height))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
