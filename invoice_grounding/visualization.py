from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from invoice_grounding.doctr_ocr import load_image
from invoice_grounding.models import GroundingResult, GroundingStatus


def render_grounding_overlay(
    image: str | Path | Image.Image | np.ndarray,
    result: GroundingResult,
    output_path: str | Path,
    *,
    include_unmatched: bool = False,
    show_labels: bool = True,
    show_word_boxes: bool = False,
    font_size: int = 48,
) -> Path:
    pil = load_image(image, preprocess=False).copy()
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _load_font(font_size)

    colors = {
        GroundingStatus.MATCHED: (32, 140, 70, 220),
        GroundingStatus.INHERITED: (40, 105, 210, 210),
        GroundingStatus.AMBIGUOUS: (230, 150, 20, 220),
        GroundingStatus.UNMATCHED: (190, 55, 55, 180),
        GroundingStatus.NOT_GROUNDABLE: (120, 120, 120, 140),
        GroundingStatus.ERROR: (190, 0, 120, 190),
    }
    for field in result.fields:
        if field.status in {GroundingStatus.UNMATCHED, GroundingStatus.NOT_GROUNDABLE} and not include_unmatched:
            continue
        if field.union_box_pixels is None:
            continue
        color = colors[field.status]
        box = field.union_box_pixels
        draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), outline=color, width=3)
        draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), fill=color[:3] + (28,))
        if show_word_boxes:
            for word_box in field.word_boxes_pixels:
                draw.rectangle(
                    (word_box.x_min, word_box.y_min, word_box.x_max, word_box.y_max),
                    outline=color[:3] + (150,),
                    width=1,
                )
        if show_labels:
            label = _value_label(field.value_as_text, field.matched_text, field.value)
            _draw_label(draw, (box.x_min, max(0, box.y_min - font_size - 8)), label, color, font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)
    return path


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: tuple[int, int, int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    padding = max(6, int(getattr(font, "size", 24) * 0.18))
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle(
        (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
        fill=color[:3] + (220,),
    )
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)


def _load_font(font_size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _value_label(value_as_text: str | None, matched_text: str | None, value: object, *, max_length: int = 42) -> str:
    label = value_as_text or matched_text or str(value)
    label = " ".join(str(label).split())
    if len(label) > max_length:
        label = f"{label[: max_length - 3]}..."
    return label
