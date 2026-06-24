from __future__ import annotations

import pytest

from invoice_grounding.models import OCRWord


@pytest.fixture
def make_word():
    def factory(
        text: str,
        order: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        line: int = 0,
        block: int = 0,
        confidence: float = 0.98,
        width: int = 1000,
        height: int = 1000,
    ) -> OCRWord:
        return OCRWord(
            id=f"p0-b{block}-l{line}-w{order}",
            text=text,
            confidence=confidence,
            page_index=0,
            block_index=block,
            line_index=line,
            word_index=order,
            reading_order=order,
            bbox_pixels=(x0, y0, x1, y1),
            bbox_normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
        )

    return factory
