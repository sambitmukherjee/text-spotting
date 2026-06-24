from __future__ import annotations

from collections import defaultdict

from invoice_grounding.models import BoundingBox, CandidateSpan, NormalizedBoundingBox, OCRWord
from invoice_grounding.normalization import compact_text


def union_pixel_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x_min=min(box.x_min for box in boxes),
        y_min=min(box.y_min for box in boxes),
        x_max=max(box.x_max for box in boxes),
        y_max=max(box.y_max for box in boxes),
    )


def union_normalized_boxes(boxes: list[NormalizedBoundingBox]) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(
        x_min=min(box.x_min for box in boxes),
        y_min=min(box.y_min for box in boxes),
        x_max=max(box.x_max for box in boxes),
        y_max=max(box.y_max for box in boxes),
    )


def generate_candidates(
    words: list[OCRWord],
    *,
    max_words_per_candidate: int = 20,
    max_lines_per_candidate: int = 5,
) -> list[CandidateSpan]:
    if not words:
        return []
    ordered = sorted(words, key=lambda word: word.reading_order)
    candidates: list[CandidateSpan] = []
    candidates.extend(_line_spans(ordered, max_words_per_candidate=max_words_per_candidate))
    candidates.extend(
        _multi_line_spans(
            ordered,
            max_words_per_candidate=max_words_per_candidate,
            max_lines_per_candidate=max_lines_per_candidate,
        )
    )
    seen: set[tuple[str, ...]] = set()
    deduped: list[CandidateSpan] = []
    for candidate in candidates:
        key = tuple(candidate.word_ids)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _line_spans(words: list[OCRWord], *, max_words_per_candidate: int) -> list[CandidateSpan]:
    by_line: dict[str, list[OCRWord]] = defaultdict(list)
    for word in words:
        by_line[word.line_id].append(word)
    candidates: list[CandidateSpan] = []
    for line_words in by_line.values():
        line_words = sorted(line_words, key=lambda word: word.reading_order)
        for start in range(len(line_words)):
            for end in range(start + 1, min(len(line_words), start + max_words_per_candidate) + 1):
                candidates.append(_make_candidate(line_words[start:end]))
    return candidates


def _multi_line_spans(
    words: list[OCRWord],
    *,
    max_words_per_candidate: int,
    max_lines_per_candidate: int,
) -> list[CandidateSpan]:
    by_line: dict[str, list[OCRWord]] = defaultdict(list)
    for word in words:
        by_line[word.line_id].append(word)
    lines = [sorted(line_words, key=lambda word: word.reading_order) for line_words in by_line.values()]
    lines.sort(key=lambda line: (min(word.bbox_pixels[1] for word in line), min(word.reading_order for word in line)))

    candidates: list[CandidateSpan] = []
    for start in range(len(lines)):
        collected: list[OCRWord] = []
        for end in range(start, min(len(lines), start + max_lines_per_candidate)):
            collected.extend(lines[end])
            if 1 < len(collected) <= max_words_per_candidate:
                candidates.append(_make_candidate(sorted(collected, key=lambda word: word.reading_order)))
            if len(collected) >= max_words_per_candidate:
                break
    return candidates


def _make_candidate(words: list[OCRWord]) -> CandidateSpan:
    text = " ".join(word.text for word in words)
    pixel_box = union_pixel_boxes([word.box_pixels for word in words])
    normalized_box = union_normalized_boxes([word.box_normalized for word in words])
    return CandidateSpan(
        id="|".join(word.id for word in words),
        text=text,
        compact_text=compact_text(text),
        words=words,
        word_ids=[word.id for word in words],
        line_ids=sorted({word.line_id for word in words}),
        bbox_pixels=pixel_box,
        bbox_normalized=normalized_box,
        mean_ocr_confidence=sum(word.confidence for word in words) / len(words),
    )
