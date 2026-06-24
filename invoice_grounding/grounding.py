from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from invoice_grounding.candidate_generation import generate_candidates, union_normalized_boxes, union_pixel_boxes
from invoice_grounding.doctr_ocr import load_image, run_doctr_ocr
from invoice_grounding.matching import inherit_field, match_groundable_value
from invoice_grounding.models import (
    BoundingBox,
    CandidateSummary,
    GroundableValue,
    GroundedField,
    GroundingConfig,
    GroundingResult,
    GroundingStatus,
    InputLoadError,
    NormalizedBoundingBox,
    OCRWord,
)
from invoice_grounding.normalization import field_canonical, normalize_text
from invoice_grounding.schema_traversal import extract_groundable_values

LOGGER = logging.getLogger(__name__)


def ground_invoice_values(
    image: str | Path | Image.Image | Any,
    extraction: str | Path | dict[str, Any],
    *,
    config: GroundingConfig | None = None,
) -> GroundingResult:
    config = config or GroundingConfig()
    _configure_logging(config)
    start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    extraction_dict = load_extraction(extraction)
    timings["extraction_loading"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    ocr_words, model_info, pil = run_doctr_ocr(image, config=config)
    timings["ocr"] = _elapsed_ms(t0)

    result = ground_invoice_values_from_ocr(
        pil,
        extraction_dict,
        ocr_words,
        config=config,
        ocr_model_info=model_info,
    )
    timings.update(result.timings_ms)
    timings["total"] = _elapsed_ms(start)
    result.timings_ms = timings
    return result


def ground_invoice_values_from_ocr(
    image: str | Path | Image.Image | Any,
    extraction: dict[str, Any],
    ocr_words: list[OCRWord],
    *,
    config: GroundingConfig | None = None,
    ocr_model_info: dict[str, Any] | None = None,
) -> GroundingResult:
    config = config or GroundingConfig()
    warnings: list[str] = []
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    pil = load_image(image, preprocess=False)
    timings["image_loading"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    values = extract_groundable_values(extraction)
    timings["schema_traversal"] = _elapsed_ms(t0)

    t0 = time.perf_counter()
    candidates = generate_candidates(
        ocr_words,
        max_words_per_candidate=config.max_words_per_candidate,
        max_lines_per_candidate=config.max_lines_per_candidate,
    )
    timings["candidate_generation"] = _elapsed_ms(t0)

    if not ocr_words:
        warnings.append("No OCR words were detected")
    LOGGER.debug("Generated %d candidate spans for %d fields", len(candidates), len(values))

    t0 = time.perf_counter()
    fields = _match_values(values, candidates, ocr_words, config)
    fields = _resolve_unmatched_fields(fields, ocr_words)
    fields = _resolve_ambiguous_fields(fields, ocr_words)
    fields = _resolve_ambiguous_fields(fields, ocr_words)
    fields = _resolve_late_inherited_fields(fields)
    timings["field_matching"] = _elapsed_ms(t0)

    return GroundingResult(
        image_width=pil.width,
        image_height=pil.height,
        page_count=1,
        ocr_engine="docTR",
        ocr_model_info=ocr_model_info or config.ocr_model_info,
        fields=fields,
        ambiguous=_review_items(fields, GroundingStatus.AMBIGUOUS),
        unmatched=_review_items(fields, GroundingStatus.UNMATCHED),
        ocr_words=ocr_words if config.include_ocr_words else [],
        warnings=warnings,
        timings_ms=timings,
    )


def load_extraction(extraction: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(extraction, dict):
        if not extraction:
            raise InputLoadError("Extraction dictionary is empty")
        return extraction
    try:
        path = Path(extraction)
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputLoadError(f"Invalid extraction JSON: {exc}") from exc
    except Exception as exc:
        raise InputLoadError(f"Unable to load extraction JSON: {exc}") from exc
    if not isinstance(loaded, dict) or not loaded:
        raise InputLoadError("Extraction JSON must contain a non-empty object")
    return loaded


def _review_items(fields: list[GroundedField], status: GroundingStatus) -> list[dict[str, Any]]:
    return [_review_item(field) for field in fields if field.status == status]


def _review_item(field: GroundedField) -> dict[str, Any]:
    return {
        "json_path": field.json_path,
        "field_name": field.field_name,
        "value": field.value,
        "value_as_text": field.value_as_text,
        "reason": field.reason,
        "alternative_candidates": [candidate.model_dump() for candidate in field.alternative_candidates],
    }


def _match_values(
    values: list[GroundableValue],
    candidates: list,
    ocr_words: list[OCRWord],
    config: GroundingConfig,
) -> list[GroundedField]:
    by_path: dict[str, GroundedField] = {}
    deferred_inherited: list[GroundableValue] = []

    for value in values:
        if value.inherited_from:
            deferred_inherited.append(value)
            continue
        try:
            field = match_groundable_value(value, candidates, ocr_words, config)
        except Exception as exc:
            LOGGER.debug("Field-specific matching error for %s", value.json_path, exc_info=True)
            field = GroundedField(
                json_path=value.json_path,
                field_name=value.field_name,
                value=value.value,
                value_as_text=value.value_as_text,
                status=GroundingStatus.ERROR,
                reason=str(exc),
            )
        by_path[value.json_path] = field

    for value in deferred_inherited:
        source = by_path.get(value.inherited_from or "")
        if source and source.status in {GroundingStatus.MATCHED, GroundingStatus.INHERITED}:
            by_path[value.json_path] = inherit_field(value, source)
            continue
        # Fall back to a direct semantic date match if no raw printed source is present.
        field = match_groundable_value(value, candidates, ocr_words, config)
        if field.status == GroundingStatus.MATCHED:
            field.match_method = "semantic_date_normalization"
            field.reason = "No corresponding printed raw date field was available"
            field.inherited_from = value.inherited_from
        else:
            field.reason = "Corresponding printed raw date field was unavailable or unmatched"
            field.inherited_from = value.inherited_from
        by_path[value.json_path] = field

    return [by_path[value.json_path] for value in values if value.json_path in by_path]


def _resolve_unmatched_fields(fields: list[GroundedField], ocr_words: list[OCRWord]) -> list[GroundedField]:
    resolved: list[GroundedField] = []
    matched_anchors = [field for field in fields if field.status in {GroundingStatus.MATCHED, GroundingStatus.INHERITED}]
    party_blocks = _party_role_blocks(matched_anchors)
    for field in fields:
        if field.status != GroundingStatus.UNMATCHED:
            resolved.append(field)
            continue
        if _is_currency_field(field):
            resolved.append(_not_groundable_currency(field))
            continue
        seller_name = _seller_name_fallback(field, ocr_words)
        party_name = _party_name_fallback(field, ocr_words, matched_anchors, party_blocks)
        party_structured = _party_structured_fallback(field, ocr_words, matched_anchors, party_blocks)
        long_text = _invoice_info_long_text_fallback(field, ocr_words)
        resolved.append(seller_name or party_name or party_structured or long_text or field)
    return resolved


def _not_groundable_currency(field: GroundedField) -> GroundedField:
    return GroundedField(
        json_path=field.json_path,
        field_name=field.field_name,
        value=field.value,
        value_as_text=field.value_as_text,
        status=GroundingStatus.NOT_GROUNDABLE,
        normalized_target=field_canonical(field.value_as_text, field.json_path) or normalize_text(field.value_as_text),
        reason="Normalized currency code was not directly printed as an OCR code or configured symbol",
    )


def _seller_name_fallback(field: GroundedField, ocr_words: list[OCRWord]) -> GroundedField | None:
    if field.field_name != "name" or ".seller." not in field.json_path.casefold():
        return None
    target_terms = _name_terms(field.value_as_text or "")
    if len(target_terms) < 2:
        return None

    direct_words, direct_terms = _seller_direct_name_words(target_terms, ocr_words)
    email_words, email_terms = _seller_email_name_words(target_terms, ocr_words)
    covered_terms = direct_terms | email_terms
    coverage = len(covered_terms) / len(set(target_terms))
    direct_coverage = len(direct_terms) / len(set(target_terms))
    email_coverage = len(email_terms) / len(set(target_terms))
    if coverage < 0.60:
        return None
    if direct_coverage < 0.35 and email_coverage < 0.80:
        return None

    selected_words = _dedupe_words(direct_words + email_words)
    if not selected_words:
        return None
    confidence = min(0.90, 0.72 + (coverage * 0.14) + (direct_coverage * 0.04))
    return _field_from_words(
        field,
        selected_words,
        match_method="seller_name_header_or_email_partial",
        confidence=confidence,
        reason="Seller name recovered from top-of-page name text and seller email/domain evidence",
    )


def _seller_direct_name_words(target_terms: list[str], ocr_words: list[OCRWord]) -> tuple[list[OCRWord], set[str]]:
    selected: list[OCRWord] = []
    covered: set[str] = set()
    for word in sorted(ocr_words, key=lambda item: item.reading_order):
        y_center = (word.bbox_normalized[1] + word.bbox_normalized[3]) / 2
        if y_center > 0.28:
            continue
        word_terms = _name_terms(word.text)
        for target_term in target_terms:
            if target_term in covered:
                continue
            if any(_token_similarity(target_term, word_term) >= 0.84 for word_term in word_terms):
                selected.append(word)
                covered.add(target_term)
                break
    return selected, covered


def _seller_email_name_words(target_terms: list[str], ocr_words: list[OCRWord]) -> tuple[list[OCRWord], set[str]]:
    selected: list[OCRWord] = []
    covered: set[str] = set()
    for word in sorted(ocr_words, key=lambda item: item.reading_order):
        text = word.text.casefold()
        if "@" not in text and "www." not in text:
            continue
        word_terms = _name_terms(text)
        matched = {
            target_term
            for target_term in target_terms
            if any(_token_similarity(target_term, word_term) >= 0.88 for word_term in word_terms)
        }
        if len(matched) >= 2:
            selected.append(word)
            covered.update(matched)
    return selected, covered


def _party_name_fallback(
    field: GroundedField,
    ocr_words: list[OCRWord],
    anchors: list[GroundedField],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> GroundedField | None:
    role = _party_role(field.json_path)
    if field.field_name != "name" or role is None:
        return None
    target_terms = _name_terms(field.value_as_text or "")
    unique_terms = list(dict.fromkeys(target_terms))
    if len(unique_terms) < 2:
        return None

    best: tuple[float, list[OCRWord], float, float] | None = None
    for region_words, evidence in _party_name_regions(role, field.json_path, ocr_words, anchors, party_blocks):
        selected_words, covered_terms = _party_name_region_matches(unique_terms, region_words)
        if len(selected_words) < 2:
            continue
        coverage = len(covered_terms) / len(set(unique_terms))
        if coverage < 0.72:
            continue
        compactness = _party_name_compactness_score(selected_words)
        if compactness < 0.10:
            continue
        score = coverage + evidence + compactness
        if best is None or score > best[0]:
            best = (score, selected_words, coverage, evidence)

    if best is None:
        return None
    _, selected_words, coverage, evidence = best
    if evidence < 0.34:
        return None
    confidence = min(0.91, 0.70 + (coverage * 0.14) + (evidence * 0.06))
    return _field_from_words(
        field,
        selected_words,
        match_method="party_name_label_or_block_partial",
        confidence=confidence,
        reason="Party name recovered from nearby party label or sibling address block evidence",
    )


def _party_name_regions(
    role: str,
    json_path: str,
    ocr_words: list[OCRWord],
    anchors: list[GroundedField],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> list[tuple[list[OCRWord], float]]:
    regions: list[tuple[list[OCRWord], float]] = []
    lines = _visual_lines(ocr_words)
    for index, line_words in enumerate(lines):
        label_score = _party_label_evidence(role, line_words)
        if label_score <= 0.0:
            continue
        collected = list(line_words)
        label_box = union_normalized_boxes([word.box_normalized for word in line_words])
        for next_line in lines[index + 1 : index + 5]:
            next_box = union_normalized_boxes([word.box_normalized for word in next_line])
            if next_box.y_min - label_box.y_max > 0.16:
                break
            if _looks_like_table_or_invoice_header(next_line):
                break
            collected.extend(next_line)
        regions.append((_dedupe_words(collected), label_score))

    prefix = _party_prefix(json_path)
    if prefix is not None:
        sibling_boxes = [
            anchor.union_box_normalized
            for anchor in anchors
            if anchor.union_box_normalized is not None
            and anchor.json_path.startswith(prefix)
            and anchor.json_path != json_path
            and anchor.field_name != "phone"
        ]
        if sibling_boxes:
            sibling_box = union_normalized_boxes(sibling_boxes)
            regions.append((_words_near_party_box(ocr_words, sibling_box), 0.42))

    if role in party_blocks:
        regions.append((_words_near_party_box(ocr_words, party_blocks[role]), 0.34))

    return [(words, evidence) for words, evidence in regions if words]


def _party_label_evidence(role: str, line_words: list[OCRWord]) -> float:
    compact = _compact(" ".join(word.text for word in line_words))
    if role == "customer":
        if any(label in compact for label in ("invoiceto", "billto", "soldto")):
            return 0.70
        if "customer" in compact and not any(label in compact for label in ("customerno", "customerref")):
            return 0.52
    if role == "shipto":
        if any(label in compact for label in ("shipto", "deliverto", "deliveryto", "despatchto", "dispatchto")):
            return 0.72
    if role == "seller":
        if any(label in compact for label in ("soldby", "seller", "supplier", "vendor")):
            return 0.58
    return 0.0


def _party_name_region_matches(
    target_terms: list[str],
    region_words: list[OCRWord],
) -> tuple[list[OCRWord], set[str]]:
    selected: list[OCRWord] = []
    covered: set[str] = set()
    for target_term in target_terms:
        best_word: OCRWord | None = None
        best_score = 0.0
        for word in region_words:
            if word.id in {item.id for item in selected}:
                continue
            for word_term in _name_terms(word.text):
                score = _token_similarity(target_term, word_term)
                if score > best_score:
                    best_score = score
                    best_word = word
        threshold = 0.82 if len(target_term) <= 5 else 0.78
        if best_word is not None and best_score >= threshold:
            selected.append(best_word)
            covered.add(target_term)
    return _dedupe_words(selected), covered


def _party_name_compactness_score(words: list[OCRWord]) -> float:
    if not words:
        return 0.0
    box = _word_union_normalized(words)
    height = box.y_max - box.y_min
    width = box.x_max - box.x_min
    if height > 0.24 or width > 0.78:
        return 0.0
    line_count = len({word.line_id for word in words})
    if line_count <= 2:
        return 0.20
    if line_count <= 4 and height <= 0.16:
        return 0.12
    return 0.04


def _words_near_party_box(ocr_words: list[OCRWord], box: NormalizedBoundingBox) -> list[OCRWord]:
    expanded = NormalizedBoundingBox(
        x_min=max(0.0, box.x_min - 0.20),
        y_min=max(0.0, box.y_min - 0.18),
        x_max=min(1.0, box.x_max + 0.08),
        y_max=min(1.0, box.y_max + 0.04),
    )
    return [
        word
        for word in sorted(ocr_words, key=lambda item: item.reading_order)
        if _box_center_inside(word.box_normalized, expanded)
    ]


def _words_in_expanded_box(
    ocr_words: list[OCRWord],
    box: NormalizedBoundingBox,
    *,
    x_pad: float,
    y_above: float,
    y_below: float,
) -> list[OCRWord]:
    expanded = NormalizedBoundingBox(
        x_min=max(0.0, box.x_min - x_pad),
        y_min=max(0.0, box.y_min - y_above),
        x_max=min(1.0, box.x_max + x_pad),
        y_max=min(1.0, box.y_max + y_below),
    )
    return [
        word
        for word in sorted(ocr_words, key=lambda item: item.reading_order)
        if _box_center_inside(word.box_normalized, expanded)
    ]


def _visual_lines(ocr_words: list[OCRWord]) -> list[list[OCRWord]]:
    by_line: dict[str, list[OCRWord]] = {}
    for word in ocr_words:
        by_line.setdefault(word.line_id, []).append(word)
    return sorted(
        (sorted(words, key=lambda item: item.reading_order) for words in by_line.values()),
        key=lambda words: (min(word.box_normalized.y_min for word in words), min(word.box_normalized.x_min for word in words)),
    )


def _looks_like_table_or_invoice_header(words: list[OCRWord]) -> bool:
    compact = _compact(" ".join(word.text for word in words))
    markers = (
        "productcode",
        "description",
        "quantity",
        "unitprice",
        "subtotal",
        "invoiceno",
        "invoicenumber",
        "taxdate",
        "pageno",
    )
    return any(marker in compact for marker in markers)


def _invoice_info_long_text_fallback(field: GroundedField, ocr_words: list[OCRWord]) -> GroundedField | None:
    if ".invoiceinfo." not in field.json_path.casefold() or field.field_name not in {"customerMemo", "paymentTerms"}:
        return None
    target_terms = _long_text_terms(field.value_as_text or "")
    if len(target_terms) < 6:
        return None

    selected: list[OCRWord] = []
    covered: set[int] = set()
    similarities: list[float] = []
    for target_index, target_term in enumerate(target_terms):
        best_word: OCRWord | None = None
        best_score = 0.0
        for word in ocr_words:
            if word.id in {item.id for item in selected}:
                continue
            context_bonus = _long_text_word_context_score(word, ocr_words)
            for word_term in _long_text_terms(word.text):
                score = _token_similarity(target_term, word_term)
                if score > best_score or (
                    best_word is not None
                    and score == best_score
                    and context_bonus > _long_text_word_context_score(best_word, ocr_words)
                ):
                    best_score = score
                    best_word = word
        threshold = 0.86 if any(char.isdigit() for char in target_term) else (0.82 if len(target_term) <= 4 else 0.74)
        if best_word is not None and best_score >= threshold:
            selected.append(best_word)
            covered.add(target_index)
            similarities.append(best_score)

    coverage = len(covered) / len(target_terms)
    if coverage < 0.58 or len(selected) < 6:
        return None
    if not _long_text_has_label_or_dense_phrase(field.field_name, selected, ocr_words):
        return None
    mean_similarity = sum(similarities) / len(similarities)
    confidence = min(0.88, 0.62 + (coverage * 0.20) + (mean_similarity * 0.05))
    return _field_from_words(
        field,
        _dedupe_words(selected),
        match_method="invoice_info_long_text_partial",
        confidence=confidence,
        reason="Long invoiceInfo text recovered from reliable OCR substrings across nearby lines",
    )


def _party_structured_fallback(
    field: GroundedField,
    ocr_words: list[OCRWord],
    anchors: list[GroundedField],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> GroundedField | None:
    role = _party_role(field.json_path)
    if role is None:
        return None
    role_words = _role_context_words(field.json_path, role, ocr_words, anchors, party_blocks)
    if not role_words:
        return None
    if field.field_name in {"postal_code", "postalCode"}:
        return _postal_code_fallback(field, role_words)
    if field.field_name == "country":
        return _country_fallback(field, role_words)
    if field.field_name == "phone":
        return _phone_fallback(field, role_words)
    if field.field_name == "email":
        return _email_fallback(field, role_words, ocr_words)
    return None


def _role_context_words(
    json_path: str,
    role: str,
    ocr_words: list[OCRWord],
    anchors: list[GroundedField],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> list[OCRWord]:
    if role in party_blocks:
        return _words_in_expanded_box(ocr_words, party_blocks[role], x_pad=0.06, y_above=0.04, y_below=0.08)
    prefix = _party_prefix(json_path)
    if prefix is None:
        return []
    boxes = [
        anchor.union_box_normalized
        for anchor in anchors
        if anchor.union_box_normalized is not None
        and anchor.json_path.startswith(prefix)
        and anchor.json_path != json_path
    ]
    if not boxes:
        return []
    return _words_in_expanded_box(ocr_words, union_normalized_boxes(boxes), x_pad=0.06, y_above=0.04, y_below=0.08)


def _postal_code_fallback(field: GroundedField, role_words: list[OCRWord]) -> GroundedField | None:
    target = _postal_compact(field.value_as_text or "")
    if len(target) < 5:
        return None
    ordered = sorted(role_words, key=lambda item: item.reading_order)
    for start in range(len(ordered)):
        for end in range(start + 1, min(len(ordered), start + 3) + 1):
            words = ordered[start:end]
            text = " ".join(word.text for word in words)
            if _postal_compact(text) == target:
                return _field_from_words(
                    field,
                    words,
                    match_method="party_postal_code_ocr_repair",
                    confidence=0.86,
                    reason="Postal code recovered from party block with OCR character repair",
                )
    return None


def _country_fallback(field: GroundedField, role_words: list[OCRWord]) -> GroundedField | None:
    target = _country_alias(field.value_as_text or "")
    if target is None:
        return None
    ordered = sorted(role_words, key=lambda item: item.reading_order)
    for start in range(len(ordered)):
        for end in range(start + 1, min(len(ordered), start + 2) + 1):
            words = ordered[start:end]
            if _country_alias(" ".join(word.text for word in words)) == target:
                return _field_from_words(
                    field,
                    words,
                    match_method="party_country_alias",
                    confidence=0.86,
                    reason="Country recovered from equivalent printed country code/name in party block",
                )
    return None


def _phone_fallback(field: GroundedField, role_words: list[OCRWord]) -> GroundedField | None:
    target_digits = re.sub(r"\D+", "", field.value_as_text or "")
    if len(target_digits) < 9:
        return None
    for word in sorted(role_words, key=lambda item: item.reading_order):
        digits = re.sub(r"\D+", "", word.text)
        if len(digits) >= 7 and (target_digits.endswith(digits) or digits.endswith(target_digits)):
            return _field_from_words(
                field,
                [word],
                match_method="party_phone_partial_suffix",
                confidence=0.80,
                reason="Phone number recovered from long matching suffix inside the party block",
            )
    return None


def _email_fallback(field: GroundedField, role_words: list[OCRWord], ocr_words: list[OCRWord]) -> GroundedField | None:
    target = _email_ocr_compact(field.value_as_text or "")
    if not target or "@" not in target:
        return None
    best_word: OCRWord | None = None
    best_score = 0.0
    for word in role_words:
        candidate = _email_ocr_compact(word.text)
        if "@" not in candidate:
            continue
        score = _token_similarity(target, candidate)
        if _line_has_label(word, ocr_words, "email"):
            score += 0.08
        if score > best_score:
            best_score = score
            best_word = word
    if best_word is None or best_score < 0.68:
        return None
    return _field_from_words(
        field,
        [best_word],
        match_method="party_email_ocr_tolerant",
        confidence=min(0.84, best_score),
        reason="Email recovered from OCR-noisy address in party block",
    )


def _postal_compact(text: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", text.casefold()).upper()
    compact = compact.replace("O", "0")
    compact = compact.replace("I", "1")
    return compact


def _country_alias(text: str) -> str | None:
    compact = _compact(text)
    aliases = {
        "gb": "GB",
        "gbr": "GB",
        "uk": "GB",
        "unitedkingdom": "GB",
        "greatbritain": "GB",
        "england": "GB",
    }
    return aliases.get(compact)


def _email_ocr_compact(text: str) -> str:
    normalized = normalize_text(text, keep_punctuation=True)
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("£", "e")
    normalized = normalized.replace("|", "l")
    return normalized


def _line_has_label(word: OCRWord, ocr_words: list[OCRWord], label: str) -> bool:
    compact = _compact(" ".join(item.text for item in ocr_words if item.line_id == word.line_id))
    return label in compact


def _long_text_terms(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "we",
        "where",
        "with",
    }
    terms = re.findall(r"[a-z0-9]+", normalize_text(text, keep_punctuation=False))
    important = [
        term
        for term in terms
        if term not in stopwords and (len(term) > 2 or (term.isdigit() and len(term) >= 3))
    ]
    return important or terms


def _long_text_word_context_score(word: OCRWord, ocr_words: list[OCRWord]) -> float:
    line_words = [item for item in ocr_words if item.line_id == word.line_id]
    line_compact = _compact(" ".join(item.text for item in line_words))
    score = 0.0
    if any(label in line_compact for label in ("courier", "memo", "note", "terms")):
        score += 0.20
    y_center = (word.box_normalized.y_min + word.box_normalized.y_max) / 2
    if y_center > 0.45:
        score += 0.05
    return score


def _long_text_has_label_or_dense_phrase(field_name: str, selected_words: list[OCRWord], ocr_words: list[OCRWord]) -> bool:
    selected_ids = {word.id for word in selected_words}
    if field_name == "customerMemo" and any(_compact(word.text).startswith(("courier", "memo", "note")) for word in selected_words):
        return True
    if field_name == "paymentTerms" and any(_compact(word.text).startswith(("terms", "payment")) for word in selected_words):
        return True
    by_line = {word.line_id: [] for word in selected_words}
    for word in ocr_words:
        if word.line_id in by_line:
            by_line[word.line_id].append(word)
    for line_words in by_line.values():
        selected_on_line = [word for word in line_words if word.id in selected_ids]
        if len(selected_on_line) >= 4:
            return True
    return False


def _name_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", normalize_text(text, keep_punctuation=False))
    stopwords = {"and", "the", "ltd", "limited", "plc", "inc", "co", "uk", "com", "www", "sales", "email"}
    return [term for term in terms if len(term) > 1 and term not in stopwords]


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    # Local fallback is enough here; the production matcher still owns broader fuzzy scoring.
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def _dedupe_words(words: list[OCRWord]) -> list[OCRWord]:
    seen: set[str] = set()
    deduped: list[OCRWord] = []
    for word in sorted(words, key=lambda item: item.reading_order):
        if word.id in seen:
            continue
        seen.add(word.id)
        deduped.append(word)
    return deduped


def _resolve_late_inherited_fields(fields: list[GroundedField]) -> list[GroundedField]:
    by_path = {field.json_path: field for field in fields}
    resolved: list[GroundedField] = []
    for field in fields:
        source = by_path.get(field.inherited_from or "")
        if (
            source is not None
            and field.status != GroundingStatus.INHERITED
            and source.status in {GroundingStatus.MATCHED, GroundingStatus.INHERITED}
        ):
            resolved.append(_inherit_grounded_field(field, source))
        else:
            resolved.append(field)
    return resolved


def _inherit_grounded_field(field: GroundedField, source: GroundedField) -> GroundedField:
    return GroundedField(
        json_path=field.json_path,
        field_name=field.field_name,
        value=field.value,
        value_as_text=field.value_as_text,
        status=GroundingStatus.INHERITED,
        match_method="inherited_after_context_resolution",
        confidence=source.confidence,
        score_breakdown=source.score_breakdown,
        matched_text=source.matched_text,
        normalized_target=field_canonical(field.value_as_text, field.json_path) or normalize_text(field.value_as_text),
        normalized_matched_text=source.normalized_matched_text,
        word_ids=source.word_ids,
        word_boxes_pixels=source.word_boxes_pixels,
        word_boxes_normalized=source.word_boxes_normalized,
        line_boxes_pixels=source.line_boxes_pixels,
        union_box_pixels=source.union_box_pixels,
        union_box_normalized=source.union_box_normalized,
        inherited_from=source.json_path,
        reason="ISO-normalized field inherits evidence after the printed date was resolved by context",
    )


def _resolve_ambiguous_fields(fields: list[GroundedField], ocr_words: list[OCRWord]) -> list[GroundedField]:
    words_by_id = {word.id: word for word in ocr_words}
    resolved: list[GroundedField] = []
    matched_anchors = [field for field in fields if field.status in {GroundingStatus.MATCHED, GroundingStatus.INHERITED}]
    party_blocks = _party_role_blocks(matched_anchors)

    for field in fields:
        if field.status != GroundingStatus.AMBIGUOUS or not field.alternative_candidates:
            resolved.append(field)
            continue
        scored = _score_ambiguous_alternatives(
            field,
            field.alternative_candidates,
            words_by_id,
            matched_anchors,
            party_blocks,
        )
        if len(scored) < 2:
            resolved.append(field)
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = _dedupe_overlapping_scored_alternatives(scored)
        if len(scored) < 2:
            best_score, best_alt = scored[0]
            if best_score >= 0.26:
                resolved.append(_field_from_alternative(field, best_alt, words_by_id, best_score))
            else:
                resolved.append(field)
            continue
        best_score, best_alt = scored[0]
        second_score = scored[1][0]
        if _can_promote_ambiguous(field, best_score, second_score):
            resolved.append(_field_from_alternative(field, best_alt, words_by_id, best_score))
        else:
            resolved.append(field)
    return resolved


def _dedupe_overlapping_scored_alternatives(
    scored: list[tuple[float, CandidateSummary]],
) -> list[tuple[float, CandidateSummary]]:
    deduped: list[tuple[float, CandidateSummary]] = []
    for score, alternative in scored:
        alt_ids = set(alternative.word_ids)
        if any(alt_ids & set(existing.word_ids) for _, existing in deduped):
            continue
        deduped.append((score, alternative))
    return deduped


def _can_promote_ambiguous(field: GroundedField, best_score: float, second_score: float) -> bool:
    margin = best_score - second_score
    if _party_role(field.json_path) is not None:
        if "addressstructured" in field.json_path.casefold():
            return best_score >= 0.72 and margin >= 0.08
        return best_score >= 0.26 and margin >= 0.18
    if ".invoiceinfo." in field.json_path.casefold():
        return best_score >= 0.42 and margin >= 0.16
    if ".totals." in field.json_path:
        return best_score >= 0.42 and margin >= 0.16
    return best_score >= 0.58 and margin >= 0.10


def _score_ambiguous_alternatives(
    field: GroundedField,
    alternatives: list[CandidateSummary],
    words_by_id: dict[str, OCRWord],
    anchors: list[GroundedField],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> list[tuple[float, CandidateSummary]]:
    scored: list[tuple[float, CandidateSummary]] = []
    for alt in alternatives:
        words = [words_by_id[word_id] for word_id in alt.word_ids if word_id in words_by_id]
        if not words:
            continue
        score = 0.0
        score += _role_label_score(field, words, words_by_id)
        score += _role_region_score(field, words)
        score += _sibling_anchor_score(field, words, anchors)
        score += _party_sibling_overlap_score(field, words, anchors)
        score += _party_address_order_score(field, words, anchors)
        score += _party_candidate_shape_score(field, words)
        score += _party_block_score(field, words, party_blocks)
        score += _invoice_info_alternative_score(field, words, words_by_id)
        score += _totals_alternative_score(field, words, words_by_id)
        scored.append((score, alt))
    return scored


def _role_label_score(field: GroundedField, words: list[OCRWord], words_by_id: dict[str, OCRWord]) -> float:
    role = _party_role(field.json_path)
    if role is None:
        return 0.0
    context = _role_local_context(words, list(words_by_id.values()))
    hints = {
        "seller": ["seller", "vendor", "supplier", "from", "tel", "email"],
        "customer": ["customer", "invoice to", "bill to", "sold to", "account"],
        "shipto": ["ship to", "deliver to", "delivery to", "consignee"],
    }[role]
    compact_context = _compact(context)
    if any(_compact(hint) in compact_context for hint in hints):
        return 0.55
    return 0.0


def _role_region_score(field: GroundedField, words: list[OCRWord]) -> float:
    role = _party_role(field.json_path)
    if role is None:
        return 0.0
    box = _word_union_normalized(words)
    x_center = (box.x_min + box.x_max) / 2
    y_center = (box.y_min + box.y_max) / 2
    if role == "seller":
        score = 0.28 if y_center < 0.35 else -0.18
        if x_center < 0.55:
            score += 0.14
        elif x_center > 0.70:
            score -= 0.08
        return score
    if role == "customer":
        return 0.28 if x_center < 0.45 else -0.14
    if role == "shipto":
        return 0.28 if x_center > 0.30 else -0.14
    return 0.0


def _sibling_anchor_score(field: GroundedField, words: list[OCRWord], anchors: list[GroundedField]) -> float:
    role_prefix = _party_prefix(field.json_path)
    if role_prefix is None:
        return 0.0
    sibling_boxes = [
        anchor.union_box_normalized
        for anchor in anchors
        if anchor.union_box_normalized is not None
        and anchor.json_path.startswith(role_prefix)
        and anchor.json_path != field.json_path
    ]
    if not sibling_boxes:
        return 0.0
    box = _word_union_normalized(words)
    distance = min(_box_distance(box, sibling_box) for sibling_box in sibling_boxes)
    if distance <= 0.08:
        return 0.36
    if distance <= 0.16:
        return 0.22
    if distance <= 0.28:
        return 0.08
    return -0.12


def _party_sibling_overlap_score(field: GroundedField, words: list[OCRWord], anchors: list[GroundedField]) -> float:
    role_prefix = _party_prefix(field.json_path)
    if role_prefix is None:
        return 0.0
    candidate_ids = {word.id for word in words}
    if not candidate_ids:
        return 0.0
    score = 0.0
    component_fields = {"city", "state", "postal_code", "postalCode", "country"}
    for anchor in anchors:
        if not anchor.json_path.startswith(role_prefix) or anchor.json_path == field.json_path:
            continue
        if not candidate_ids & set(anchor.word_ids):
            continue
        if field.field_name in component_fields and anchor.field_name in {"name", "address"}:
            score -= 0.66
        elif anchor.field_name != field.field_name:
            score -= 0.28
    return score


def _party_candidate_shape_score(field: GroundedField, words: list[OCRWord]) -> float:
    if _party_role(field.json_path) is None:
        return 0.0
    line_ids = {word.line_id for word in words}
    if len(line_ids) <= 1:
        return 0.0
    if field.field_name in {"phone", "email", "city", "state", "postal_code", "postalCode", "country"}:
        return -0.48
    return -0.16


def _party_address_order_score(field: GroundedField, words: list[OCRWord], anchors: list[GroundedField]) -> float:
    if _party_role(field.json_path) is None or "addressstructured" not in field.json_path.casefold():
        return 0.0
    if field.field_name not in {"city", "state"}:
        return 0.0
    role_prefix = _party_prefix(field.json_path)
    if role_prefix is None:
        return 0.0
    box = _word_union_normalized(words)
    y_center = (box.y_min + box.y_max) / 2
    sibling_boxes = {
        anchor.field_name: anchor.union_box_normalized
        for anchor in anchors
        if anchor.union_box_normalized is not None
        and anchor.json_path.startswith(role_prefix)
        and anchor.json_path != field.json_path
    }
    postal_box = sibling_boxes.get("postal_code") or sibling_boxes.get("postalCode")
    if field.field_name == "state" and postal_box is not None:
        postal_y = (postal_box.y_min + postal_box.y_max) / 2
        gap = postal_y - y_center
        if 0.0 <= gap <= 0.08:
            return max(0.0, 0.34 - (gap * 3.0))
        if gap < 0:
            return -0.18
    if field.field_name == "city":
        state_box = sibling_boxes.get("state")
        lower_box = state_box or postal_box
        if lower_box is not None:
            lower_y = (lower_box.y_min + lower_box.y_max) / 2
            gap = lower_y - y_center
            if 0.0 <= gap <= 0.10:
                return max(0.0, 0.24 - (gap * 1.8))
            if gap < 0:
                return -0.14
    return 0.0


def _party_role_blocks(anchors: list[GroundedField]) -> dict[str, NormalizedBoundingBox]:
    blocks: dict[str, NormalizedBoundingBox] = {}
    for role in ("seller", "customer", "shipto"):
        boxes = [
            anchor.union_box_normalized
            for anchor in anchors
            if anchor.union_box_normalized is not None
            and _party_role(anchor.json_path) == role
            and _is_stable_party_block_anchor(anchor)
        ]
        if not boxes:
            boxes = [
                anchor.union_box_normalized
                for anchor in anchors
                if anchor.union_box_normalized is not None
                and _party_role(anchor.json_path) == role
                and anchor.field_name != "phone"
            ]
        if boxes:
            blocks[role] = _expand_normalized_box(union_normalized_boxes(boxes), x_pad=0.035, y_pad=0.045)
    return blocks


def _is_stable_party_block_anchor(field: GroundedField) -> bool:
    if field.status not in {GroundingStatus.MATCHED, GroundingStatus.INHERITED}:
        return False
    if field.field_name in {"name", "phone"}:
        return False
    if field.union_box_normalized is None:
        return False
    box = field.union_box_normalized
    if box.x_max - box.x_min > 0.55:
        return False
    return True


def _party_block_score(
    field: GroundedField,
    words: list[OCRWord],
    party_blocks: dict[str, NormalizedBoundingBox],
) -> float:
    role = _party_role(field.json_path)
    if role is None or role not in party_blocks:
        return 0.0
    box = _word_union_normalized(words)
    own_block = party_blocks[role]
    score = 0.0
    if _box_center_inside(box, own_block):
        score += 0.58
    else:
        distance = _box_distance(box, own_block)
        if distance <= 0.08:
            score += 0.36
        elif distance <= 0.16:
            score += 0.16
        else:
            score -= 0.16

    other_distances = [
        _box_distance(box, other_block)
        for other_role, other_block in party_blocks.items()
        if other_role != role
    ]
    if other_distances:
        own_distance = _box_distance(box, own_block)
        nearest_other = min(other_distances)
        if nearest_other + 0.035 < own_distance:
            score -= 0.42
        elif nearest_other < 0.04 and not _box_center_inside(box, own_block):
            score -= 0.24
    return score


def _totals_alternative_score(
    field: GroundedField,
    words: list[OCRWord],
    words_by_id: dict[str, OCRWord],
) -> float:
    if ".totals." not in field.json_path:
        return 0.0
    all_words = list(words_by_id.values())
    context = _compact(_preceding_window_text(words, all_words))
    row_context = _compact(_row_label_text(words, all_words))
    summary_context = _compact(_visual_row_text(words, all_words))
    field_name = field.field_name
    if ".othercharges" in field.json_path.casefold() and field_name == "value":
        field_name = "otherChargesValue"
    if ".othercharges" in field.json_path.casefold() and field_name == "key":
        field_name = "otherChargesKey"
    summary_score = _totals_summary_evidence_score(field_name, words, row_context, summary_context)
    row_pair_score = _totals_row_pair_score(field_name, words, all_words)
    extra_text_penalty = _numeric_total_extra_text_penalty(field_name, field, words)
    if field_name == "totalIncludingTax":
        score = summary_score + row_pair_score + extra_text_penalty
        combined = row_context + context
        if any(label in combined for label in ("documenttotal", "grandtotal", "invoicetotal", "totalprice")):
            score += 0.62
        elif "totalgbp" in combined or "total" in combined:
            score += 0.36
        if any(label in combined for label in ("lessamountpaid", "amountpaid", "amountdue", "balancedue", "nettotal", "paymentdetails", "trustpayments")):
            score -= 0.44
        box = _word_union_normalized(words)
        if (box.y_min + box.y_max) / 2 >= 0.5:
            score += 0.12
        return score
    positive = {
        "totalIncludingTax": ["invoice total", "grand total", "document total", "total"],
        "deposit": ["less amount paid", "amount paid", "paid"],
        "balanceDue": ["amount due", "balance due", "total due"],
        "taxPercentage": ["vat rate", "tax rate", "vat", "tax"],
        "taxAmount": ["vat", "tax", "tax amount", "total tax"],
        "taxName": ["vat", "sales tax", "gst", "tax"],
        "discountTotal": ["discount", "less discount"],
        "discountPercentage": ["discount", "discount %", "disc"],
        "otherChargesValue": ["surcharge", "s/charge", "other", "charge", "pennies charity donation", "donation"],
        "subtotal": ["subtotal", "sub total", "net amount", "goods"],
        "totalExcludingTax": ["total net amount", "net amount", "goods"],
    }.get(field_name, [])
    negative = {
        "totalIncludingTax": ["less amount paid", "amount paid", "amount due"],
        "deposit": ["invoice total", "grand total", "amount due"],
        "taxPercentage": ["zero rated", "unit", "qty", "product description"],
        "taxAmount": ["subtotal", "total paid", "amount paid", "unit", "qty"],
        "taxName": ["registered", "regl no", "email", "web"],
        "discountTotal": ["vat", "tax", "subtotal", "total paid", "unit", "qty"],
        "discountPercentage": ["vat", "tax", "subtotal", "total paid", "unit", "qty"],
        "otherChargesValue": ["customer credit", "paypal", "subtotal", "vat", "tax"],
        "subtotal": ["invoice total", "grand total", "amount paid", "amount due", "donation"],
        "totalExcludingTax": ["invoice total", "grand total", "amount paid", "amount due", "donation"],
    }.get(field_name, [])
    score = summary_score + row_pair_score + extra_text_penalty
    positive_compacts = [_compact(label) for label in positive]
    negative_compacts = [_compact(label) for label in negative]
    if any(label in row_context for label in positive_compacts):
        score += 0.62
    elif any(label in context for label in positive_compacts):
        score += 0.38
    if any(label in row_context for label in negative_compacts):
        score -= 0.54
    elif any(label in context for label in negative_compacts):
        score -= 0.36
    box = _word_union_normalized(words)
    y_center = (box.y_min + box.y_max) / 2
    if y_center >= 0.5:
        score += 0.12
    elif y_center < 0.28:
        score -= 0.10
    return score


def _totals_row_pair_score(field_name: str, words: list[OCRWord], all_words: list[OCRWord]) -> float:
    candidate_text = " ".join(word.text for word in words)
    candidate_compact = _compact(candidate_text)
    row_left_raw = _row_label_text(words, all_words)
    row_left = _compact(row_left_raw)
    row_all = _compact(_visual_row_text(words, all_words))
    preceding = _compact(_preceding_window_text(words, all_words))
    context = row_left + row_all

    if field_name == "otherChargesKey":
        if candidate_compact.rstrip(":") in {"delivery", "shipping", "postage", "carriage", "courier"}:
            if candidate_text.strip().endswith(":"):
                return 0.62
            if any(label in row_left for label in ("delivery", "shipping", "postage", "carriage", "courier")):
                return 0.38
        return 0.0

    if not _looks_like_money_or_number(candidate_text):
        return 0.0
    if _looks_like_table_body_context(context) and not _has_strong_summary_label(row_left):
        return -0.18

    if field_name == "totalExcludingTax":
        if any(label in row_left for label in ("totalexvat", "totalexclvat", "totalexcludingvat", "nettotal", "subtotal", "productstotalexvat")):
            return 0.70
        if "totalexvat" in preceding or "productstotalexvat" in preceding:
            return 0.42
    if field_name == "subtotal":
        if any(label in row_left for label in ("subtotal", "itemsubtotal", "productstotalexvat", "subtotaltotal")):
            return 0.66
        if "subtotal" in preceding and not any(label in context for label in ("totalpaid", "balancedue", "amountdue")):
            return 0.34
    if field_name == "totalIncludingTax":
        if any(label in row_left for label in ("totalgbp", "totalprice", "grandtotal", "documenttotal", "invoicetotal", "totalatcheckout", "totalinclvat", "totalincvat")):
            return 0.70
        if row_left in {"total", "totaldue"}:
            return 0.54
    if field_name == "taxAmount":
        if any(label in row_left for label in ("vat", "tax", "vatsubtotal", "totalvat", "vatamount")):
            return 0.58
        if any(label in preceding for label in ("vatamount", "vatsubtotal", "vat20", "taxamount")):
            return 0.36
    if field_name == "otherChargesValue":
        if any(label in row_left for label in ("shippingcharges", "delivery", "postage", "carriage", "courier", "other")):
            return 0.78 if not _has_currency_amount(row_left_raw) else 0.30
    if field_name in {"discountTotal", "discountPercentage"}:
        if "discount" in row_left and not _looks_like_table_body_context(context):
            return 0.40
    return 0.0


def _numeric_total_extra_text_penalty(field_name: str, field: GroundedField, words: list[OCRWord]) -> float:
    if field_name in {"taxName", "taxPercentage", "discountPercentage", "otherChargesKey"}:
        return 0.0
    if len(words) <= 1 or not _looks_like_money_or_number(field.value_as_text or ""):
        return 0.0
    for word in words:
        token = word.text.casefold().strip("():")
        if re.search(r"[a-z]", token) and token not in {"gbp", "usd", "eur"}:
            return -0.24
    return 0.0


def _totals_summary_evidence_score(
    field_name: str,
    words: list[OCRWord],
    row_context: str,
    summary_context: str,
) -> float:
    if _looks_like_table_body_context(summary_context):
        return 0.0
    candidate_text = " ".join(word.text for word in words)
    candidate_compact = _compact(candidate_text)
    context = row_context + summary_context
    if field_name == "totalIncludingTax":
        if any(label in row_context for label in ("totalgbp", "totalprice", "grandtotal", "documenttotal", "invoicetotal")):
            if not any(label in context for label in ("lessamountpaid", "amountpaid", "balancedue", "amountdue")):
                return 0.46
    if field_name == "taxName":
        tax_labels = {"vat", "gst", "tax", "salestax"}
        if candidate_compact in tax_labels and _row_has_money_or_amount(summary_context):
            if not any(label in context for label in ("registered", "reglno", "email", "web", "telephone", "tel")):
                return 0.38
    if field_name == "otherChargesValue":
        if any(label in row_context for label in ("other", "surcharge", "scharge", "donation")):
            return 0.48
    if field_name in {"discountTotal", "discountPercentage"}:
        if "discount" in row_context and not _looks_like_table_body_context(context):
            return 0.36
    return 0.0


def _looks_like_table_body_context(compact_text_value: str) -> bool:
    table_markers = (
        "productdescription",
        "productno",
        "descriptionqty",
        "descriptionquantity",
        "qtyunit",
        "quantityunit",
        "unitprice",
        "unitofsale",
        "productcode",
        "itemcode",
        "sku",
    )
    return any(marker in compact_text_value for marker in table_markers)


def _has_strong_summary_label(compact_text_value: str) -> bool:
    labels = (
        "subtotal",
        "totalexvat",
        "totalexclvat",
        "productstotalexvat",
        "grandtotal",
        "invoicetotal",
        "documenttotal",
        "totalprice",
        "totalatcheckout",
        "shippingcharges",
        "delivery",
        "vatsubtotal",
        "vatamount",
        "totalvat",
    )
    return any(label in compact_text_value for label in labels)


def _looks_like_money_or_number(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    if re.search(r"(gbp|usd|eur|£|\$|€)?\s*\d{1,3}(?:[, ]\d{3})*(?:[.,]\d+)?", text.casefold()):
        return True
    return compact.replace("gbp", "").replace("usd", "").replace("eur", "").isdigit()


def _has_currency_amount(text: str) -> bool:
    return bool(re.search(r"(gbp|usd|eur|£|\$|€)?\s*\d+[.,]\d{2}", text.casefold()))


def _row_has_money_or_amount(compact_text_value: str) -> bool:
    if any(currency in compact_text_value for currency in ("gbp", "usd", "eur", "total", "shipping", "discount")):
        return True
    return bool(re.search(r"\d+[.,]\d{2}", compact_text_value))


def _invoice_info_alternative_score(
    field: GroundedField,
    words: list[OCRWord],
    words_by_id: dict[str, OCRWord],
) -> float:
    if ".invoiceinfo." not in field.json_path.casefold():
        return 0.0
    all_words = list(words_by_id.values())
    context = _compact(_preceding_window_text(words, all_words))
    row_context = _compact(_row_label_text(words, all_words))
    field_name = field.field_name
    positive = {
        "issueDate": ["invoice date", "issue date", "order date", "payment date"],
        "dueDate": ["due date", "payment due"],
        "documentNumber": ["invoice number", "invoice no"],
        "purchaseOrderNumber": ["purchase order", "po number"],
        "paymentTerms": ["payment terms", "terms", "type"],
    }.get(field_name, [])
    negative = {
        "issueDate": ["tax point", "due date", "payment due", "booking date", "activation date"],
        "dueDate": ["invoice date", "issue date", "order date", "tax point", "activation date", "payment date"],
        "documentNumber": ["purchase order", "po number", "job no", "customer copy"],
        "purchaseOrderNumber": ["invoice number", "invoice no"],
        "paymentTerms": ["customer", "deliver to", "trade cash sale", "invoice totals"],
    }.get(field_name, [])
    score = 0.0
    positive_compacts = [_compact(label) for label in positive]
    negative_compacts = [_compact(label) for label in negative]
    row_positive = any(label in row_context for label in positive_compacts)
    row_negative = any(label in row_context for label in negative_compacts)
    if row_positive:
        score += 0.56
    elif any(label in context for label in positive_compacts):
        score += 0.32
    if row_negative:
        score -= 0.46
    elif not row_positive and any(label in context for label in negative_compacts):
        score -= 0.24
    if field_name == "documentNumber" and "customercopy" in context:
        score -= 0.34
    if field_name == "issueDate" and any(label in row_context for label in ("bookingdate", "activationdate")):
        score -= 0.36
    if field_name == "paymentTerms" and "tradecashsale" in row_context:
        score -= 0.42
    return score


def _field_from_alternative(
    original: GroundedField,
    alternative: CandidateSummary,
    words_by_id: dict[str, OCRWord],
    context_score: float,
) -> GroundedField:
    words = [words_by_id[word_id] for word_id in alternative.word_ids if word_id in words_by_id]
    pixel_boxes = [word.box_pixels for word in words]
    normalized_boxes = [word.box_normalized for word in words]
    union_pixels = union_pixel_boxes(pixel_boxes) if pixel_boxes else None
    union_normalized = union_normalized_boxes(normalized_boxes) if normalized_boxes else None
    line_boxes = _line_boxes(words)
    confidence = min(0.98, max(original.confidence or 0.0, alternative.confidence + min(0.04, context_score / 20)))
    return GroundedField(
        json_path=original.json_path,
        field_name=original.field_name,
        value=original.value,
        value_as_text=original.value_as_text,
        status=GroundingStatus.MATCHED,
        match_method=f"{alternative.match_method or original.match_method}_resolved_by_context",
        confidence=confidence,
        score_breakdown=original.score_breakdown,
        matched_text=alternative.matched_text,
        normalized_target=field_canonical(original.value_as_text, original.json_path)
        or normalize_text(original.value_as_text),
        normalized_matched_text=field_canonical(alternative.matched_text, original.json_path)
        or normalize_text(alternative.matched_text),
        word_ids=alternative.word_ids,
        word_boxes_pixels=pixel_boxes,
        word_boxes_normalized=normalized_boxes,
        line_boxes_pixels=line_boxes,
        union_box_pixels=union_pixels,
        union_box_normalized=union_normalized,
        candidate_rank=1,
        alternative_candidates=original.alternative_candidates,
        reason="Ambiguous candidates resolved using label, region, or sibling-field context",
    )


def _field_from_words(
    original: GroundedField,
    words: list[OCRWord],
    *,
    match_method: str,
    confidence: float,
    reason: str,
) -> GroundedField:
    pixel_boxes = [word.box_pixels for word in words]
    normalized_boxes = [word.box_normalized for word in words]
    union_pixels = union_pixel_boxes(pixel_boxes) if pixel_boxes else None
    union_normalized = union_normalized_boxes(normalized_boxes) if normalized_boxes else None
    matched_text = " ".join(word.text for word in sorted(words, key=lambda item: item.reading_order))
    return GroundedField(
        json_path=original.json_path,
        field_name=original.field_name,
        value=original.value,
        value_as_text=original.value_as_text,
        status=GroundingStatus.MATCHED,
        match_method=match_method,
        confidence=confidence,
        matched_text=matched_text,
        normalized_target=field_canonical(original.value_as_text, original.json_path)
        or normalize_text(original.value_as_text),
        normalized_matched_text=field_canonical(matched_text, original.json_path) or normalize_text(matched_text),
        word_ids=[word.id for word in sorted(words, key=lambda item: item.reading_order)],
        word_boxes_pixels=pixel_boxes,
        word_boxes_normalized=normalized_boxes,
        line_boxes_pixels=_line_boxes(words),
        union_box_pixels=union_pixels,
        union_box_normalized=union_normalized,
        candidate_rank=1,
        reason=reason,
    )


def _line_boxes(words: list[OCRWord]) -> list[BoundingBox]:
    grouped: dict[str, list[BoundingBox]] = {}
    for word in words:
        grouped.setdefault(word.line_id, []).append(word.box_pixels)
    return [union_pixel_boxes(boxes) for _, boxes in sorted(grouped.items())]


def _nearby_text(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    box = _word_union_normalized(words)
    min_order = min(word.reading_order for word in words)
    parts: list[str] = []
    for word in all_words:
        if word.id in word_ids:
            continue
        word_box = word.box_normalized
        same_or_above = word_box.y_min <= box.y_max + 0.055
        close_y = abs(((word_box.y_min + word_box.y_max) / 2) - ((box.y_min + box.y_max) / 2)) <= 0.07
        left_or_above = word.reading_order < min_order or word_box.y_max <= box.y_min + 0.055
        close_x = word_box.x_min <= box.x_max + 0.10 and word_box.x_max >= box.x_min - 0.10
        if same_or_above and left_or_above and (close_y or close_x):
            parts.append(word.text)
    return " ".join(parts)


def _role_local_context(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    box = _word_union_normalized(words)
    min_order = min(word.reading_order for word in words)
    parts: list[str] = []
    for word in all_words:
        if word.id in word_ids:
            continue
        word_box = word.box_normalized
        word_x_center = (word_box.x_min + word_box.x_max) / 2
        horizontally_aligned = box.x_min - 0.08 <= word_x_center <= box.x_max + 0.08
        just_above = 0 <= box.y_min - word_box.y_max <= 0.055
        same_line_left = abs(((word_box.y_min + word_box.y_max) / 2) - ((box.y_min + box.y_max) / 2)) <= 0.025 and word_box.x_max <= box.x_min
        near_before = 0 < min_order - word.reading_order <= 4
        if (horizontally_aligned and just_above) or (same_line_left and near_before):
            parts.append(word.text)
    return " ".join(parts)


def _reading_window_text(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    min_order = min(word.reading_order for word in words)
    max_order = max(word.reading_order for word in words)
    parts = [
        word.text
        for word in all_words
        if word.id not in word_ids and min_order - 8 <= word.reading_order <= max_order + 3
    ]
    return " ".join(parts)


def _preceding_window_text(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    min_order = min(word.reading_order for word in words)
    parts = [
        word.text
        for word in all_words
        if word.id not in word_ids and min_order - 10 <= word.reading_order < min_order
    ]
    return " ".join(parts)


def _row_label_text(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    box = _word_union_normalized(words)
    min_order = min(word.reading_order for word in words)
    y_center = (box.y_min + box.y_max) / 2
    parts: list[str] = []
    for word in all_words:
        if word.id in word_ids or word.reading_order >= min_order:
            continue
        word_box = word.box_normalized
        word_y_center = (word_box.y_min + word_box.y_max) / 2
        same_visual_row = abs(word_y_center - y_center) <= 0.025
        same_ocr_line = word.line_id in {item.line_id for item in words}
        left_of_candidate = word_box.x_max <= box.x_min + 0.02
        if left_of_candidate and (same_visual_row or same_ocr_line):
            parts.append(word.text)
    return " ".join(parts)


def _visual_row_text(words: list[OCRWord], all_words: list[OCRWord]) -> str:
    word_ids = {word.id for word in words}
    box = _word_union_normalized(words)
    y_center = (box.y_min + box.y_max) / 2
    candidate_line_ids = {word.line_id for word in words}
    parts: list[str] = []
    for word in all_words:
        if word.id in word_ids:
            continue
        word_box = word.box_normalized
        word_y_center = (word_box.y_min + word_box.y_max) / 2
        same_visual_row = abs(word_y_center - y_center) <= 0.025
        same_ocr_line = word.line_id in candidate_line_ids
        if same_visual_row or same_ocr_line:
            parts.append(word.text)
    return " ".join(parts)


def _word_union_normalized(words: list[OCRWord]) -> NormalizedBoundingBox:
    return union_normalized_boxes([word.box_normalized for word in words])


def _expand_normalized_box(
    box: NormalizedBoundingBox,
    *,
    x_pad: float,
    y_pad: float,
) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(
        x_min=max(0.0, box.x_min - x_pad),
        y_min=max(0.0, box.y_min - y_pad),
        x_max=min(1.0, box.x_max + x_pad),
        y_max=min(1.0, box.y_max + y_pad),
    )


def _box_center_inside(inner: NormalizedBoundingBox, outer: NormalizedBoundingBox) -> bool:
    x_center = (inner.x_min + inner.x_max) / 2
    y_center = (inner.y_min + inner.y_max) / 2
    return outer.x_min <= x_center <= outer.x_max and outer.y_min <= y_center <= outer.y_max


def _box_distance(left: NormalizedBoundingBox, right: NormalizedBoundingBox) -> float:
    left_center = ((left.x_min + left.x_max) / 2, (left.y_min + left.y_max) / 2)
    right_center = ((right.x_min + right.x_max) / 2, (right.y_min + right.y_max) / 2)
    return ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5


def _party_role(path: str) -> str | None:
    lowered = path.casefold()
    if ".seller." in lowered:
        return "seller"
    if ".customer." in lowered:
        return "customer"
    if ".shipto." in lowered:
        return "shipto"
    return None


def _party_prefix(path: str) -> str | None:
    pieces = path.split(".")
    lowered = [piece.casefold() for piece in pieces]
    for role in ("seller", "customer", "shipto"):
        if role in lowered:
            index = lowered.index(role)
            return ".".join(pieces[: index + 1]) + "."
    return None


def _is_currency_field(field: GroundedField) -> bool:
    return field.field_name == "currency" or field.json_path.casefold().endswith(".currency")


def _compact(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _configure_logging(config: GroundingConfig) -> None:
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
