from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only without optional dependency
    fuzz = None

from invoice_grounding.candidate_generation import union_normalized_boxes, union_pixel_boxes
from invoice_grounding.models import (
    BoundingBox,
    CandidateSpan,
    CandidateSummary,
    GroundableValue,
    GroundedField,
    GroundingConfig,
    GroundingStatus,
    NormalizedBoundingBox,
    OCRWord,
    ScoreBreakdown,
)
from invoice_grounding.normalization import (
    canonical_email,
    canonical_phone,
    compact_text,
    date_equivalent,
    field_canonical,
    looks_date_path,
    looks_email_path,
    looks_numeric_path,
    looks_phone_path,
    normalize_text,
    numeric_equivalent,
)

LOGGER = logging.getLogger(__name__)

FIELD_LABEL_HINTS: dict[str, list[str]] = {
    "documentNumber": ["invoice number", "invoice no", "invoice #", "inv no", "inv #", "receipt no"],
    "issueDate": ["invoice date", "issue date", "date issued"],
    "dueDate": ["due date", "payment due"],
    "purchaseOrderNumber": ["purchase order", "po number", "po no", "po #"],
    "customerNumber": ["customer number", "customer #", "account number"],
    "paymentTerms": ["payment terms", "terms"],
    "customerMemo": ["memo", "customer memo", "description", "reference"],
    "subtotal": ["subtotal", "sub total", "net amount", "net total", "total net amount", "goods"],
    "totalExcludingTax": ["total excluding tax", "net total", "net amount", "total net amount", "goods"],
    "taxAmount": ["tax amount", "total tax", "sales tax", "vat amount", "vat", "gst"],
    "taxPercentage": ["tax rate", "tax %", "vat rate", "gst rate", "vat %"],
    "taxName": ["sales tax", "vat", "v.a.t", "gst"],
    "discountTotal": ["discount", "less discount"],
    "discountPercentage": ["discount", "discount %"],
    "deposit": ["deposit", "amount paid", "less amount paid", "paid"],
    "totalIncludingTax": ["invoice total", "grand total", "document total", "total due", "total inc tax", "total"],
    "balanceDue": ["balance due", "amount due", "total due", "please pay", "net due"],
    "key": ["surcharge", "s/charge", "charge"],
    "value": ["surcharge", "s/charge", "charge"],
    "carrier": ["carrier", "ship via", "shipping carrier"],
    "trackingNumber": ["tracking", "tracking no", "tracking #", "waybill", "awb"],
    "name": ["seller", "vendor", "customer", "bill to", "ship to", "sold to"],
    "address": ["address", "bill to", "ship to", "sold to"],
    "phone": ["phone", "tel", "telephone", "mobile"],
    "email": ["email", "e-mail"],
    "currency": ["currency", "document total", "invoice total", "total"],
}

CURRENCY_SYMBOLS_BY_CODE = {
    "GBP": {"£"},
    "USD": {"$"},
    "EUR": {"€"},
    "JPY": {"¥"},
    "CNY": {"¥"},
    "INR": {"₹"},
}

LONG_TEXT_FIELDS = {"paymentTerms", "customerMemo"}
LONG_TEXT_STOPWORDS = {
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
    "upon",
    "with",
}

ORG_SUFFIX_TOKENS = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "limited",
    "ltd",
    "llc",
    "plc",
}

ADDRESS_COMPONENT_FIELDS = {"address"}


class _ScoredCandidate:
    def __init__(
        self,
        candidate: CandidateSpan,
        method: str,
        text_score: float,
        context_score: float,
        spatial_score: float,
        tightness_score: float,
        final_score: float,
    ) -> None:
        self.candidate = candidate
        self.method = method
        self.text_score = text_score
        self.context_score = context_score
        self.spatial_score = spatial_score
        self.tightness_score = tightness_score
        self.final_score = final_score


class _LongTextMatch:
    def __init__(self, candidate: CandidateSpan, score: float, coverage: float, order_score: float) -> None:
        self.candidate = candidate
        self.score = score
        self.coverage = coverage
        self.order_score = order_score


class _ComponentMatch:
    def __init__(self, candidate: CandidateSpan, score: float, method: str) -> None:
        self.candidate = candidate
        self.score = score
        self.method = method


def match_groundable_value(
    target: GroundableValue,
    candidates: list[CandidateSpan],
    words: list[OCRWord],
    config: GroundingConfig,
) -> GroundedField:
    if not target.groundable:
        return GroundedField(
            json_path=target.json_path,
            field_name=target.field_name,
            value=target.value,
            value_as_text=target.value_as_text,
            status=GroundingStatus.NOT_GROUNDABLE,
            reason=target.reason,
        )
    if not target.value_as_text:
        return _unmatched(target, "Empty value is not groundable")

    scored = [
        scored_candidate
        for candidate in candidates
        if (scored_candidate := _score_candidate(target, candidate, words, config)) is not None
    ]
    scored.sort(key=_ranking_key, reverse=True)

    if config.debug:
        LOGGER.debug(
            "Top candidates for %s: %s",
            target.json_path,
            [(item.candidate.text, item.method, round(item.final_score, 3)) for item in scored[:5]],
        )

    if not scored:
        return _unmatched(target, "No OCR candidate passed text prefilters")

    top = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = top.final_score - second.final_score if second is not None else None
    final_score = _confidence_with_margin(top.final_score, margin)
    alternatives = [_summary(item) for item in scored[1:4]]
    if top.final_score < config.min_confidence:
        return _unmatched(
            target,
            "Best candidate did not meet the minimum confidence threshold",
            alternatives=[_summary(item) for item in scored[:3]],
            confidence=final_score,
        )
    if _should_mark_ambiguous(target, top, second, margin, config):
        return GroundedField(
            json_path=target.json_path,
            field_name=target.field_name,
            value=target.value,
            value_as_text=target.value_as_text,
            status=GroundingStatus.AMBIGUOUS,
            match_method=top.method,
            confidence=final_score,
            score_breakdown=_breakdown(top, margin),
            normalized_target=field_canonical(target.value_as_text, target.json_path)
            or normalize_text(target.value_as_text),
            alternative_candidates=[_summary(item) for item in scored[:5]],
            reason="Multiple OCR candidates have similar scores",
        )

    return _matched_field(target, top, margin, alternatives, final_score)


def _score_candidate(
    target: GroundableValue,
    candidate: CandidateSpan,
    words: list[OCRWord],
    config: GroundingConfig,
) -> _ScoredCandidate | None:
    target_text = target.value_as_text or ""
    target_norm = normalize_text(target_text)
    candidate_norm = normalize_text(candidate.text)
    text_score = 0.0
    method = "unmatched"
    component_match = None
    if target_norm != candidate_norm and compact_text(target_text) != candidate.compact_text:
        component_match = _component_match(target, candidate)
    long_text_match = None
    if target_norm != candidate_norm and compact_text(target_text) != candidate.compact_text:
        long_text_match = _long_text_match(target, candidate)

    if component_match is not None:
        candidate = component_match.candidate
        text_score = component_match.score
        method = component_match.method
    elif long_text_match is not None:
        candidate = long_text_match.candidate
        text_score = long_text_match.score
        method = "partial_long_text_match"
    elif _is_currency_field(target) and config.enable_currency_symbol_mapping and _currency_symbol_match(target_text, candidate.text):
        text_score = 0.93
        method = "currency_symbol"
    elif target_norm == candidate_norm:
        text_score = 1.0
        method = "exact_raw_text"
    elif compact_text(target_text) == candidate.compact_text:
        text_score = 0.98
        method = "exact_compact_text"
    elif _tax_label_match(target, candidate.text):
        text_score = 0.96
        method = "tax_label_normalized"
    else:
        canonical_target = field_canonical(target_text, target.json_path)
        canonical_candidate = field_canonical(candidate.text, target.json_path)
        if canonical_target and canonical_candidate and canonical_target == canonical_candidate:
            text_score = 0.94
            method = "field_aware_canonical"
        elif _field_equivalent(target, candidate):
            text_score = 0.91
            method = "field_aware_equivalence"
        else:
            ratio = _fuzzy_ratio(normalize_text(target_text, keep_punctuation=False), normalize_text(candidate.text, keep_punctuation=False))
            length_penalty = _length_penalty(target_text, candidate.text)
            text_score = ratio * length_penalty
            method = "fuzzy"

    if text_score < 0.55:
        return None

    context_score = _context_score(target, candidate, words)
    context_score = max(0.0, min(1.0, context_score + _label_semantic_adjustment(target, candidate, words)))
    if method == "partial_long_text_match":
        context_score = max(context_score, 0.74)
    if method in {"organization_name_partial", "address_component_match", "tax_label_normalized"}:
        context_score = max(context_score, 0.72)
    if method == "fuzzy" and not _acceptable_fuzzy_match(target, text_score, context_score):
        return None
    tightness_score = _tightness_score(target_text, candidate.text)
    if method in {"field_aware_canonical", "field_aware_equivalence"} and _has_extra_label_text(target, candidate):
        text_score *= 0.93
    spatial_score = _spatial_score(target, candidate)
    ocr_score = max(0.0, min(1.0, candidate.mean_ocr_confidence))
    # Text remains the anchor, while context/tightness resolve common invoice duplicates.
    final = (
        (text_score * 0.62)
        + (ocr_score * 0.10)
        + (context_score * 0.16)
        + (spatial_score * 0.05)
        + (tightness_score * 0.07)
    )
    return _ScoredCandidate(candidate, method, text_score, context_score, spatial_score, tightness_score, final)


def _field_equivalent(target: GroundableValue, candidate: CandidateSpan) -> bool:
    path = target.json_path
    value = target.value_as_text
    if looks_numeric_path(path) and numeric_equivalent(value, candidate.text):
        return True
    if looks_date_path(path) and date_equivalent(value, candidate.text):
        return True
    if looks_phone_path(path):
        return bool(canonical_phone(value)) and canonical_phone(value) == canonical_phone(candidate.text)
    if looks_email_path(path):
        return bool(canonical_email(value)) and canonical_email(value) == canonical_email(candidate.text)
    return False


def _component_match(target: GroundableValue, candidate: CandidateSpan) -> _ComponentMatch | None:
    if target.field_name in ADDRESS_COMPONENT_FIELDS:
        return _address_component_match(target, candidate)
    if target.field_name == "name":
        return _organization_name_match(target, candidate)
    return None


def _address_component_match(target: GroundableValue, candidate: CandidateSpan) -> _ComponentMatch | None:
    target_terms = _text_terms(target.value_as_text or "")
    target_terms = [term for term in target_terms if len(term) > 1]
    if len(target_terms) < 2:
        return None
    matches = _match_terms_to_words(target_terms, candidate.words, threshold=0.78)
    coverage = len(matches) / len(target_terms)
    if coverage < 0.72 or len(matches) < 2:
        return None
    selected_words = [candidate.words[index] for index in sorted(set(matches.values()))]
    if not _spatially_coherent_words(selected_words):
        return None
    trimmed = _candidate_from_words(selected_words)
    mean_similarity = sum(_token_similarity(term, _best_word_term(term, selected_words)) for term in matches) / len(matches)
    score = min(0.93, (coverage * 0.62) + (mean_similarity * 0.28) + 0.05)
    return _ComponentMatch(trimmed, score, "address_component_match")


def _organization_name_match(target: GroundableValue, candidate: CandidateSpan) -> _ComponentMatch | None:
    if not _looks_like_organization_name(target.value_as_text or ""):
        return None
    target_terms = _organization_terms(target.value_as_text or "")
    candidate_terms = _organization_terms(candidate.text)
    if len(target_terms) < 2 or len(candidate_terms) < 2:
        return None
    matched_terms: dict[str, int] = {}
    used: set[int] = set()
    for target_term in target_terms:
        best_index = None
        best_score = 0.0
        for index, candidate_term in enumerate(candidate_terms):
            if index in used:
                continue
            score = _token_similarity(_singularize(target_term), _singularize(candidate_term))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.82:
            used.add(best_index)
            matched_terms[target_term] = best_index

    coverage = len(matched_terms) / len(target_terms)
    candidate_coverage = len(matched_terms) / len(candidate_terms)
    if len(matched_terms) < 2:
        return None
    if coverage < 0.50 and candidate_coverage < 0.78:
        return None
    selected_words = _words_for_terms(candidate.words, set(candidate_terms[index] for index in used))
    if not selected_words:
        selected_words = candidate.words
    trimmed = _candidate_from_words(selected_words)
    score = min(0.90, 0.56 + (coverage * 0.22) + (candidate_coverage * 0.12))
    return _ComponentMatch(trimmed, score, "organization_name_partial")


def _looks_like_organization_name(text: str) -> bool:
    terms = set(_text_terms(text))
    if terms & ORG_SUFFIX_TOKENS:
        return True
    return any(token in terms for token in {"company", "corp", "motor", "factors", "spares", "wholesale", "components", "food"})


def _match_terms_to_words(target_terms: list[str], words: list[OCRWord], *, threshold: float) -> dict[str, int]:
    matches: dict[str, int] = {}
    used: set[int] = set()
    word_terms = [(index, _text_terms(word.text)) for index, word in enumerate(words)]
    for target_term in target_terms:
        best_index = None
        best_score = 0.0
        for index, terms in word_terms:
            if index in used:
                continue
            for term in terms:
                score = _token_similarity(_singularize(target_term), _singularize(term))
                if score > best_score:
                    best_score = score
                    best_index = index
        if best_index is not None and best_score >= threshold:
            matches[target_term] = best_index
            used.add(best_index)
    return matches


def _best_word_term(target_term: str, words: list[OCRWord]) -> str:
    best_term = ""
    best_score = 0.0
    for word in words:
        for term in _text_terms(word.text):
            score = _token_similarity(target_term, term)
            if score > best_score:
                best_score = score
                best_term = term
    return best_term


def _words_for_terms(words: list[OCRWord], terms: set[str]) -> list[OCRWord]:
    selected: list[OCRWord] = []
    for word in words:
        word_terms = set(_organization_terms(word.text))
        if word_terms & terms:
            selected.append(word)
    return selected


def _organization_terms(text: str) -> list[str]:
    terms = [_singularize(term) for term in _text_terms(text)]
    return [term for term in terms if len(term) > 1 and term not in ORG_SUFFIX_TOKENS]


def _singularize(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _spatially_coherent_words(words: list[OCRWord]) -> bool:
    if len(words) <= 1:
        return True
    y_values = [(word.bbox_normalized[1] + word.bbox_normalized[3]) / 2 for word in words]
    return max(y_values) - min(y_values) <= 0.09


def _candidate_from_words(words: list[OCRWord]) -> CandidateSpan:
    ordered = sorted(words, key=lambda word: word.reading_order)
    text = " ".join(word.text for word in ordered)
    return CandidateSpan(
        id="|".join(word.id for word in ordered),
        text=text,
        compact_text=compact_text(text),
        words=ordered,
        word_ids=[word.id for word in ordered],
        line_ids=sorted({word.line_id for word in ordered}),
        bbox_pixels=union_pixel_boxes([word.box_pixels for word in ordered]),
        bbox_normalized=union_normalized_boxes([word.box_normalized for word in ordered]),
        mean_ocr_confidence=sum(word.confidence for word in ordered) / len(ordered),
    )


def _tax_label_match(target: GroundableValue, candidate_text: str) -> bool:
    if target.field_name != "taxName":
        return False
    target_label = _label_canonical(target.value_as_text or "")
    candidate_label = _label_canonical(candidate_text)
    aliases = {
        "vat": {"vat", "vattax"},
        "gst": {"gst", "gsttax"},
        "salestax": {"salestax", "tax"},
    }
    if target_label == candidate_label:
        return True
    for canonical, variants in aliases.items():
        if target_label == canonical and candidate_label in variants:
            return True
        if target_label in variants and candidate_label == canonical:
            return True
    return False


def _label_canonical(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if ch.isalnum())


def _context_score(target: GroundableValue, candidate: CandidateSpan, words: list[OCRWord]) -> float:
    hints = _field_label_hints(target)
    if not hints:
        return 0.5
    hint_compacts = [compact_text(hint, keep_punctuation=False) for hint in hints]
    min_order = min(word.reading_order for word in candidate.words)
    candidate_line_ids = set(candidate.line_ids)
    local_text = _local_label_context(candidate, words)
    same_line_before = [
        word.text
        for word in words
        if word.line_id in candidate_line_ids and word.reading_order < min_order
    ]
    nearby_before = [
        word.text
        for word in words
        if 0 < min_order - word.reading_order <= 12 and word.id not in candidate.word_ids
    ]
    context_text = compact_text(" ".join(same_line_before + nearby_before), keep_punctuation=False)
    if any(hint in local_text for hint in hint_compacts):
        return 0.98
    if any(hint in context_text for hint in hint_compacts):
        return 0.95
    if any(any(token in context_text for token in hint.split()) for hint in hint_compacts):
        return 0.7
    return 0.5


def _label_semantic_adjustment(target: GroundableValue, candidate: CandidateSpan, words: list[OCRWord]) -> float:
    if ".totals." not in target.json_path:
        return 0.0
    label_text = _same_and_nearby_label_text(candidate, words)
    field_name = target.field_name
    positive = {
        "totalIncludingTax": ["invoice total", "grand total", "document total", "total"],
        "deposit": ["less amount paid", "amount paid", "paid"],
        "balanceDue": ["amount due", "balance due", "total due"],
        "subtotal": ["subtotal", "sub total", "net amount", "goods"],
        "totalExcludingTax": ["total net amount", "net amount", "goods"],
        "taxPercentage": ["vat rate", "tax rate", "vat", "tax"],
        "taxAmount": ["vat", "tax"],
        "taxName": ["vat", "tax", "gst"],
    }.get(field_name, [])
    negative = {
        "totalIncludingTax": ["less amount paid", "amount paid", "amount due", "balance due", "zero rated"],
        "deposit": ["invoice total", "grand total", "amount due", "balance due"],
        "subtotal": ["invoice total", "grand total", "amount paid", "amount due"],
        "totalExcludingTax": ["invoice total", "grand total", "amount paid", "amount due"],
        "taxPercentage": ["zero rated"],
        "taxName": ["registered", "reg no", "regl no", "email", "web", "telephone"],
    }.get(field_name, [])
    adjustment = 0.0
    if any(label in label_text for label in [compact_text(item, keep_punctuation=False) for item in positive]):
        adjustment += 0.22
    if any(label in label_text for label in [compact_text(item, keep_punctuation=False) for item in negative]):
        adjustment -= 0.42 if field_name == "taxName" else 0.26
    return adjustment


def _same_and_nearby_label_text(candidate: CandidateSpan, words: list[OCRWord]) -> str:
    word_ids = set(candidate.word_ids)
    candidate_line_ids = set(candidate.line_ids)
    min_order = min(word.reading_order for word in candidate.words)
    candidate_y_min = candidate.bbox_normalized.y_min
    candidate_y_max = candidate.bbox_normalized.y_max
    parts: list[str] = []
    for word in words:
        if word.id in word_ids:
            continue
        same_line_before = word.line_id in candidate_line_ids and word.reading_order < min_order
        vertically_near = -0.035 <= word.bbox_normalized[1] - candidate_y_max <= 0.05
        above_same_column = 0 <= candidate_y_min - word.bbox_normalized[3] <= 0.055
        if same_line_before or vertically_near or above_same_column:
            parts.append(word.text)
    return compact_text(" ".join(parts), keep_punctuation=False)


def _field_label_hints(target: GroundableValue) -> list[str]:
    path = target.json_path.casefold()
    if ".shipto." in path:
        return ["ship to", "deliver to", "delivery to", "consignee"]
    if ".customer." in path:
        return ["customer", "invoice to", "bill to", "sold to", "account"]
    if ".seller." in path:
        return ["seller", "vendor", "from", "supplier", "tel", "email"]
    return FIELD_LABEL_HINTS.get(target.field_name, [])


def _local_label_context(candidate: CandidateSpan, words: list[OCRWord]) -> str:
    candidate_x_min = candidate.bbox_normalized.x_min
    candidate_x_max = candidate.bbox_normalized.x_max
    candidate_y_min = candidate.bbox_normalized.y_min
    parts: list[str] = []
    for word in words:
        if word.id in candidate.word_ids:
            continue
        word_y_max = word.bbox_normalized[3]
        word_x_center = (word.bbox_normalized[0] + word.bbox_normalized[2]) / 2
        horizontally_near = candidate_x_min - 0.08 <= word_x_center <= candidate_x_max + 0.08
        just_above = 0 <= candidate_y_min - word_y_max <= 0.045
        if horizontally_near and just_above:
            parts.append(word.text)
    return compact_text(" ".join(parts), keep_punctuation=False)


def _spatial_score(target: GroundableValue, candidate: CandidateSpan) -> float:
    # Penalize spans that are very tall relative to their word count; addresses still score acceptably.
    height = candidate.bbox_pixels.y_max - candidate.bbox_pixels.y_min
    widths = [word.bbox_pixels[2] - word.bbox_pixels[0] for word in candidate.words]
    mean_width = sum(widths) / len(widths)
    if height <= mean_width * 1.5:
        coherence = 0.9
    elif height <= mean_width * 4:
        coherence = 0.72
    else:
        coherence = 0.55
    if ".totals." in target.json_path:
        y_center = (candidate.bbox_normalized.y_min + candidate.bbox_normalized.y_max) / 2
        if y_center >= 0.55:
            return min(1.0, coherence + 0.08)
        if y_center < 0.35:
            return max(0.45, coherence - 0.18)
    return max(0.0, min(1.0, coherence + _role_region_adjustment(target, candidate)))


def _role_region_adjustment(target: GroundableValue, candidate: CandidateSpan) -> float:
    path = target.json_path.casefold()
    x_center = (candidate.bbox_normalized.x_min + candidate.bbox_normalized.x_max) / 2
    y_center = (candidate.bbox_normalized.y_min + candidate.bbox_normalized.y_max) / 2
    if ".seller." in path:
        return 0.08 if y_center < 0.35 else -0.08
    if ".customer." in path:
        if x_center < 0.48:
            return 0.08
        if x_center > 0.70:
            return -0.05
    if ".shipto." in path:
        if x_center > 0.45:
            return 0.08
        if x_center < 0.25:
            return -0.05
    return 0.0


def _fuzzy_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return max(fuzz.ratio(left, right), fuzz.token_sort_ratio(left, right)) / 100.0
    return SequenceMatcher(None, left, right).ratio()


def _length_penalty(left: str, right: str) -> float:
    left_len = max(1, len(compact_text(left)))
    right_len = max(1, len(compact_text(right)))
    ratio = min(left_len, right_len) / max(left_len, right_len)
    return max(0.55, ratio)


def _long_text_match(target: GroundableValue, candidate: CandidateSpan) -> _LongTextMatch | None:
    if not _is_long_text_target(target):
        return None
    target_text = target.value_as_text or ""
    target_terms = _important_long_text_terms(target_text)
    candidate_terms = _candidate_word_terms(candidate.words)
    if len(target_terms) < 3 or len(candidate_terms) < 3:
        return None

    matched_positions: list[int] = []
    similarities: list[float] = []
    used_positions: set[int] = set()
    for target_term in target_terms:
        best_index: int | None = None
        best_score = 0.0
        for index, candidate_term in candidate_terms:
            if index in used_positions:
                continue
            score = _token_similarity(target_term, candidate_term)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= _long_text_token_threshold(target_term):
            used_positions.add(best_index)
            matched_positions.append(best_index)
            similarities.append(best_score)

    coverage = len(matched_positions) / len(target_terms)
    if coverage < 0.62 or len(matched_positions) < 3:
        return None

    order_score = _order_score(matched_positions)
    mean_similarity = sum(similarities) / len(similarities)
    density = _match_density(matched_positions)
    score = (coverage * 0.55) + (mean_similarity * 0.25) + (order_score * 0.12) + (density * 0.08)
    if score < 0.68:
        return None

    trimmed = _trim_candidate_to_word_range(candidate, min(matched_positions), max(matched_positions))
    return _LongTextMatch(trimmed, min(0.93, score), coverage, order_score)


def _is_long_text_target(target: GroundableValue) -> bool:
    if target.field_name in LONG_TEXT_FIELDS:
        return True
    if looks_numeric_path(target.json_path) or looks_date_path(target.json_path):
        return False
    if looks_phone_path(target.json_path) or looks_email_path(target.json_path):
        return False
    return len(_important_long_text_terms(target.value_as_text or "")) >= 6


def _important_long_text_terms(text: str) -> list[str]:
    terms = _text_terms(text)
    important = [term for term in terms if term not in LONG_TEXT_STOPWORDS and (len(term) > 2 or term.isdigit())]
    return important or terms


def _candidate_word_terms(words: list[OCRWord]) -> list[tuple[int, str]]:
    terms: list[tuple[int, str]] = []
    for index, word in enumerate(words):
        for term in _text_terms(word.text):
            if term not in LONG_TEXT_STOPWORDS:
                terms.append((index, term))
    return terms


def _text_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text, keep_punctuation=False))


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return _fuzzy_ratio(left, right)


def _long_text_token_threshold(term: str) -> float:
    if any(char.isdigit() for char in term):
        return 0.86
    if len(term) <= 4:
        return 0.84
    return 0.74


def _order_score(positions: list[int]) -> float:
    if len(positions) <= 1:
        return 1.0
    ordered_pairs = sum(1 for left, right in zip(positions, positions[1:]) if right >= left)
    return ordered_pairs / (len(positions) - 1)


def _match_density(positions: list[int]) -> float:
    if not positions:
        return 0.0
    span = max(positions) - min(positions) + 1
    return len(set(positions)) / span


def _trim_candidate_to_word_range(candidate: CandidateSpan, start: int, end: int) -> CandidateSpan:
    words = candidate.words[start : end + 1]
    if not words:
        return candidate
    text = " ".join(word.text for word in words)
    return CandidateSpan(
        id="|".join(word.id for word in words),
        text=text,
        compact_text=compact_text(text),
        words=words,
        word_ids=[word.id for word in words],
        line_ids=sorted({word.line_id for word in words}),
        bbox_pixels=union_pixel_boxes([word.box_pixels for word in words]),
        bbox_normalized=union_normalized_boxes([word.box_normalized for word in words]),
        mean_ocr_confidence=sum(word.confidence for word in words) / len(words),
    )


def _tightness_score(target_text: str, candidate_text: str) -> float:
    target_compact = compact_text(target_text, keep_punctuation=False)
    candidate_compact = compact_text(candidate_text, keep_punctuation=False)
    if not target_compact or not candidate_compact:
        return 0.5
    if target_compact == candidate_compact:
        return 1.0
    if target_compact in candidate_compact:
        return max(0.42, min(0.95, len(target_compact) / len(candidate_compact)))
    return max(0.35, min(len(target_compact), len(candidate_compact)) / max(len(target_compact), len(candidate_compact)))


def _has_extra_label_text(target: GroundableValue, candidate: CandidateSpan) -> bool:
    target_text = target.value_as_text or ""
    target_compact = compact_text(target_text, keep_punctuation=False)
    candidate_compact = compact_text(candidate.text, keep_punctuation=False)
    if target_compact == candidate_compact:
        return False
    if target_compact not in candidate_compact:
        return False
    target_has_letters = any(ch.isalpha() for ch in target_compact)
    candidate_has_extra_letters = any(ch.isalpha() for ch in candidate_compact.replace(target_compact, "", 1))
    return candidate_has_extra_letters or len(candidate.words) > max(1, len(str(target_text).split()) + 1)


def _acceptable_fuzzy_match(target: GroundableValue, text_score: float, context_score: float) -> bool:
    if target.field_name in {"name", "address"}:
        return text_score >= 0.82 or (text_score >= 0.76 and context_score >= 0.9)
    if looks_numeric_path(target.json_path):
        return text_score >= 0.78
    return text_score >= 0.70 or (text_score >= 0.64 and context_score >= 0.9)


def _is_currency_field(target: GroundableValue) -> bool:
    return target.field_name == "currency" or target.json_path.casefold().endswith(".currency")


def _currency_symbol_match(target_text: str, candidate_text: str) -> bool:
    symbols = CURRENCY_SYMBOLS_BY_CODE.get(target_text.upper().strip())
    if not symbols:
        return False
    candidate = normalize_text(candidate_text)
    return candidate in symbols


def _confidence_with_margin(score: float, margin: float | None) -> float:
    if margin is None:
        return min(0.99, max(0.0, score + 0.03))
    return min(0.99, max(0.0, score + min(0.04, margin / 3)))


def _ranking_key(item: _ScoredCandidate) -> tuple[float, float, float, float, int]:
    return (
        item.final_score,
        item.text_score,
        item.context_score,
        item.tightness_score,
        -len(item.candidate.words),
    )


def _should_mark_ambiguous(
    target: GroundableValue,
    top: _ScoredCandidate,
    second: _ScoredCandidate | None,
    margin: float | None,
    config: GroundingConfig,
) -> bool:
    if second is None or margin is None or margin >= config.ambiguity_margin:
        return False
    if not _same_strength(top, second):
        return False
    if _overlapping_candidates(top, second):
        return False
    if top.text_score >= 0.98 and second.text_score < 0.98:
        return False
    if top.tightness_score - second.tightness_score >= 0.22:
        return False
    if top.context_score - second.context_score >= 0.18:
        return False
    if _is_currency_field(target):
        return False
    return True


def _same_strength(left: _ScoredCandidate, right: _ScoredCandidate) -> bool:
    return left.text_score >= 0.88 and right.text_score >= 0.88


def _overlapping_candidates(left: _ScoredCandidate, right: _ScoredCandidate) -> bool:
    return bool(set(left.candidate.word_ids) & set(right.candidate.word_ids))


def _matched_field(
    target: GroundableValue,
    scored: _ScoredCandidate,
    margin: float | None,
    alternatives: list[CandidateSummary],
    confidence: float,
) -> GroundedField:
    candidate = scored.candidate
    word_boxes = [word.box_pixels for word in candidate.words]
    normalized_word_boxes = [word.box_normalized for word in candidate.words]
    line_boxes = _line_boxes(candidate.words)
    return GroundedField(
        json_path=target.json_path,
        field_name=target.field_name,
        value=target.value,
        value_as_text=target.value_as_text,
        status=GroundingStatus.MATCHED,
        match_method=scored.method,
        confidence=confidence,
        score_breakdown=_breakdown(scored, margin),
        matched_text=candidate.text,
        normalized_target=field_canonical(target.value_as_text, target.json_path)
        or normalize_text(target.value_as_text),
        normalized_matched_text=field_canonical(candidate.text, target.json_path) or normalize_text(candidate.text),
        word_ids=candidate.word_ids,
        word_boxes_pixels=word_boxes,
        word_boxes_normalized=normalized_word_boxes,
        line_boxes_pixels=line_boxes,
        union_box_pixels=candidate.bbox_pixels,
        union_box_normalized=candidate.bbox_normalized,
        candidate_rank=1,
        alternative_candidates=alternatives,
        reason=(
            "Matched by OCR-noise-tolerant partial long-text token coverage"
            if scored.method == "partial_long_text_match"
            else None
        ),
    )


def _breakdown(scored: _ScoredCandidate, margin: float | None) -> ScoreBreakdown:
    return ScoreBreakdown(
        text_score=scored.text_score,
        ocr_score=scored.candidate.mean_ocr_confidence,
        context_score=scored.context_score,
        spatial_score=(scored.spatial_score + scored.tightness_score) / 2,
        ambiguity_margin=margin,
        final_score=scored.final_score,
    )


def _summary(scored: _ScoredCandidate) -> CandidateSummary:
    return CandidateSummary(
        matched_text=scored.candidate.text,
        confidence=scored.final_score,
        word_ids=scored.candidate.word_ids,
        match_method=scored.method,
    )


def _unmatched(
    target: GroundableValue,
    reason: str,
    *,
    alternatives: list[CandidateSummary] | None = None,
    confidence: float | None = None,
) -> GroundedField:
    return GroundedField(
        json_path=target.json_path,
        field_name=target.field_name,
        value=target.value,
        value_as_text=target.value_as_text,
        status=GroundingStatus.UNMATCHED,
        confidence=confidence,
        alternative_candidates=alternatives or [],
        reason=reason,
    )


def _line_boxes(words: list[OCRWord]) -> list[BoundingBox]:
    grouped: dict[str, list[BoundingBox]] = {}
    for word in words:
        grouped.setdefault(word.line_id, []).append(word.box_pixels)
    return [union_pixel_boxes(boxes) for _, boxes in sorted(grouped.items())]


def inherit_field(target: GroundableValue, source: GroundedField) -> GroundedField:
    return GroundedField(
        json_path=target.json_path,
        field_name=target.field_name,
        value=target.value,
        value_as_text=target.value_as_text,
        status=GroundingStatus.INHERITED,
        match_method="inherited_from_printed_date",
        confidence=source.confidence,
        score_breakdown=source.score_breakdown,
        matched_text=source.matched_text,
        normalized_target=field_canonical(target.value_as_text, target.json_path)
        or normalize_text(target.value_as_text),
        normalized_matched_text=source.normalized_matched_text,
        word_ids=source.word_ids,
        word_boxes_pixels=source.word_boxes_pixels,
        word_boxes_normalized=source.word_boxes_normalized,
        line_boxes_pixels=source.line_boxes_pixels,
        union_box_pixels=source.union_box_pixels,
        union_box_normalized=source.union_box_normalized,
        inherited_from=source.json_path,
        reason="ISO-normalized field inherits evidence from the corresponding printed date",
    )
