from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from dateutil import parser as date_parser


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s@.+%()-]", re.UNICODE)
_NUMERIC_TOKEN_RE = re.compile(r"[-+()]?[\$€£¥₹]?\s*[A-Z]{0,3}\s*[\d.,\s]+%?")
_CURRENCY_SYMBOLS = "$€£¥₹"
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


@dataclass(frozen=True)
class NumericCanonical:
    value: Decimal
    is_percent: bool
    is_negative: bool


def normalize_text(value: object, *, keep_punctuation: bool = True) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.translate(_DASH_TRANSLATION).translate(_QUOTE_TRANSLATION)
    text = _WHITESPACE_RE.sub(" ", text).strip().casefold()
    if not keep_punctuation:
        text = _PUNCT_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def compact_text(value: object, *, keep_punctuation: bool = True) -> str:
    return normalize_text(value, keep_punctuation=keep_punctuation).replace(" ", "")


def normalize_identifier(value: object) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def canonical_phone(value: object) -> str:
    text = normalize_text(value)
    return re.sub(r"\D+", "", text)


def canonical_email(value: object) -> str:
    text = compact_text(value)
    text = text.replace(" at ", "@").replace("(at)", "@")
    return text


def _strip_currency_and_codes(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\b[A-Z]{3}\b", "", text, flags=re.IGNORECASE)
    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    return text.strip()


def parse_numeric(value: object) -> NumericCanonical | None:
    if value is None:
        return None
    raw = normalize_text(value).upper()
    if not raw:
        return None

    is_percent = "%" in raw
    negative = False
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1]
    if raw.startswith("-"):
        negative = True
        raw = raw[1:]

    raw = _strip_currency_and_codes(raw.replace("%", ""))
    raw = raw.replace(" ", "")
    raw = re.sub(r"[^0-9,.-]", "", raw)
    if not raw or not re.search(r"\d", raw):
        return None

    candidates = _numeric_string_candidates(raw)
    parsed: set[Decimal] = set()
    for candidate in candidates:
        try:
            parsed.add(Decimal(candidate))
        except InvalidOperation:
            continue
    if len(parsed) != 1:
        return None
    number = parsed.pop()
    if negative:
        number = -abs(number)
    return NumericCanonical(value=number.normalize(), is_percent=is_percent, is_negative=number < 0)


def _numeric_string_candidates(raw: str) -> set[str]:
    if "," not in raw and "." not in raw:
        return {raw}

    candidates: set[str] = set()
    last_comma = raw.rfind(",")
    last_dot = raw.rfind(".")
    separators = [sep for sep in (",", ".") if sep in raw]

    if len(separators) == 2:
        decimal_sep = "," if last_comma > last_dot else "."
        thousand_sep = "." if decimal_sep == "," else ","
        candidates.add(raw.replace(thousand_sep, "").replace(decimal_sep, "."))
        return candidates

    sep = separators[0]
    parts = raw.split(sep)
    if len(parts) > 2:
        if all(len(part) == 3 for part in parts[1:]):
            candidates.add("".join(parts))
        return candidates

    left, right = parts
    if not left or not right:
        return set()
    if len(right) == 3 and len(left) <= 3:
        candidates.add(left + right)
    if len(right) in {1, 2}:
        candidates.add(left + "." + right)
    return candidates


def numeric_equivalent(left: object, right: object) -> bool:
    parsed_left = parse_numeric(left)
    parsed_right = parse_numeric(right)
    return parsed_left is not None and parsed_left == parsed_right


def parse_dates(value: object) -> set[date]:
    text = normalize_text(value)
    if not text:
        return set()
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso_match:
        try:
            return {date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))}
        except ValueError:
            return set()
    # Avoid turning standalone numbers such as invoice IDs into dates.
    if not re.search(r"[/-]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text):
        return set()
    if not _has_complete_date_evidence(text):
        return set()

    parsed: set[date] = set()
    for dayfirst in (False, True):
        try:
            dt = date_parser.parse(text, fuzzy=False, dayfirst=dayfirst)
        except (ValueError, OverflowError):
            continue
        parsed.add(dt.date())
    return parsed


def _has_complete_date_evidence(text: str) -> bool:
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text):
        return True
    if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", text):
        return True
    has_year = bool(re.search(r"\b\d{4}\b|\b\d{2}\b", text))
    has_month_name = bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text))
    has_day = bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", text))
    return has_year and has_month_name and has_day


def date_equivalent(left: object, right: object) -> bool:
    left_dates = parse_dates(left)
    right_dates = parse_dates(right)
    if not left_dates or not right_dates:
        return False
    if len(left_dates) > 1 or len(right_dates) > 1:
        return False
    return left_dates == right_dates


def looks_numeric_path(path: str) -> bool:
    lowered = path.casefold()
    return any(
        token in lowered
        for token in (
            "amount",
            "total",
            "price",
            "quantity",
            "percent",
            "percentage",
            "rate",
            "deposit",
            "balance",
            "subtotal",
            "value",
        )
    )


def looks_date_path(path: str) -> bool:
    return "date" in path.casefold()


def looks_phone_path(path: str) -> bool:
    return "phone" in path.casefold()


def looks_email_path(path: str) -> bool:
    return "email" in path.casefold()


def field_canonical(value: object, path: str) -> str | None:
    if looks_phone_path(path):
        phone = canonical_phone(value)
        return phone or None
    if looks_email_path(path):
        email = canonical_email(value)
        return email or None
    if looks_numeric_path(path):
        parsed = parse_numeric(value)
        if parsed is not None:
            suffix = "%" if parsed.is_percent else ""
            return f"{parsed.value}{suffix}"
    if looks_date_path(path):
        dates = parse_dates(value)
        if len(dates) == 1:
            return next(iter(dates)).isoformat()
    return normalize_text(value, keep_punctuation=False)
