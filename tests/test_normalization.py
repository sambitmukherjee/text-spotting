from __future__ import annotations

from invoice_grounding.normalization import (
    canonical_email,
    canonical_phone,
    compact_text,
    date_equivalent,
    normalize_text,
    numeric_equivalent,
    parse_numeric,
)


def test_unicode_whitespace_quotes_and_dashes_are_normalized() -> None:
    assert normalize_text("  ACME\u00a0\u201cInvoice\u201d \u2013 42  ") == 'acme "invoice" - 42'


def test_compact_text_handles_ocr_inserted_spaces() -> None:
    assert compact_text("INV - 10042") == "inv-10042"


def test_currency_numeric_equivalence() -> None:
    assert numeric_equivalent("$1,234.50", "1234.5")
    assert numeric_equivalent("1 234,50", "1234.50")


def test_parentheses_negative_amount() -> None:
    parsed = parse_numeric("(102.68)")
    assert parsed is not None
    assert str(parsed.value) == "-102.68"


def test_percentage_normalization() -> None:
    assert numeric_equivalent("10%", "10.0 %")


def test_phone_and_email_normalization() -> None:
    assert canonical_phone("+1 (555) 010-0200") == "15550100200"
    assert canonical_email("AR @ Example . COM") == "ar@example.com"


def test_date_equivalence() -> None:
    assert date_equivalent("2026-06-01", "June 1, 2026")
