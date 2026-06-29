from __future__ import annotations

from invoice_grounding.candidate_generation import generate_candidates
from invoice_grounding.matching import match_groundable_value
from invoice_grounding.models import GroundableValue, GroundingConfig, GroundingStatus


def test_exact_span_matching(make_word) -> None:
    words = [
        make_word("Grand", 0, 100, 100, 160, 125),
        make_word("Total", 1, 165, 100, 220, 125),
        make_word("$1,234.50", 2, 260, 100, 360, 125),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.totalIncludingTax.originalValue",
        field_name="totalIncludingTax",
        value="$1,234.50",
        value_as_text="$1,234.50",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[2].id]
    assert field.union_box_pixels is not None
    assert field.union_box_pixels.x_min == 260


def test_ocr_split_identifier_compact_match(make_word) -> None:
    words = [
        make_word("Invoice", 0, 10, 10, 80, 30),
        make_word("#", 1, 85, 10, 95, 30),
        make_word("INV", 2, 120, 10, 160, 30),
        make_word("-", 3, 162, 10, 170, 30),
        make_word("10042", 4, 172, 10, 230, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.invoiceInfo.documentNumber",
        field_name="documentNumber",
        value="INV-10042",
        value_as_text="INV-10042",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "exact_compact_text"
    assert field.matched_text == "INV - 10042"


def test_multiline_address_matching(make_word) -> None:
    words = [
        make_word("123", 0, 20, 20, 50, 40, line=0),
        make_word("Main", 1, 55, 20, 100, 40, line=0),
        make_word("St", 2, 105, 20, 130, 40, line=0),
        make_word("Austin,", 3, 20, 45, 80, 65, line=1),
        make_word("TX", 4, 85, 45, 110, 65, line=1),
        make_word("78701", 5, 115, 45, 165, 65, line=1),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.parties.customer.addressStructured.address",
        field_name="address",
        value="123 Main St Austin, TX 78701",
        value_as_text="123 Main St Austin, TX 78701",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert len(field.line_boxes_pixels) == 2


def test_duplicate_value_becomes_ambiguous(make_word) -> None:
    words = [
        make_word("1", 0, 10, 10, 20, 30, line=0),
        make_word("1", 1, 10, 50, 20, 70, line=1),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.taxAmount.originalValue",
        field_name="taxAmount",
        value="1",
        value_as_text="1",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.AMBIGUOUS
    assert len(field.alternative_candidates) >= 2


def test_tight_exact_value_beats_larger_containing_span(make_word) -> None:
    words = [
        make_word("VAT", 0, 10, 10, 40, 30),
        make_word("1.81", 1, 50, 10, 90, 30),
        make_word("Total", 2, 100, 10, 145, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.taxAmount.originalValue",
        field_name="taxAmount",
        value="1.81",
        value_as_text="1.81",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.matched_text == "1.81"


def test_numeric_candidate_clustering_keeps_one_tight_span_per_occurrence(make_word) -> None:
    words = [
        make_word("Subtotal", 0, 100, 100, 180, 125),
        make_word("24.14", 1, 220, 100, 280, 125),
        make_word("Line", 2, 20, 200, 60, 225, line=1),
        make_word("24.14", 3, 800, 200, 860, 225, line=1),
        make_word("24.14", 4, 220, 300, 280, 325, line=2),
        make_word("Total", 5, 700, 300, 760, 325, line=2),
        make_word("(GBP)", 6, 765, 300, 825, 325, line=2),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.subtotal.originalValue",
        field_name="subtotal",
        value="24.14",
        value_as_text="24.14",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    candidates = [(field.matched_text, field.word_ids)] if field.matched_text else []
    candidates.extend((item.matched_text, item.word_ids) for item in field.alternative_candidates)
    distinct = {tuple(word_ids) for _, word_ids in candidates}

    assert distinct == {(words[1].id,), (words[3].id,), (words[4].id,)}
    assert all(text == "24.14" for text, _ in candidates)


def test_numeric_candidate_clustering_preserves_distinct_tax_occurrences(make_word) -> None:
    words = [
        make_word("VAT", 0, 100, 100, 140, 125),
        make_word("TOTAL", 1, 145, 100, 205, 125),
        make_word("336.00", 2, 700, 100, 780, 125),
        make_word("TOTAL", 3, 100, 140, 160, 165, line=1),
        make_word("VAT", 4, 100, 220, 140, 245, line=2),
        make_word("@", 5, 145, 220, 160, 245, line=2),
        make_word("20%", 6, 165, 220, 205, 245, line=2),
        make_word("336.00", 7, 700, 220, 780, 245, line=2),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.taxAmount.originalValue",
        field_name="taxAmount",
        value="336.00",
        value_as_text="336.00",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    candidates = [(field.matched_text, field.word_ids)] if field.matched_text else []
    candidates.extend((item.matched_text, item.word_ids) for item in field.alternative_candidates)
    distinct = {tuple(word_ids) for _, word_ids in candidates}

    assert distinct == {(words[2].id,), (words[7].id,)}
    assert all(text == "336.00" for text, _ in candidates)


def test_invoice_summary_candidates_suppress_table_body_occurrences(make_word) -> None:
    words = [
        make_word("Description", 0, 80, 180, 180, 205, width=1000, height=800),
        make_word("Qty", 1, 300, 180, 340, 205, width=1000, height=800),
        make_word("Unit", 2, 500, 180, 540, 205, width=1000, height=800),
        make_word("Price", 3, 545, 180, 595, 205, width=1000, height=800),
        make_word("Subtotal", 4, 750, 180, 835, 205, width=1000, height=800),
        make_word("Service", 5, 80, 240, 150, 265, line=1, width=1000, height=800),
        make_word("1", 6, 310, 240, 325, 265, line=1, width=1000, height=800),
        make_word("50.00", 7, 520, 240, 580, 265, line=1, width=1000, height=800),
        make_word("50.00", 8, 760, 240, 820, 265, line=1, width=1000, height=800),
        make_word("Subtotal:", 9, 650, 340, 735, 365, line=2, width=1000, height=800),
        make_word("£50.00", 10, 760, 340, 830, 365, line=2, width=1000, height=800),
        make_word("Total:", 11, 680, 380, 735, 405, line=3, width=1000, height=800),
        make_word("£50.00", 12, 760, 380, 830, 405, line=3, width=1000, height=800),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.subtotal.originalValue",
        field_name="subtotal",
        value="£50.00",
        value_as_text="£50.00",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    candidate_ids = {tuple(item.word_ids) for item in field.alternative_candidates}
    if field.word_ids:
        candidate_ids.add(tuple(field.word_ids))

    assert (words[7].id,) not in candidate_ids
    assert (words[8].id,) not in candidate_ids
    assert (words[10].id,) in candidate_ids
    assert (words[12].id,) in candidate_ids


def test_table_candidate_remains_when_no_summary_occurrence_exists(make_word) -> None:
    words = [
        make_word("Description", 0, 80, 180, 180, 205, width=1000, height=800),
        make_word("Qty", 1, 300, 180, 340, 205, width=1000, height=800),
        make_word("Unit", 2, 500, 180, 540, 205, width=1000, height=800),
        make_word("Price", 3, 545, 180, 595, 205, width=1000, height=800),
        make_word("Service", 4, 80, 240, 150, 265, line=1, width=1000, height=800),
        make_word("1", 5, 310, 240, 325, 265, line=1, width=1000, height=800),
        make_word("62.50", 6, 520, 240, 580, 265, line=1, width=1000, height=800),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.totalExcludingTax.originalValue",
        field_name="totalExcludingTax",
        value="62.50",
        value_as_text="62.50",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())

    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[6].id]


def test_currency_symbol_can_ground_normalized_currency(make_word) -> None:
    words = [
        make_word("Invoice", 0, 10, 10, 60, 30),
        make_word("Total", 1, 65, 10, 110, 30),
        make_word("£", 2, 120, 10, 135, 30),
        make_word("10.00", 3, 140, 10, 190, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.currency",
        field_name="currency",
        value="GBP",
        value_as_text="GBP",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "currency_symbol"


def test_long_text_partial_match_tolerates_extra_and_noisy_ocr_words(make_word) -> None:
    words = [
        make_word("all", 0, 10, 10, 35, 30),
        make_word("accounts", 1, 40, 10, 100, 30),
        make_word("are", 2, 105, 10, 130, 30),
        make_word("strictly", 3, 135, 10, 195, 30),
        make_word("payable", 4, 200, 10, 260, 30),
        make_word("with", 5, 265, 10, 300, 30),
        make_word("CASH", 6, 305, 10, 350, 30),
        make_word("or", 7, 355, 10, 375, 30),
        make_word("CARD", 8, 380, 10, 425, 30),
        make_word("CoChdlc", 9, 430, 10, 500, 30),
        make_word("upon", 10, 505, 10, 550, 30),
        make_word("delivery", 11, 555, 10, 620, 30),
        make_word("or", 12, 625, 10, 645, 30),
        make_word("collection.", 13, 650, 10, 735, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.invoiceInfo.paymentTerms",
        field_name="paymentTerms",
        value="strictly payable with CASH or CARD upon delivery or collection",
        value_as_text="strictly payable with CASH or CARD upon delivery or collection",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "partial_long_text_match"
    assert "strictly payable" in field.matched_text
    assert field.reason == "Matched by OCR-noise-tolerant partial long-text token coverage"


def test_long_text_partial_match_trims_extra_prefix(make_word) -> None:
    words = [
        make_word("Tilbury", 0, 10, 10, 70, 30),
        make_word("STORAGE", 1, 75, 10, 145, 30),
        make_word("DELIVERY", 2, 150, 10, 225, 30),
        make_word("B0-87606", 3, 230, 10, 310, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.invoiceInfo.customerMemo",
        field_name="customerMemo",
        value="STORAGE DELIVERY BO-87606",
        value_as_text="STORAGE DELIVERY BO-87606",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "partial_long_text_match"
    assert field.matched_text == "STORAGE DELIVERY B0-87606"


def test_tax_name_normalizes_punctuation(make_word) -> None:
    words = [make_word("V.A.T.:", 0, 10, 10, 70, 30)]
    target = GroundableValue(
        json_path="invoiceOutputData.totals.taxName",
        field_name="taxName",
        value="V.A.T",
        value_as_text="V.A.T",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "tax_label_normalized"


def test_organization_name_partial_match_tolerates_short_printed_name(make_word) -> None:
    words = [
        make_word("ROUND", 0, 10, 10, 60, 30),
        make_word("TOWERS", 1, 65, 10, 130, 30),
        make_word("SPARES,", 2, 135, 10, 205, 30),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.parties.seller.name",
        field_name="name",
        value="Round Tower Spares & Motor Factors",
        value_as_text="Round Tower Spares & Motor Factors",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "organization_name_partial"
    assert field.matched_text == "ROUND TOWERS SPARES,"


def test_address_component_match_tolerates_ocr_suffix_noise(make_word) -> None:
    words = [
        make_word("Oban", 0, 10, 10, 55, 30),
        make_word("Roadi", 1, 60, 10, 110, 30),
        make_word("Longford", 2, 10, 35, 85, 55, line=1),
    ]
    target = GroundableValue(
        json_path="invoiceOutputData.parties.seller.addressStructured.address",
        field_name="address",
        value="Oban Road, Longford",
        value_as_text="Oban Road, Longford",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "address_component_match"
    assert field.matched_text == "Oban Roadi Longford"


def test_union_box_calculation(make_word) -> None:
    words = [make_word("ACME", 0, 10, 10, 50, 30), make_word("Corp", 1, 60, 12, 95, 32)]
    target = GroundableValue(
        json_path="invoiceOutputData.parties.seller.name",
        field_name="name",
        value="ACME Corp",
        value_as_text="ACME Corp",
        groundable=True,
    )
    field = match_groundable_value(target, generate_candidates(words), words, GroundingConfig())
    assert field.union_box_pixels is not None
    assert field.union_box_pixels.x_min == 10
    assert field.union_box_pixels.x_max == 95
