from __future__ import annotations

import json
import os

import pytest
from PIL import Image, ImageDraw

from invoice_grounding.grounding import ground_invoice_values, ground_invoice_values_from_ocr
from invoice_grounding.models import GroundingConfig, GroundingStatus


def test_grounding_inherits_normalized_date(make_word) -> None:
    image = Image.new("RGB", (500, 300), "white")
    words = [
        make_word("Invoice", 0, 20, 20, 80, 40, width=500, height=300),
        make_word("Date", 1, 85, 20, 125, 40, width=500, height=300),
        make_word("June", 2, 150, 20, 195, 40, width=500, height=300),
        make_word("1,", 3, 200, 20, 220, 40, width=500, height=300),
        make_word("2026", 4, 225, 20, 270, 40, width=500, height=300),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"invoiceInfo": {"issueDate": "June 1, 2026", "issueDateISO": "2026-06-01"}}},
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    assert by_path["invoiceOutputData.invoiceInfo.issueDate"].status == GroundingStatus.MATCHED
    iso = by_path["invoiceOutputData.invoiceInfo.issueDateISO"]
    assert iso.status == GroundingStatus.INHERITED
    assert iso.inherited_from == "invoiceOutputData.invoiceInfo.issueDate"
    assert iso.word_ids == by_path["invoiceOutputData.invoiceInfo.issueDate"].word_ids


def test_not_groundable_field_in_result() -> None:
    image = Image.new("RGB", (200, 100), "white")
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"invoiceStatus": "unpaid"}},
        [],
        config=GroundingConfig(),
    )
    assert result.fields[0].status == GroundingStatus.NOT_GROUNDABLE


def test_normalized_currency_without_printed_symbol_is_not_groundable(make_word) -> None:
    image = Image.new("RGB", (500, 200), "white")
    words = [
        make_word("Invoice", 0, 20, 20, 90, 40, width=500, height=200),
        make_word("Total", 1, 95, 20, 145, 40, width=500, height=200),
        make_word("10.00", 2, 150, 20, 205, 40, width=500, height=200),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"currency": "GBP"}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.NOT_GROUNDABLE
    assert "Normalized currency code" in (field.reason or "")


def test_result_json_includes_ambiguous_and_unmatched_review_lists(make_word) -> None:
    image = Image.new("RGB", (500, 300), "white")
    words = [
        make_word("1", 0, 20, 20, 35, 40, width=500, height=300),
        make_word("1", 1, 20, 70, 35, 90, line=1, width=500, height=300),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "invoiceInfo": {"documentNumber": "INV-404"},
                "totals": {"taxAmount": {"originalValue": "1"}},
            }
        },
        words,
        config=GroundingConfig(),
    )

    payload = json.loads(result.model_dump_json())
    assert len(payload["ambiguous"]) == 1
    assert len(payload["unmatched"]) == 1
    assert payload["ambiguous"][0]["json_path"] == "invoiceOutputData.totals.taxAmount.originalValue"
    assert payload["ambiguous"][0]["value_as_text"] == "1"
    assert payload["ambiguous"][0]["alternative_candidates"]
    assert payload["unmatched"][0] == {
        "json_path": "invoiceOutputData.invoiceInfo.documentNumber",
        "field_name": "documentNumber",
        "value": "INV-404",
        "value_as_text": "INV-404",
        "reason": "No OCR candidate passed text prefilters",
        "alternative_candidates": [],
    }


def test_seller_name_can_use_header_and_email_domain_evidence(make_word) -> None:
    image = Image.new("RGB", (1000, 500), "white")
    words = [
        make_word("PLASTICS", 0, 40, 30, 150, 60, width=1000, height=500),
        make_word("Email:", 1, 40, 130, 95, 150, line=1, width=1000, height=500),
        make_word("sales@plastics-express.co.uk", 2, 100, 130, 310, 150, line=1, width=1000, height=500),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"parties": {"seller": {"name": "PLASTICS EXPRESS"}}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "seller_name_header_or_email_partial"
    assert field.word_ids == [words[0].id, words[2].id]


def test_late_date_resolution_feeds_iso_inheritance(make_word) -> None:
    image = Image.new("RGB", (800, 300), "white")
    words = [
        make_word("Tax", 0, 20, 20, 50, 40, width=800, height=300),
        make_word("Point", 1, 55, 20, 100, 40, width=800, height=300),
        make_word("Date:", 2, 105, 20, 150, 40, width=800, height=300),
        make_word("13/10/2025", 3, 170, 20, 260, 40, width=800, height=300),
        make_word("Order", 4, 20, 70, 70, 90, line=1, width=800, height=300),
        make_word("Date:", 5, 75, 70, 120, 90, line=1, width=800, height=300),
        make_word("13/10/2025", 6, 170, 70, 260, 90, line=1, width=800, height=300),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"invoiceInfo": {"issueDate": "13/10/2025", "issueDateISO": "2025-10-13"}}},
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    issue_date = by_path["invoiceOutputData.invoiceInfo.issueDate"]
    iso = by_path["invoiceOutputData.invoiceInfo.issueDateISO"]
    assert issue_date.status == GroundingStatus.MATCHED
    assert issue_date.word_ids == [words[6].id]
    assert iso.status == GroundingStatus.INHERITED
    assert iso.word_ids == issue_date.word_ids


def test_invoice_info_prefers_invoice_date_over_booking_date(make_word) -> None:
    image = Image.new("RGB", (1000, 500), "white")
    words = [
        make_word("Invoice", 0, 50, 80, 120, 105, width=1000, height=500),
        make_word("Date:", 1, 125, 80, 180, 105, width=1000, height=500),
        make_word("16", 2, 220, 80, 250, 105, width=1000, height=500),
        make_word("Jul", 3, 255, 80, 290, 105, width=1000, height=500),
        make_word("2024", 4, 295, 80, 345, 105, width=1000, height=500),
        make_word("Booking", 5, 500, 80, 580, 105, width=1000, height=500),
        make_word("Date:", 6, 585, 80, 640, 105, width=1000, height=500),
        make_word("16", 7, 680, 80, 710, 105, width=1000, height=500),
        make_word("Jul", 8, 715, 80, 750, 105, width=1000, height=500),
        make_word("2024", 9, 755, 80, 805, 105, width=1000, height=500),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"invoiceInfo": {"issueDate": "16 Jul 2024", "issueDateISO": "2024-07-16"}}},
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    issue_date = by_path["invoiceOutputData.invoiceInfo.issueDate"]
    iso = by_path["invoiceOutputData.invoiceInfo.issueDateISO"]
    assert issue_date.status == GroundingStatus.MATCHED
    assert issue_date.word_ids == [words[2].id, words[3].id, words[4].id]
    assert iso.status == GroundingStatus.INHERITED
    assert iso.word_ids == issue_date.word_ids


def test_payment_terms_prefers_type_cash_over_trade_cash_sale(make_word) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    words = [
        make_word("TRADE", 0, 80, 100, 150, 125, width=1000, height=700),
        make_word("CASH", 1, 155, 100, 215, 125, width=1000, height=700),
        make_word("SALE", 2, 220, 100, 280, 125, width=1000, height=700),
        make_word("Type:", 3, 80, 560, 135, 585, line=1, width=1000, height=700),
        make_word("CASH", 4, 200, 560, 260, 585, line=1, width=1000, height=700),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"invoiceInfo": {"paymentTerms": "CASH"}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[4].id]
    assert "resolved_by_context" in (field.match_method or "")


def test_customer_memo_fallback_recovers_long_text_fragments(make_word) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    words = [
        make_word("In", 0, 80, 300, 105, 325, width=1000, height=700),
        make_word("the", 1, 110, 300, 145, 325, width=1000, height=700),
        make_word("case", 2, 150, 300, 200, 325, width=1000, height=700),
        make_word("where", 3, 205, 300, 260, 325, width=1000, height=700),
        make_word("goods", 4, 265, 300, 325, 325, width=1000, height=700),
        make_word("are", 5, 330, 300, 365, 325, width=1000, height=700),
        make_word("imported", 6, 370, 300, 455, 325, width=1000, height=700),
        make_word("outside", 7, 460, 300, 535, 325, width=1000, height=700),
        make_word("UK", 8, 540, 300, 570, 325, width=1000, height=700),
        make_word("reserve", 9, 80, 335, 155, 360, line=1, width=1000, height=700),
        make_word("right", 10, 160, 335, 210, 360, line=1, width=1000, height=700),
        make_word("adjust", 11, 215, 335, 275, 360, line=1, width=1000, height=700),
        make_word("lead", 12, 280, 335, 330, 360, line=1, width=1000, height=700),
        make_word("times", 13, 335, 335, 390, 360, line=1, width=1000, height=700),
        make_word("pricing", 14, 80, 370, 155, 395, line=2, width=1000, height=700),
        make_word("accordingly.", 15, 160, 370, 270, 395, line=2, width=1000, height=700),
        make_word("Courier:", 16, 80, 500, 160, 525, line=3, width=1000, height=700),
        make_word("DPD", 17, 165, 500, 205, 525, line=3, width=1000, height=700),
        make_word("NDD", 18, 230, 500, 270, 525, line=3, width=1000, height=700),
        make_word("WISTON", 19, 300, 500, 370, 525, line=3, width=1000, height=700),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "invoiceInfo": {
                    "customerMemo": (
                        "Courier: DPD - NDD - WISTON. In the case where goods are imported from "
                        "outside of the UK, we reserve the right to adjust lead times and pricing accordingly."
                    )
                }
            }
        },
        words,
        config=GroundingConfig(max_lines_per_candidate=1),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.match_method == "invoice_info_long_text_partial"
    assert words[16].id in field.word_ids
    assert words[6].id in field.word_ids


def test_party_duplicate_names_resolve_with_block_labels(make_word) -> None:
    image = Image.new("RGB", (1000, 500), "white")
    words = [
        make_word("Invoice", 0, 50, 90, 120, 115, width=1000, height=500),
        make_word("To:", 1, 125, 90, 155, 115, width=1000, height=500),
        make_word("Deliver", 2, 500, 90, 575, 115, width=1000, height=500),
        make_word("To:", 3, 580, 90, 610, 115, width=1000, height=500),
        make_word("TAXI", 4, 50, 130, 100, 155, line=1, width=1000, height=500),
        make_word("DRIVERS", 5, 105, 130, 180, 155, line=1, width=1000, height=500),
        make_word("TAXI", 6, 500, 130, 550, 155, line=2, width=1000, height=500),
        make_word("DRIVERS", 7, 555, 130, 630, 155, line=2, width=1000, height=500),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {"name": "TAXI DRIVERS"},
                    "shipTo": {"name": "TAXI DRIVERS"},
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    customer = by_path["invoiceOutputData.parties.customer.name"]
    ship_to = by_path["invoiceOutputData.parties.shipTo.name"]
    assert customer.status == GroundingStatus.MATCHED
    assert ship_to.status == GroundingStatus.MATCHED
    assert customer.word_ids == [words[4].id, words[5].id]
    assert ship_to.word_ids == [words[6].id, words[7].id]
    assert "resolved_by_context" in (customer.match_method or "")
    assert "resolved_by_context" in (ship_to.match_method or "")


def test_party_block_anchors_resolve_duplicate_contact_values(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("Bill", 0, 50, 80, 90, 105, width=1000, height=600),
        make_word("To:", 1, 95, 80, 125, 105, width=1000, height=600),
        make_word("Ship", 2, 500, 80, 545, 105, width=1000, height=600),
        make_word("To:", 3, 550, 80, 580, 105, width=1000, height=600),
        make_word("Customer", 4, 50, 130, 130, 155, line=1, width=1000, height=600),
        make_word("Road", 5, 135, 130, 180, 155, line=1, width=1000, height=600),
        make_word("Ship", 6, 500, 130, 540, 155, line=2, width=1000, height=600),
        make_word("Road", 7, 545, 130, 590, 155, line=2, width=1000, height=600),
        make_word("07789968406", 8, 50, 175, 160, 198, line=3, width=1000, height=600),
        make_word("07789968406", 9, 500, 175, 610, 198, line=4, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {
                        "addressStructured": {"address": "Customer Road"},
                        "phone": "07789968406",
                    },
                    "shipTo": {
                        "addressStructured": {"address": "Ship Road"},
                        "phone": "07789968406",
                    },
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    customer_phone = by_path["invoiceOutputData.parties.customer.phone"]
    ship_to_phone = by_path["invoiceOutputData.parties.shipTo.phone"]
    assert customer_phone.status == GroundingStatus.MATCHED
    assert ship_to_phone.status == GroundingStatus.MATCHED
    assert customer_phone.word_ids == [words[8].id]
    assert ship_to_phone.word_ids == [words[9].id]
    assert "resolved_by_context" in (customer_phone.match_method or "")
    assert "resolved_by_context" in (ship_to_phone.match_method or "")


def test_address_component_avoids_reusing_street_address_words(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("Berth", 0, 50, 120, 100, 145, width=1000, height=600),
        make_word("29,", 1, 105, 120, 135, 145, width=1000, height=600),
        make_word("Tilbury", 2, 140, 120, 205, 145, width=1000, height=600),
        make_word("Freeport", 3, 210, 120, 285, 145, width=1000, height=600),
        make_word("Tilbury", 4, 50, 170, 115, 195, line=1, width=1000, height=600),
        make_word("RM18", 5, 50, 220, 100, 245, line=2, width=1000, height=600),
        make_word("7EH", 6, 105, 220, 145, 245, line=2, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {
                        "addressStructured": {
                            "address": "Berth 29, Tilbury Freeport",
                            "city": "Tilbury",
                            "postal_code": "RM18 7EH",
                        }
                    }
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    city = by_path["invoiceOutputData.parties.customer.addressStructured.city"]
    assert city.status == GroundingStatus.MATCHED
    assert city.word_ids == [words[4].id]
    assert "resolved_by_context" in (city.match_method or "")


def test_customer_name_fallback_uses_invoice_to_multiline_label(make_word) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    words = [
        make_word("Invoice", 0, 80, 120, 150, 145, width=1000, height=700),
        make_word("To", 1, 155, 120, 185, 145, width=1000, height=700),
        make_word("Serena", 2, 80, 170, 150, 195, line=1, width=1000, height=700),
        make_word("Nice", 3, 80, 205, 130, 230, line=2, width=1000, height=700),
        make_word("Bites", 4, 135, 205, 190, 230, line=2, width=1000, height=700),
        make_word("Shipston", 5, 80, 245, 160, 270, line=3, width=1000, height=700),
        make_word("CV36", 6, 80, 280, 130, 305, line=4, width=1000, height=700),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {
                        "name": "Serena, Nice Bites Cafe",
                        "addressStructured": {"city": "Shipston", "postal_code": "CV36"},
                    }
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    field = by_path["invoiceOutputData.parties.customer.name"]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[2].id, words[3].id, words[4].id]
    assert field.match_method == "party_name_label_or_block_partial"


def test_shipto_name_fallback_prefers_despatch_to_over_invoice_to(make_word) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    words = [
        make_word("DESPATCHTO", 0, 100, 100, 210, 125, width=1000, height=700),
        make_word("DETAILED", 1, 230, 100, 310, 125, width=1000, height=700),
        make_word("PAINTWORK", 2, 315, 100, 420, 125, width=1000, height=700),
        make_word("DAVID", 3, 230, 135, 290, 160, line=1, width=1000, height=700),
        make_word("INVOICE", 4, 100, 260, 180, 285, line=2, width=1000, height=700),
        make_word("TO", 5, 185, 260, 215, 285, line=2, width=1000, height=700),
        make_word("DETAILED", 6, 230, 260, 310, 285, line=2, width=1000, height=700),
        make_word("PAINTWORK", 7, 315, 260, 420, 285, line=2, width=1000, height=700),
        make_word("DAVID", 8, 230, 295, 290, 320, line=3, width=1000, height=700),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {"name": "DETAILED PAINTWORK DAVID"},
                    "shipTo": {"name": "DETAILED PAINTWORK DAVID"},
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    customer = by_path["invoiceOutputData.parties.customer.name"]
    ship_to = by_path["invoiceOutputData.parties.shipTo.name"]
    assert customer.status == GroundingStatus.MATCHED
    assert ship_to.status == GroundingStatus.MATCHED
    assert customer.word_ids == [words[6].id, words[7].id, words[8].id]
    assert ship_to.word_ids == [words[1].id, words[2].id, words[3].id]


def test_customer_name_fallback_uses_sibling_address_block(make_word) -> None:
    image = Image.new("RGB", (1000, 700), "white")
    words = [
        make_word("SARAH", 0, 100, 100, 160, 125, width=1000, height=700),
        make_word("HUTCHINS", 1, 165, 100, 260, 125, width=1000, height=700),
        make_word("SPECIALITY", 2, 100, 150, 210, 175, line=1, width=1000, height=700),
        make_word("CAKES", 3, 215, 150, 280, 175, line=1, width=1000, height=700),
        make_word("HIGH", 4, 285, 150, 340, 175, line=1, width=1000, height=700),
        make_word("STREET", 5, 345, 150, 420, 175, line=1, width=1000, height=700),
        make_word("WOKING", 6, 100, 200, 180, 225, line=2, width=1000, height=700),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {
            "invoiceOutputData": {
                "parties": {
                    "customer": {
                        "name": "SPECIALITY CAKES, Sarah Hutchins",
                        "addressStructured": {"address": "HIGH STREET", "city": "WOKING"},
                    }
                }
            }
        },
        words,
        config=GroundingConfig(),
    )
    by_path = {field.json_path: field for field in result.fields}
    field = by_path["invoiceOutputData.parties.customer.name"]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[0].id, words[1].id, words[2].id, words[3].id]
    assert field.match_method == "party_name_label_or_block_partial"


def test_totals_summary_label_resolves_total_duplicate(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("TOTAL", 0, 500, 260, 560, 285, width=1000, height=600),
        make_word("GBP", 1, 565, 260, 610, 285, width=1000, height=600),
        make_word("56.84", 2, 700, 260, 760, 285, width=1000, height=600),
        make_word("Less", 3, 500, 305, 545, 330, line=1, width=1000, height=600),
        make_word("Amount", 4, 550, 305, 620, 330, line=1, width=1000, height=600),
        make_word("Paid", 5, 625, 305, 670, 330, line=1, width=1000, height=600),
        make_word("56.84", 6, 700, 305, 760, 330, line=1, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"totalIncludingTax": {"originalValue": "56.84"}}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[2].id]
    assert "resolved_by_context" in (field.match_method or "")


def test_totals_summary_label_resolves_tax_name_over_footer_vat(make_word) -> None:
    image = Image.new("RGB", (1000, 800), "white")
    words = [
        make_word("Discount", 0, 500, 420, 585, 445, width=1000, height=800),
        make_word("50.00", 1, 700, 420, 760, 445, width=1000, height=800),
        make_word("GBP", 2, 765, 420, 810, 445, width=1000, height=800),
        make_word("VAT", 3, 500, 455, 540, 480, line=1, width=1000, height=800),
        make_word("134.41", 4, 700, 455, 770, 480, line=1, width=1000, height=800),
        make_word("GBP", 5, 775, 455, 820, 480, line=1, width=1000, height=800),
        make_word("Registered", 6, 500, 720, 600, 745, line=2, width=1000, height=800),
        make_word("VAT", 7, 605, 720, 645, 745, line=2, width=1000, height=800),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"taxName": "VAT"}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[3].id]
    assert "resolved_by_context" in (field.match_method or "")


def test_totals_row_pair_resolves_total_excluding_tax(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("Product", 0, 80, 180, 140, 205, width=1000, height=600),
        make_word("No.", 1, 145, 180, 180, 205, width=1000, height=600),
        make_word("Description", 2, 185, 180, 280, 205, width=1000, height=600),
        make_word("Quantity", 3, 285, 180, 360, 205, width=1000, height=600),
        make_word("AIR", 4, 80, 225, 115, 250, line=1, width=1000, height=600),
        make_word("CON", 5, 120, 225, 155, 250, line=1, width=1000, height=600),
        make_word("1", 6, 285, 225, 300, 250, line=1, width=1000, height=600),
        make_word("£62.50", 7, 700, 225, 770, 250, line=1, width=1000, height=600),
        make_word("Products", 8, 500, 270, 590, 295, line=2, width=1000, height=600),
        make_word("Total", 9, 595, 270, 645, 295, line=2, width=1000, height=600),
        make_word("Ex", 10, 650, 270, 675, 295, line=2, width=1000, height=600),
        make_word("Vat:", 11, 680, 270, 720, 295, line=2, width=1000, height=600),
        make_word("£62.50", 12, 760, 270, 830, 295, line=2, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"totalExcludingTax": {"originalValue": "£62.50"}}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[12].id]
    assert "resolved_by_context" in (field.match_method or "")


def test_totals_row_pair_resolves_shipping_charge_value(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("ASIN:", 0, 100, 250, 155, 275, width=1000, height=600),
        make_word("B0862838MV", 1, 160, 250, 270, 275, width=1000, height=600),
        make_word("Shipping", 2, 500, 250, 585, 275, width=1000, height=600),
        make_word("Charges", 3, 590, 250, 665, 275, width=1000, height=600),
        make_word("£0.00", 4, 730, 250, 790, 275, width=1000, height=600),
        make_word("£0.00", 5, 820, 250, 880, 275, width=1000, height=600),
        make_word("Total", 6, 500, 310, 550, 335, line=1, width=1000, height=600),
        make_word("£32.99", 7, 730, 310, 800, 335, line=1, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"otherCharges": [{"value": {"originalValue": "£0.00"}}]}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[4].id]
    assert "resolved_by_context" in (field.match_method or "")


def test_numeric_total_prefers_clean_amount_over_label_span(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("Vat", 0, 500, 250, 535, 275, width=1000, height=600),
        make_word("Plastic", 1, 540, 250, 600, 275, width=1000, height=600),
        make_word("4.83", 2, 730, 250, 790, 275, width=1000, height=600),
        make_word("4.83", 3, 300, 310, 360, 335, line=1, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"taxAmount": {"originalValue": "4.83"}}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[2].id]
    assert field.matched_text == "4.83"
    assert "resolved_by_context" in (field.match_method or "")


def test_totals_row_pair_resolves_other_charge_key_label(make_word) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    words = [
        make_word("Delivery", 0, 80, 90, 150, 115, width=1000, height=600),
        make_word("Address", 1, 155, 90, 230, 115, width=1000, height=600),
        make_word("Delivery:", 2, 500, 360, 585, 385, line=1, width=1000, height=600),
        make_word("£12.99", 3, 700, 360, 770, 385, line=1, width=1000, height=600),
        make_word("courier", 4, 80, 500, 145, 525, line=2, width=1000, height=600),
        make_word("delivery", 5, 150, 500, 220, 525, line=2, width=1000, height=600),
    ]
    result = ground_invoice_values_from_ocr(
        image,
        {"invoiceOutputData": {"totals": {"otherCharges": [{"key": "Delivery"}]}}},
        words,
        config=GroundingConfig(),
    )
    field = result.fields[0]
    assert field.status == GroundingStatus.MATCHED
    assert field.word_ids == [words[2].id]
    assert "resolved_by_context" in (field.match_method or "")


@pytest.mark.integration
def test_doctr_integration_synthetic_invoice(tmp_path) -> None:
    if os.environ.get("RUN_DOCTR_INTEGRATION") != "1":
        pytest.skip("Set RUN_DOCTR_INTEGRATION=1 to run docTR integration test")
    image_path = tmp_path / "invoice.png"
    img = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Invoice Number INV-10042", fill="black")
    draw.text((50, 100), "Grand Total $123.45", fill="black")
    img.save(image_path)
    result = ground_invoice_values(
        image_path,
        {
            "invoiceOutputData": {
                "invoiceInfo": {"documentNumber": "INV-10042"},
                "totals": {"totalIncludingTax": {"originalValue": "$123.45"}},
            }
        },
        config=GroundingConfig(device="cpu", min_confidence=0.6),
    )
    assert result.ocr_words
    assert any(field.status in {GroundingStatus.MATCHED, GroundingStatus.AMBIGUOUS} for field in result.fields)
