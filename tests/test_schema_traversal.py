from __future__ import annotations

from invoice_grounding.schema_traversal import extract_groundable_values


def test_line_items_are_skipped() -> None:
    extraction = {
        "invoiceOutputData": {
            "invoiceInfo": {"documentNumber": "INV-1"},
            "lineItems": [
                {"description": "Widget", "quantity": {"originalValue": "2", "value": 2}},
                {"unitPrice": {"originalValue": "$10.00"}},
            ]
        }
    }
    paths = [value.json_path for value in extract_groundable_values(extraction)]
    assert paths == ["invoiceOutputData.invoiceInfo.documentNumber"]
    assert "invoiceOutputData.lineItems[0].description" not in paths
    assert "invoiceOutputData.lineItems[0].quantity.originalValue" not in paths
    assert "invoiceOutputData.lineItems[0].quantity.value" not in paths


def test_postprocessed_payload_ignores_metadata() -> None:
    values = extract_groundable_values(
        {
            "document_id": "D001",
            "metadata": {"language": "English"},
            "annotation_meta": {"field_metadata": [{"name": "debug", "value": "ignore me"}]},
            "postprocessed": {
                "invoiceInfo": {"documentNumber": "INV-1"},
                "totals": {"totalIncludingTax": {"originalValue": "$10.00"}},
                "lineItems": [{"description": "Ignored table row"}],
            },
        }
    )
    paths = [value.json_path for value in values]
    assert paths == [
        "postprocessed.invoiceInfo.documentNumber",
        "postprocessed.totals.totalIncludingTax.originalValue",
    ]


def test_empty_postprocessed_falls_back_to_data_payload() -> None:
    values = extract_groundable_values(
        {
            "postprocessed": {},
            "data": {
                "invoiceInfo": {"documentNumber": "INV-DATA"},
                "totals": {"totalIncludingTax": {"originalValue": "$12.34"}},
                "lineItems": [{"description": "Ignored row"}],
            },
        }
    )
    paths = [value.json_path for value in values]
    assert paths == [
        "data.invoiceInfo.documentNumber",
        "data.totals.totalIncludingTax.originalValue",
    ]


def test_iso_date_is_marked_for_inheritance() -> None:
    values = extract_groundable_values(
        {"invoiceOutputData": {"invoiceInfo": {"issueDate": "06/01/2026", "issueDateISO": "2026-06-01"}}}
    )
    iso = next(value for value in values if value.json_path.endswith("issueDateISO"))
    assert iso.inherited_from == "invoiceOutputData.invoiceInfo.issueDate"


def test_non_groundable_fields_are_detected() -> None:
    values = extract_groundable_values(
        {"invoiceOutputData": {"invoiceStatus": "unpaid", "categoryReasoning": "Looks like office supplies"}}
    )
    assert all(not value.groundable for value in values)
