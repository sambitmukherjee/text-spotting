from __future__ import annotations

from typing import Any

from invoice_grounding.models import GroundableValue


PAYLOAD_KEYS = ("invoiceOutputData", "postprocessed", "data")
SPLIT_PAYLOAD_KEYS = ("invoiceInfo", "parties", "totals", "shippingInfo", "currency", "invoiceStatus")
SKIPPED_CONTAINER_KEYS = {
    "lineItems",
    "annotation_meta",
    "metadata",
    "field_metadata",
    "document_id",
}

NON_GROUNDABLE_FIELD_NAMES = {
    "documentType",
    "documentTypeConfidence",
    "documentTypeAlternatives",
    "language",
    "sourceCountryIfObvious",
    "category",
    "categoryReasoning",
    "invoiceStatus",
    "isOverflowPage",
    "isOverflowPageReasoning",
    "applyTaxAfterDiscount",
    "confidence",
}

ISO_DATE_SOURCE_NAMES = {
    "issueDateISO": "issueDate",
    "dueDateISO": "dueDate",
    "serviceDateISO": "serviceDate",
    "deliveryDateISO": "deliveryDate",
    "extractionDateISO": "extractionDate",
}


def extract_groundable_values(extraction: dict[str, Any]) -> list[GroundableValue]:
    values: list[GroundableValue] = []
    payload, prefix = _select_invoice_payload(extraction)
    _walk(payload, prefix, values)
    return values


def _select_invoice_payload(extraction: dict[str, Any]) -> tuple[Any, str]:
    for key in PAYLOAD_KEYS:
        payload = extraction.get(key)
        if isinstance(payload, dict) and payload:
            return payload, key
    if any(key in extraction for key in SPLIT_PAYLOAD_KEYS):
        return extraction, ""
    return extraction, ""


def _walk(value: Any, path: str, values: list[GroundableValue]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        if "originalValue" in value and value.get("originalValue") is not None:
            child_path = _join(path, "originalValue")
            values.append(_leaf(child_path, value["originalValue"]))
            return
        for key, child in value.items():
            if key in SKIPPED_CONTAINER_KEYS:
                continue
            _walk(child, _join(path, str(key)), values)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", values)
        return
    values.append(_leaf(path, value))


def _leaf(path: str, value: Any) -> GroundableValue:
    field_name = _field_name(path)
    if isinstance(value, bool):
        return GroundableValue(
            json_path=path,
            field_name=field_name,
            value=value,
            value_as_text=str(value),
            groundable=False,
            reason="Boolean fields are semantic flags and are not directly groundable",
        )
    if field_name in ISO_DATE_SOURCE_NAMES:
        source_name = ISO_DATE_SOURCE_NAMES[field_name]
        return GroundableValue(
            json_path=path,
            field_name=field_name,
            value=value,
            value_as_text=str(value),
            groundable=True,
            inherited_from=_sibling_path(path, source_name),
        )
    if field_name in NON_GROUNDABLE_FIELD_NAMES or field_name.endswith("Reasoning"):
        return GroundableValue(
            json_path=path,
            field_name=field_name,
            value=value,
            value_as_text=str(value),
            groundable=False,
            reason=f"{field_name} is inferred, classified, or diagnostic metadata",
        )
    if isinstance(value, (str, int, float)):
        text = str(value)
        return GroundableValue(
            json_path=path,
            field_name=_parent_field_name(path) if field_name == "originalValue" else field_name,
            value=value,
            value_as_text=text,
            groundable=bool(text.strip()),
        )
    return GroundableValue(
        json_path=path,
        field_name=field_name,
        value=value,
        value_as_text=str(value),
        groundable=False,
        reason=f"Values of type {type(value).__name__} are not directly groundable",
    )


def _join(path: str, key: str) -> str:
    return key if not path else f"{path}.{key}"


def _field_name(path: str) -> str:
    tail = path.split(".")[-1]
    if "[" in tail:
        tail = tail.split("[", 1)[0]
    return tail


def _parent_field_name(path: str) -> str:
    pieces = path.split(".")
    if len(pieces) >= 2:
        return _field_name(pieces[-2])
    return _field_name(path)


def _sibling_path(path: str, sibling_name: str) -> str:
    pieces = path.split(".")
    pieces[-1] = sibling_name
    return ".".join(pieces)
