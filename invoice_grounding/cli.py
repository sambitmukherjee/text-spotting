from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from invoice_grounding.grounding import ground_invoice_values
from invoice_grounding.models import GroundingConfig, InvoiceGroundingError
from invoice_grounding.visualization import render_grounding_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Ground structured invoice extraction values to OCR word boxes.")
    parser.add_argument("--image", required=True, help="Invoice or receipt image path")
    parser.add_argument("--extraction", required=True, help="Extraction JSON path")
    parser.add_argument("--output-json", required=True, help="Output grounding JSON path")
    parser.add_argument("--overlay", help="Optional overlay image path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--include-ocr-words", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = GroundingConfig(
        device=args.device,
        min_confidence=args.min_confidence,
        include_ocr_words=args.include_ocr_words,
        debug=args.debug,
        preprocess=args.preprocess,
    )
    try:
        result = ground_invoice_values(args.image, args.extraction, config=config)
    except InvoiceGroundingError as exc:
        logging.error("%s", exc)
        return 2

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if args.overlay:
        render_grounding_overlay(args.image, result, args.overlay)

    counts = Counter(field.status.value for field in result.fields)
    summary = {"fields": len(result.fields), **dict(sorted(counts.items()))}
    logging.info("Grounding complete: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
