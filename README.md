# Invoice Grounding

Ground structured invoice extraction values to the OCR words and bounding boxes that support them.

This prototype does not extract invoice fields from scratch. It accepts an invoice or receipt image plus an existing extraction JSON/dict, runs docTR OCR once, then deterministically aligns printable non-table leaf values to OCR word spans.

Table line items are intentionally out of scope for this first pass. Any `lineItems` container is skipped entirely, including descriptions, item codes, quantities, unit prices, and line totals.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The required OCR package is `python-doctr`. The first OCR invocation may download pretrained docTR model weights. Those weights are cache artifacts and should not be committed.

The package targets Python 3.10 or newer and runs on CPU. Use `--device cuda` or `GroundingConfig(device="cuda")` when CUDA is available; `auto` will try CUDA when PyTorch reports it is available.

## CLI

```bash
python -m invoice_grounding.cli \
  --image examples/invoice.png \
  --extraction examples/invoice.json \
  --output-json output/grounding.json \
  --overlay output/overlay.png
```

Useful options:

```bash
--device auto|cpu|cuda
--min-confidence 0.72
--include-ocr-words
--debug
--preprocess
```

Unmatched fields are expected and do not make the CLI fail. The CLI returns nonzero only for fatal input or OCR errors.

## Python Usage

```python
from invoice_grounding import ground_invoice_values
from invoice_grounding.visualization import render_grounding_overlay

result = ground_invoice_values(
    image="invoice.png",
    extraction="invoice.json",
)

with open("grounding.json", "w", encoding="utf-8") as f:
    f.write(result.model_dump_json(indent=2))

render_grounding_overlay(
    image="invoice.png",
    result=result,
    output_path="overlay.png",
)
```

## Output

`GroundingResult` includes:

- image width and height;
- OCR engine/model metadata;
- timing metadata;
- one `GroundedField` per meaningful non-table extraction leaf;
- optional OCR words with IDs, confidence, reading order, normalized boxes, and pixel boxes;
- warnings.

Field statuses are:

- `matched`: a supported OCR span was found.
- `inherited`: a derived field, such as `issueDateISO`, copied evidence from its printed source.
- `ambiguous`: multiple plausible OCR spans remained.
- `unmatched`: the value should be printable, but no candidate passed the threshold.
- `not_groundable`: the field is classified, inferred, boolean, reasoning, or metadata.
- `error`: field-specific matching failed without aborting the page.

Pixel boxes use the EXIF-orientation-corrected image coordinate system. `x_max` and `y_max` are consistently treated as the right/bottom edge used by PIL drawing. Normalized boxes are in `[0, 1]`.

## Matching Stages

The matcher ranks bounded OCR candidates generated from single words, same-line spans, and adjacent-line spans:

1. Exact raw text after Unicode and whitespace normalization.
2. Exact compact text for OCR-split identifiers such as `INV - 10042`.
3. Field-aware canonical matching for amounts, dates, phones, and emails.
4. Fuzzy matching using RapidFuzz when available.
5. Contextual reranking with label hints such as `Grand Total`, `Invoice #`, `Bill To`, and `Tracking No`.

Repeated values with insufficient separation are marked `ambiguous`; the implementation does not choose the first repeated amount blindly.

## Limitations

This is a reliable baseline for invoice headers, parties, addresses, shipping fields, dates, and totals. It deliberately does not ground table line items, because table row and column reasoning is a separate harder problem. Repeated totals or repeated scalar values and weak OCR may still produce `ambiguous` or `unmatched` results. Derived and inferred fields are not assigned fabricated boxes.

The `currency` field can match common printed symbols such as `£`, `$`, and `€` to normalized ISO codes such as `GBP`, `USD`, and `EUR`. Disable this with `GroundingConfig(enable_currency_symbol_mapping=False)` if you only want literal ISO-code matches.

## Tests

Fast unit tests do not run docTR and do not download weights:

```bash
pytest
```

The optional docTR integration test creates a synthetic invoice image and runs OCR:

```bash
RUN_DOCTR_INTEGRATION=1 pytest -m integration
```

## Overlay Images

Use `render_grounding_overlay(...)` or pass `--overlay` to the CLI. Matched, inherited, and ambiguous fields are drawn with distinct colors. Labels are shortened on the image and expanded in a legend to avoid long paths covering the invoice.
