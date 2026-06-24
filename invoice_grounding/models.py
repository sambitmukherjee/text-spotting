from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


if hasattr(BaseModel, "model_dump"):
    CompatBaseModel = BaseModel
else:

    class CompatBaseModel(BaseModel):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.dict(*args, **kwargs)

        def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
            return self.json(*args, **kwargs)


class GroundingStatus(str, Enum):
    MATCHED = "matched"
    INHERITED = "inherited"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    NOT_GROUNDABLE = "not_groundable"
    ERROR = "error"


class BoundingBox(CompatBaseModel):
    model_config = ConfigDict(frozen=True)

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class NormalizedBoundingBox(CompatBaseModel):
    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class OCRWord(CompatBaseModel):
    id: str
    text: str
    confidence: float = 0.0
    page_index: int = 0
    block_index: int = 0
    line_index: int = 0
    word_index: int = 0
    reading_order: int = 0
    bbox_normalized: tuple[float, float, float, float]
    bbox_pixels: tuple[int, int, int, int]
    polygon_normalized: list[tuple[float, float]] | None = None
    polygon_pixels: list[tuple[int, int]] | None = None

    @property
    def line_id(self) -> str:
        return f"p{self.page_index}-b{self.block_index}-l{self.line_index}"

    @property
    def box_pixels(self) -> BoundingBox:
        return BoundingBox(
            x_min=self.bbox_pixels[0],
            y_min=self.bbox_pixels[1],
            x_max=self.bbox_pixels[2],
            y_max=self.bbox_pixels[3],
        )

    @property
    def box_normalized(self) -> NormalizedBoundingBox:
        return NormalizedBoundingBox(
            x_min=self.bbox_normalized[0],
            y_min=self.bbox_normalized[1],
            x_max=self.bbox_normalized[2],
            y_max=self.bbox_normalized[3],
        )


class CandidateSpan(CompatBaseModel):
    id: str
    text: str
    compact_text: str
    words: list[OCRWord]
    word_ids: list[str]
    line_ids: list[str]
    bbox_pixels: BoundingBox
    bbox_normalized: NormalizedBoundingBox
    mean_ocr_confidence: float


class CandidateSummary(CompatBaseModel):
    matched_text: str
    confidence: float
    word_ids: list[str]
    match_method: str | None = None


class ScoreBreakdown(CompatBaseModel):
    text_score: float
    ocr_score: float
    context_score: float
    spatial_score: float
    ambiguity_margin: float | None
    final_score: float


class GroundableValue(CompatBaseModel):
    json_path: str
    field_name: str
    value: Any
    value_as_text: str | None
    groundable: bool
    reason: str | None = None
    inherited_from: str | None = None


class GroundedField(CompatBaseModel):
    json_path: str
    field_name: str
    value: Any
    value_as_text: str | None
    status: GroundingStatus
    match_method: str | None = None
    confidence: float | None = None
    score_breakdown: ScoreBreakdown | None = None
    matched_text: str | None = None
    normalized_target: str | None = None
    normalized_matched_text: str | None = None
    word_ids: list[str] = Field(default_factory=list)
    word_boxes_pixels: list[BoundingBox] = Field(default_factory=list)
    word_boxes_normalized: list[NormalizedBoundingBox] = Field(default_factory=list)
    line_boxes_pixels: list[BoundingBox] = Field(default_factory=list)
    union_box_pixels: BoundingBox | None = None
    union_box_normalized: NormalizedBoundingBox | None = None
    candidate_rank: int | None = None
    alternative_candidates: list[CandidateSummary] = Field(default_factory=list)
    inherited_from: str | None = None
    reason: str | None = None


class GroundingResult(CompatBaseModel):
    image_width: int
    image_height: int
    page_count: int = 1
    ocr_engine: str = "docTR"
    ocr_model_info: dict[str, Any] = Field(default_factory=dict)
    fields: list[GroundedField]
    ambiguous: list[dict[str, Any]] = Field(default_factory=list)
    unmatched: list[dict[str, Any]] = Field(default_factory=list)
    ocr_words: list[OCRWord]
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class GroundingConfig(CompatBaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    min_confidence: float = 0.72
    ambiguity_margin: float = 0.08
    max_words_per_candidate: int = 20
    max_lines_per_candidate: int = 5
    include_ocr_words: bool = True
    preprocess: bool = False
    upscale_small_images: bool = False
    enable_currency_symbol_mapping: bool = True
    debug: bool = False
    ocr_model_info: dict[str, Any] = Field(default_factory=dict)


class InvoiceGroundingError(Exception):
    """Base exception for fatal invoice grounding failures."""


class InputLoadError(InvoiceGroundingError):
    """Raised when an image or extraction input cannot be loaded."""


class OCRError(InvoiceGroundingError):
    """Raised when OCR initialization or inference fails."""


ImageInput = str | Path | Any
ExtractionInput = str | Path | dict[str, Any]
