"""Data models for bounding boxes, annotations, predictions, and evaluation records."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates [xmin, ymin, xmax, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: int
    score: float = 1.0
    class_name: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def as_xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def as_normalized_xyxy(self, img_w: int, img_h: int) -> List[float]:
        return [
            self.xmin / img_w,
            self.ymin / img_h,
            self.xmax / img_w,
            self.ymax / img_h
        ]

    def clip(self, img_w: int, img_h: int) -> "BoundingBox":
        return BoundingBox(
            xmin=max(0.0, min(float(img_w), self.xmin)),
            ymin=max(0.0, min(float(img_h), self.ymin)),
            xmax=max(0.0, min(float(img_w), self.xmax)),
            ymax=max(0.0, min(float(img_h), self.ymax)),
            class_id=self.class_id,
            score=self.score,
            class_name=self.class_name
        )


@dataclass
class ImageDetectionResult:
    """Predictions and optional ground truths for a single image."""
    image_id: str
    image_path: Optional[str] = None
    width: int = 800
    height: int = 800
    predictions: List[BoundingBox] = field(default_factory=list)
    ground_truths: List[BoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TileCrop:
    """A tile crop region defined in full-image coordinates."""
    tile_id: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin
