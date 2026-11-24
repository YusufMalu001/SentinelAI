"""Object-density stratification analysis for aerial imagery.

Density Protocol:
-----------------
Aerial scenes vary drastically from isolated rural targets to congested urban/harbor
scenes. Images are grouped by ground-truth instance count per image using configurable
thresholds:
- Default Protocol (configurable):
  - Sparse:   0 <= count <= 3 objects/image
  - Moderate: 4 <= count <= 10 objects/image
  - Dense:    count >= 11 objects/image
Note: These thresholds are dataset-specific operational definitions for controlled
comparison, not universal physical constants.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from sentinelai.dataset.structures import ImageDetectionResult
from sentinelai.evaluation.metrics import evaluate_detections_at_iou


class ObjectDensityAnalyzer:
    """Stratifies detection performance across sparse, moderate, and dense scenes."""

    DEFAULT_DENSITY_BINS: Dict[str, Tuple[int, int]] = {
        "sparse": (0, 3),
        "moderate": (4, 10),
        "dense": (11, 10**6),
    }

    def __init__(
        self,
        density_bins: Optional[Dict[str, Tuple[int, int]]] = None,
        iou_threshold: float = 0.50,
    ):
        self.density_bins = density_bins or self.DEFAULT_DENSITY_BINS
        self.iou_threshold = iou_threshold

    def categorize_image(self, result: ImageDetectionResult) -> str:
        """Assigns an image to a density bin based on ground-truth object count."""
        gt_count = len(result.ground_truths)
        for bin_name, (min_c, max_c) in self.density_bins.items():
            if min_c <= gt_count <= max_c:
                return bin_name
        return "dense"

    def analyze(self, results: List[ImageDetectionResult]) -> Dict[str, Any]:
        """Evaluates precision, recall, F1, and mAP@0.5 across density categories."""
        grouped: Dict[str, List[ImageDetectionResult]] = {k: [] for k in self.density_bins}
        for r in results:
            cat = self.categorize_image(r)
            grouped[cat].append(r)

        report: Dict[str, Any] = {
            "density_protocol": {k: [v[0], v[1]] for k, v in self.density_bins.items()},
            "categories": {},
        }

        for cat_name, cat_results in grouped.items():
            if not cat_results:
                report["categories"][cat_name] = {
                    "image_count": 0,
                    "gt_count": 0,
                    "pred_count": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "mAP50": 0.0,
                }
                continue

            ev = evaluate_detections_at_iou(cat_results, iou_threshold=self.iou_threshold)
            p = ev["precision"]
            r = ev["recall"]
            f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

            report["categories"][cat_name] = {
                "image_count": len(cat_results),
                "gt_count": sum(len(x.ground_truths) for x in cat_results),
                "pred_count": sum(len(x.predictions) for x in cat_results),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50": round(ev["ap"], 4),
            }

        return report
