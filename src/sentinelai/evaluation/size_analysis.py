"""Object-size stratification and scale-dependent detection error analysis.

Grounded Protocol Definition:
----------------------------
In remote sensing imagery, fixed-size physical objects appear at varying pixel footprints
depending on sensor ground sampling distance (GSD). We define two configurable protocols:
1. Relative Area Protocol (A_rel = box_area / image_area):
   - Small: A_rel <= 0.0016 (equivalent to <= 32^2 on 800x800 image)
   - Medium: 0.0016 < A_rel <= 0.0144 (equivalent to 32^2 < area <= 96^2)
   - Large: A_rel > 0.0144 (equivalent to area > 96^2)
2. Absolute Pixel Area Protocol:
   - Small: area <= 1024 px^2 (32^2)
   - Medium: 1024 < area <= 9216 px^2 (96^2)
   - Large: area > 9216 px^2
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.evaluation.metrics import evaluate_detections_at_iou, compute_ap
from sentinelai.postprocessing.fusion import compute_iou_matrix


class ObjectSizeAnalyzer:
    """Stratifies object detection metrics and diagnostic errors across object size bins."""

    DEFAULT_SIZE_BINS = {
        "small": (0.0, 32.0 ** 2),
        "medium": (32.0 ** 2, 96.0 ** 2),
        "large": (96.0 ** 2, float("inf")),
    }

    DEFAULT_RELATIVE_BINS = {
        "small": (0.0, 0.0016),
        "medium": (0.0016, 0.0144),
        "large": (0.0144, float("inf")),
    }

    def __init__(
        self,
        use_relative_area: bool = True,
        custom_bins: Optional[Dict[str, Tuple[float, float]]] = None,
        iou_threshold: float = 0.50,
    ):
        """Initializes the object size analyzer.

        Args:
            use_relative_area: If True, uses normalized relative area (box_area / img_area).
                               If False, uses absolute pixel area.
            custom_bins: Optional dictionary mapping bin name to (min_area, max_area).
            iou_threshold: IoU matching threshold for TP/FP assignment.
        """
        self.use_relative_area = use_relative_area
        self.iou_threshold = iou_threshold

        if custom_bins is not None:
            self.size_bins = custom_bins
        else:
            self.size_bins = self.DEFAULT_RELATIVE_BINS if use_relative_area else self.DEFAULT_SIZE_BINS

    def get_box_area_metric(self, box: BoundingBox, img_w: int, img_h: int) -> float:
        """Returns either relative area ratio or absolute pixel area."""
        pixel_area = box.area
        if self.use_relative_area:
            img_area = max(1.0, float(img_w * img_h))
            return pixel_area / img_area
        return pixel_area

    def categorize_box(self, box: BoundingBox, img_w: int, img_h: int) -> str:
        """Categorizes a box into one of the configured size bins."""
        val = self.get_box_area_metric(box, img_w, img_h)
        for bin_name, (min_v, max_v) in self.size_bins.items():
            if min_v <= val < max_v:
                return bin_name
        return "large"

    def analyze(self, results: List[ImageDetectionResult]) -> Dict[str, Any]:
        """Performs full size-stratified evaluation:

        - Object counts (ground truth vs predicted)
        - Recall per size bin
        - Precision per size bin
        - AP@0.5 per size bin
        - Error distributions (localization, class confusion, duplicate, background FPs, misses).
        """
        report: Dict[str, Any] = {
            "use_relative_area": self.use_relative_area,
            "bins": {k: [v[0], v[1]] for k, v in self.size_bins.items()},
            "metrics_per_bin": {},
        }

        # Index ground truths and predictions by size bin
        for bin_name in self.size_bins:
            bin_gts_count = 0
            bin_preds_count = 0

            # Filtered detection results containing only current size bin
            bin_results: List[ImageDetectionResult] = []

            for r in results:
                w, h = r.width, r.height
                filtered_gts = [g for g in r.ground_truths if self.categorize_box(g, w, h) == bin_name]
                filtered_preds = [p for p in r.predictions if self.categorize_box(p, w, h) == bin_name]

                bin_gts_count += len(filtered_gts)
                bin_preds_count += len(filtered_preds)

                bin_results.append(ImageDetectionResult(
                    image_id=r.image_id,
                    width=w,
                    height=h,
                    ground_truths=filtered_gts,
                    predictions=filtered_preds,
                ))

            # Evaluate detection performance for this bin
            eval_metrics = evaluate_detections_at_iou(bin_results, iou_threshold=self.iou_threshold)

            # Granular error breakdown for this size bin
            error_stats = self._breakdown_bin_errors(bin_results)

            report["metrics_per_bin"][bin_name] = {
                "ground_truth_count": bin_gts_count,
                "prediction_count": bin_preds_count,
                "true_positives": eval_metrics["tp"],
                "false_positives": eval_metrics["fp"],
                "precision": round(eval_metrics["precision"], 4),
                "recall": round(eval_metrics["recall"], 4),
                "mAP50": round(eval_metrics["ap"], 4),
                "error_breakdown": error_stats,
            }

        return report

    def _breakdown_bin_errors(self, bin_results: List[ImageDetectionResult]) -> Dict[str, int]:
        """Categorizes prediction errors specifically within a size bin."""
        loc_errors = 0
        confusions = 0
        duplicate_fps = 0
        background_fps = 0
        missed_gts = 0

        for r in bin_results:
            gts = r.ground_truths
            preds = r.predictions

            if not gts:
                background_fps += len(preds)
                continue

            gt_matched = [False] * len(gts)
            gt_coords = np.array([g.as_xyxy() for g in gts], dtype=np.float32)

            for p in sorted(preds, key=lambda x: x.score, reverse=True):
                p_coord = np.array([p.as_xyxy()], dtype=np.float32)
                ious = compute_iou_matrix(p_coord, gt_coords)[0]
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]
                best_gt = gts[best_idx]

                if best_iou >= self.iou_threshold:
                    if p.class_id == best_gt.class_id:
                        if not gt_matched[best_idx]:
                            gt_matched[best_idx] = True
                        else:
                            duplicate_fps += 1
                    else:
                        confusions += 1
                elif best_iou >= 0.10:
                    if p.class_id == best_gt.class_id:
                        loc_errors += 1
                    else:
                        confusions += 1
                else:
                    background_fps += 1

            missed_gts += sum(1 for m in gt_matched if not m)

        return {
            "localization_errors": loc_errors,
            "class_confusions": confusions,
            "duplicate_fps": duplicate_fps,
            "background_fps": background_fps,
            "missed_gts": missed_gts,
        }
