"""Detection matching, error categorization, boundary analysis, and IoU sensitivity.

Matching Strategy:
------------------
Predictions are sorted descending by confidence score and matched greedily against
ground-truth boxes using pairwise IoU:
- True Positive (TP): IoU >= iou_threshold, matching class_id, GT not yet matched.
- Duplicate FP: IoU >= iou_threshold, matching class_id, GT already matched.
- Localization Error: bg_iou_threshold <= IoU < iou_threshold, matching class_id.
- Class Confusion: IoU >= bg_iou_threshold, wrong class_id.
- Background FP: IoU < bg_iou_threshold with all ground-truth boxes.
- False Negative (FN): Ground-truth box not matched by any TP prediction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult, TileCrop
from sentinelai.dataset.constants import CLASS_ID_TO_NAME, SIZE_BINS
from sentinelai.postprocessing.fusion import compute_iou_matrix


@dataclass
class ImageMatchResult:
    """Detailed per-image matching output for diagnostics and visualization."""
    image_id: str
    true_positives: List[BoundingBox] = field(default_factory=list)
    localization_errors: List[BoundingBox] = field(default_factory=list)
    class_confusions: List[Tuple[BoundingBox, int]] = field(default_factory=list)
    duplicate_fps: List[BoundingBox] = field(default_factory=list)
    background_fps: List[BoundingBox] = field(default_factory=list)
    false_negatives: List[BoundingBox] = field(default_factory=list)

    @property
    def all_false_positives(self) -> List[BoundingBox]:
        return (
            self.localization_errors
            + [b for b, _ in self.class_confusions]
            + self.duplicate_fps
            + self.background_fps
        )


class DetectionErrorAnalyzer:
    """Performs class-aware detection matching and diagnostic error categorization."""

    def __init__(
        self,
        iou_threshold: float = 0.50,
        bg_iou_threshold: float = 0.10,
        boundary_margin: int = 32,
    ):
        self.iou_threshold = iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.boundary_margin = boundary_margin

    def is_near_boundary(self, box: BoundingBox, img_w: int, img_h: int) -> bool:
        """Checks if box is within boundary_margin pixels of the original image edges."""
        return (
            box.xmin <= self.boundary_margin
            or box.ymin <= self.boundary_margin
            or box.xmax >= img_w - self.boundary_margin
            or box.ymax >= img_h - self.boundary_margin
        )

    def is_near_tile_boundary(self, box: BoundingBox, tiles: List[TileCrop]) -> bool:
        """Checks if box is within boundary_margin of any internal tile seam."""
        for t in tiles:
            # Check if box intersects or is close to tile edges
            if (
                abs(box.xmin - t.xmin) <= self.boundary_margin
                or abs(box.xmax - t.xmax) <= self.boundary_margin
                or abs(box.ymin - t.ymin) <= self.boundary_margin
                or abs(box.ymax - t.ymax) <= self.boundary_margin
            ):
                return True
        return False

    def match_image(self, result: ImageDetectionResult) -> ImageMatchResult:
        """Performs class-aware IoU matching for a single image and returns categorized boxes."""
        gts = result.ground_truths
        preds = result.predictions
        match_res = ImageMatchResult(image_id=result.image_id)

        if not gts:
            match_res.background_fps = list(preds)
            return match_res

        gt_matched = [False] * len(gts)
        gt_boxes = np.array([g.as_xyxy() for g in gts], dtype=np.float32)
        sorted_preds = sorted(preds, key=lambda x: x.score, reverse=True)

        for p in sorted_preds:
            p_box = np.array([p.as_xyxy()], dtype=np.float32)
            ious = compute_iou_matrix(p_box, gt_boxes)[0]
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            best_gt = gts[best_idx]

            if best_iou >= self.iou_threshold:
                if p.class_id == best_gt.class_id:
                    if not gt_matched[best_idx]:
                        match_res.true_positives.append(p)
                        gt_matched[best_idx] = True
                    else:
                        match_res.duplicate_fps.append(p)
                else:
                    match_res.class_confusions.append((p, best_gt.class_id))
            elif best_iou >= self.bg_iou_threshold:
                if p.class_id == best_gt.class_id:
                    match_res.localization_errors.append(p)
                else:
                    match_res.class_confusions.append((p, best_gt.class_id))
            else:
                match_res.background_fps.append(p)

        for idx, matched in enumerate(gt_matched):
            if not matched:
                match_res.false_negatives.append(gts[idx])

        return match_res

    def analyze_image(self, result: ImageDetectionResult) -> Dict[str, Any]:
        """Returns scalar error counts for a single image."""
        m = self.match_image(result)
        w, h = result.width, result.height

        small_misses = sum(1 for g in m.false_negatives if g.area <= SIZE_BINS["small"][1])
        boundary_misses = sum(1 for g in m.false_negatives if self.is_near_boundary(g, w, h))

        return {
            "tp": len(m.true_positives),
            "localization_errors": len(m.localization_errors),
            "class_confusions": len(m.class_confusions),
            "duplicate_fps": len(m.duplicate_fps),
            "background_fps": len(m.background_fps),
            "missed_gts": len(m.false_negatives),
            "small_object_misses": small_misses,
            "boundary_misses": boundary_misses,
            "total_preds": len(result.predictions),
            "total_gts": len(result.ground_truths),
        }

    def aggregate_error_breakdown(self, results: List[ImageDetectionResult]) -> Dict[str, Any]:
        """Aggregates error metrics and confidence distributions across the dataset."""
        totals = {
            "tp": 0,
            "localization_errors": 0,
            "class_confusions": 0,
            "duplicate_fps": 0,
            "background_fps": 0,
            "missed_gts": 0,
            "small_object_misses": 0,
            "boundary_misses": 0,
            "total_preds": 0,
            "total_gts": 0,
        }
        tp_scores: List[float] = []
        fp_scores: List[float] = []

        for r in results:
            m = self.match_image(r)
            img_errs = self.analyze_image(r)
            for k in totals:
                totals[k] += img_errs[k]

            tp_scores.extend(b.score for b in m.true_positives)
            fp_scores.extend(b.score for b in m.all_false_positives)

        total_fps = (
            totals["localization_errors"]
            + totals["class_confusions"]
            + totals["duplicate_fps"]
            + totals["background_fps"]
        )
        totals["total_fps"] = total_fps
        totals["fp_shares"] = {
            "localization": round(totals["localization_errors"] / max(1, total_fps), 4),
            "confusion": round(totals["class_confusions"] / max(1, total_fps), 4),
            "duplicate": round(totals["duplicate_fps"] / max(1, total_fps), 4),
            "background": round(totals["background_fps"] / max(1, total_fps), 4),
        }
        totals["confidence_stats"] = {
            "tp_mean": round(float(np.mean(tp_scores)), 4) if tp_scores else 0.0,
            "fp_mean": round(float(np.mean(fp_scores)), 4) if fp_scores else 0.0,
            "tp_scores": tp_scores,
            "fp_scores": fp_scores,
        }
        return totals

    def analyze_iou_sensitivity(
        self,
        results: List[ImageDetectionResult],
        iou_thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates how TP, localization errors, and recall change across IoU thresholds."""
        thresholds = iou_thresholds or [0.30, 0.50, 0.75]
        sensitivity: List[Dict[str, Any]] = []
        orig_iou = self.iou_threshold

        for thr in thresholds:
            self.iou_threshold = thr
            agg = self.aggregate_error_breakdown(results)
            recall = agg["tp"] / max(1, agg["total_gts"])
            precision = agg["tp"] / max(1, agg["tp"] + agg["total_fps"])
            sensitivity.append({
                "iou_threshold": thr,
                "tp": agg["tp"],
                "localization_errors": agg["localization_errors"],
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            })

        self.iou_threshold = orig_iou
        return sensitivity

    def analyze_boundary_performance(
        self,
        results: List[ImageDetectionResult],
        tiles: Optional[List[TileCrop]] = None,
    ) -> Dict[str, Any]:
        """Evaluates detection recall for image-boundary vs interior vs tile-boundary objects."""
        img_bound_gt, img_bound_miss = 0, 0
        tile_bound_gt, tile_bound_miss = 0, 0
        interior_gt, interior_miss = 0, 0

        for r in results:
            m = self.match_image(r)
            missed_set = set(id(b) for b in m.false_negatives)

            for g in r.ground_truths:
                is_missed = id(g) in missed_set
                near_img = self.is_near_boundary(g, r.width, r.height)
                near_tile = self.is_near_tile_boundary(g, tiles) if tiles else False

                if near_img:
                    img_bound_gt += 1
                    if is_missed:
                        img_bound_miss += 1
                elif near_tile:
                    tile_bound_gt += 1
                    if is_missed:
                        tile_bound_miss += 1
                else:
                    interior_gt += 1
                    if is_missed:
                        interior_miss += 1

        return {
            "image_boundary": {
                "gt_count": img_bound_gt,
                "missed": img_bound_miss,
                "recall": round(1.0 - (img_bound_miss / max(1, img_bound_gt)), 4) if img_bound_gt else 0.0,
            },
            "tile_boundary": {
                "gt_count": tile_bound_gt,
                "missed": tile_bound_miss,
                "recall": round(1.0 - (tile_bound_miss / max(1, tile_bound_gt)), 4) if tile_bound_gt else 0.0,
            },
            "interior": {
                "gt_count": interior_gt,
                "missed": interior_miss,
                "recall": round(1.0 - (interior_miss / max(1, interior_gt)), 4) if interior_gt else 0.0,
            },
        }
