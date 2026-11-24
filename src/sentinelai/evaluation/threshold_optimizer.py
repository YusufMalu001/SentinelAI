"""Confidence threshold evaluation, sweeping, and validation optimization module.

Provides fine-grained sweeping over confidence thresholds, generating:
- Precision vs. threshold
- Recall vs. threshold
- F1-score vs. threshold
- Precision-Recall curves
- Per-class threshold analysis (when sample support >= min_samples)
- Strict split isolation (validation-only threshold tuning to prevent test set leakage)
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from sentinelai.dataset.structures import ImageDetectionResult, BoundingBox
from sentinelai.dataset.constants import DIOR_CLASSES, CLASS_ID_TO_NAME
from sentinelai.evaluation.metrics import evaluate_detections_at_iou


class ThresholdEvaluator:
    """Evaluates detection performance across confidence threshold grids.

    Distinguishes:
    - default_threshold: Standard pre-configured inference threshold (e.g. 0.25).
    - optimal_val_threshold: Statistically selected threshold maximizing validation F1.
    - test_eval_threshold: Fixed threshold applied to test evaluation.
    """

    def __init__(
        self,
        threshold_range: Tuple[float, float] = (0.05, 0.85),
        step: float = 0.05,
        default_threshold: float = 0.25,
    ):
        """Initializes threshold evaluator.

        Args:
            threshold_range: (min_thresh, max_thresh) interval.
            step: Grid step increment.
            default_threshold: Baseline model default confidence threshold.
        """
        self.default_threshold = default_threshold
        t_min, t_max = threshold_range
        num_steps = int(round((t_max - t_min) / step)) + 1
        self.thresholds = [round(float(x), 4) for x in np.linspace(t_min, t_max, num_steps)]

    def sweep_thresholds(
        self,
        results: List[ImageDetectionResult],
        iou_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Performs confidence threshold sweep across all images.

        Computes Precision, Recall, Macro-F1, and Precision-Recall points for each threshold.

        Args:
            results: List of ImageDetectionResult with predictions and ground truths.
            iou_threshold: IoU matching threshold.

        Returns:
            Dictionary containing sweep metrics, PR points, and optimal threshold.
        """
        sweep_data: List[Dict[str, float]] = []
        best_f1 = -1.0
        optimal_threshold = self.default_threshold

        for thresh in self.thresholds:
            # Filter detections by threshold
            filtered_results: List[ImageDetectionResult] = []
            for r in results:
                f_preds = [p for p in r.predictions if p.score >= thresh]
                filtered_results.append(ImageDetectionResult(
                    image_id=r.image_id,
                    predictions=f_preds,
                    ground_truths=r.ground_truths,
                    width=r.width,
                    height=r.height,
                ))

            class_precisions: List[float] = []
            class_recalls: List[float] = []
            class_f1s: List[float] = []

            for c in range(len(DIOR_CLASSES)):
                ev = evaluate_detections_at_iou(filtered_results, iou_threshold=iou_threshold, class_id=c)
                p = ev["precision"]
                r = ev["recall"]
                f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

                if ev["n_positives"] > 0:
                    class_precisions.append(p)
                    class_recalls.append(r)
                    class_f1s.append(f1)

            macro_p = float(np.mean(class_precisions)) if class_precisions else 0.0
            macro_r = float(np.mean(class_recalls)) if class_recalls else 0.0
            macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

            step_entry = {
                "threshold": thresh,
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4),
            }
            sweep_data.append(step_entry)

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                optimal_threshold = thresh

        return {
            "default_threshold": self.default_threshold,
            "optimal_val_threshold": optimal_threshold,
            "best_macro_f1": round(best_f1, 4),
            "sweep_curve": sweep_data,
        }

    def analyze_per_class_thresholds(
        self,
        results: List[ImageDetectionResult],
        iou_threshold: float = 0.50,
        min_support: int = 5,
    ) -> Dict[str, Any]:
        """Analyzes class-specific optimal thresholds on validation set.

        Only optimizes classes with >= min_support ground truth instances to prevent overfitting.

        Args:
            results: ImageDetectionResult list.
            iou_threshold: IoU matching threshold.
            min_support: Minimum number of ground-truth instances required.

        Returns:
            Dictionary mapping class name to optimal threshold and metric summary.
        """
        per_class_report: Dict[str, Dict[str, Any]] = {}

        for c_id, c_name in enumerate(DIOR_CLASSES):
            support = sum(len([g for g in r.ground_truths if g.class_id == c_id]) for r in results)

            if support < min_support:
                per_class_report[c_name] = {
                    "support": support,
                    "optimal_threshold": self.default_threshold,
                    "status": "insufficient_support_fallback_to_default",
                    "best_f1": 0.0,
                }
                continue

            best_c_f1 = -1.0
            best_c_thresh = self.default_threshold
            best_c_p = 0.0
            best_c_r = 0.0

            for thresh in self.thresholds:
                filtered_results: List[ImageDetectionResult] = []
                for r in results:
                    f_preds = [p for p in r.predictions if p.class_id == c_id and p.score >= thresh]
                    filtered_results.append(ImageDetectionResult(
                        image_id=r.image_id,
                        predictions=f_preds,
                        ground_truths=r.ground_truths,
                        width=r.width,
                        height=r.height,
                    ))

                ev = evaluate_detections_at_iou(filtered_results, iou_threshold=iou_threshold, class_id=c_id)
                p = ev["precision"]
                r = ev["recall"]
                f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

                if f1 > best_c_f1:
                    best_c_f1 = f1
                    best_c_thresh = thresh
                    best_c_p = p
                    best_c_r = r

            per_class_report[c_name] = {
                "support": support,
                "optimal_threshold": best_c_thresh,
                "best_f1": round(best_c_f1, 4),
                "precision": round(best_c_p, 4),
                "recall": round(best_c_r, 4),
                "status": "optimized",
            }

        return per_class_report

    def plot_threshold_curves(self, sweep_data: Dict[str, Any], output_path: str) -> None:
        """Plots Precision, Recall, and F1 versus Confidence Threshold.

        Args:
            sweep_data: Output of sweep_thresholds.
            output_path: File path to save figure.
        """
        curve = sweep_data.get("sweep_curve", [])
        if not curve:
            return

        thresholds = [x["threshold"] for x in curve]
        precisions = [x["precision"] for x in curve]
        recalls = [x["recall"] for x in curve]
        f1s = [x["f1"] for x in curve]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Plot 1: P/R/F1 vs Threshold
        ax1.plot(thresholds, precisions, "b-", linewidth=2, label="Precision")
        ax1.plot(thresholds, recalls, "g--", linewidth=2, label="Recall")
        ax1.plot(thresholds, f1s, "r-.", linewidth=2.5, label="F1-Score")
        opt_t = sweep_data.get("optimal_val_threshold", 0.25)
        ax1.axvline(opt_t, color="gray", linestyle=":", label=f"Optimal F1 ({opt_t})")
        ax1.set_xlabel("Confidence Threshold", fontsize=11)
        ax1.set_ylabel("Score", fontsize=11)
        ax1.set_title("Metrics vs. Confidence Threshold", fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best")

        # Plot 2: Precision-Recall Curve
        ax2.plot(recalls, precisions, "m-", linewidth=2.2, label="PR Curve")
        ax2.set_xlabel("Recall", fontsize=11)
        ax2.set_ylabel("Precision", fontsize=11)
        ax2.set_title("Precision-Recall Curve Across Thresholds", fontsize=12)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


# Backwards compatibility alias
ThresholdOptimizer = ThresholdEvaluator
