"""Scientific visualization utilities for detection results, comparative diagnostics, and research plots.

Consistent Color Convention (RGB):
- True Positive (TP):     Green   (0, 200, 0)
- False Positive (FP):    Red     (220, 30, 30)
- False Negative (FN):    Orange  (255, 165, 0)
- Ground Truth (GT):      Blue    (30, 144, 255)
- Experimental Gain:      Cyan    (0, 220, 220)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.dataset.constants import CLASS_ID_TO_NAME
from sentinelai.evaluation.error_analysis import DetectionErrorAnalyzer


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[BoundingBox],
    color: tuple = (0, 200, 0),
    draw_label: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draws bounding boxes and labels on an RGB image."""
    canvas = image.copy()
    for b in boxes:
        p1 = (int(b.xmin), int(b.ymin))
        p2 = (int(b.xmax), int(b.ymax))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if draw_label:
            name = b.class_name if b.class_name else CLASS_ID_TO_NAME.get(b.class_id, str(b.class_id))
            label = f"{name} {b.score:.2f}"
            cv2.putText(
                canvas, label, (p1[0], max(15, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )
    return canvas


def plot_detection_comparison(
    image_path: str,
    ground_truths: List[BoundingBox],
    predictions: List[BoundingBox],
    output_path: str,
    title: str = "Ground Truth vs Model Predictions",
) -> None:
    """Saves a side-by-side comparison figure of Ground Truths vs Model Predictions."""
    img = cv2.imread(image_path) if os.path.exists(image_path) else None
    if img is None:
        img = np.zeros((800, 800, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_vis = draw_bounding_boxes(img_rgb, ground_truths, color=(30, 144, 255), draw_label=True)
    pred_vis = draw_bounding_boxes(img_rgb, predictions, color=(220, 30, 30), draw_label=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(gt_vis)
    axes[0].set_title(f"Ground Truth ({len(ground_truths)} objects)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(pred_vis)
    axes[1].set_title(f"Predictions ({len(predictions)} detections)", fontsize=13)
    axes[1].axis("off")

    plt.suptitle(title, fontsize=15, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_comparative_diagnostics(
    image_path: str,
    baseline_result: ImageDetectionResult,
    exp_result: ImageDetectionResult,
    output_path: str,
    iou_threshold: float = 0.50,
    title: str = "Baseline vs Experimental Diagnostic Comparison",
) -> None:
    """Generates a 3-panel diagnostic figure:
    1. Baseline (TP=Green, FP=Red, FN=Orange)
    2. Experimental (TP=Green, FP=Red, FN=Orange)
    3. Detection Differences (GT=Blue, Experimental=Cyan)
    """
    img = cv2.imread(image_path) if os.path.exists(image_path) else None
    if img is None:
        img = np.zeros((baseline_result.height, baseline_result.width, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    analyzer = DetectionErrorAnalyzer(iou_threshold=iou_threshold)
    base_match = analyzer.match_image(baseline_result)
    exp_match = analyzer.match_image(exp_result)

    # Panel 1: Baseline Diagnostics
    p1_img = draw_bounding_boxes(img_rgb, base_match.true_positives, color=(0, 200, 0), draw_label=False)
    p1_img = draw_bounding_boxes(p1_img, base_match.all_false_positives, color=(220, 30, 30), draw_label=False)
    p1_img = draw_bounding_boxes(p1_img, base_match.false_negatives, color=(255, 165, 0), draw_label=False)

    # Panel 2: Experimental Diagnostics
    p2_img = draw_bounding_boxes(img_rgb, exp_match.true_positives, color=(0, 200, 0), draw_label=False)
    p2_img = draw_bounding_boxes(p2_img, exp_match.all_false_positives, color=(220, 30, 30), draw_label=False)
    p2_img = draw_bounding_boxes(p2_img, exp_match.false_negatives, color=(255, 165, 0), draw_label=False)

    # Panel 3: Detection Differences (GT in Blue, Exp Predictions in Cyan)
    p3_img = draw_bounding_boxes(img_rgb, baseline_result.ground_truths, color=(30, 144, 255), draw_label=False)
    p3_img = draw_bounding_boxes(p3_img, exp_result.predictions, color=(0, 220, 220), draw_label=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(p1_img)
    axes[0].set_title(
        f"Baseline (TP:{len(base_match.true_positives)} FP:{len(base_match.all_false_positives)} FN:{len(base_match.false_negatives)})",
        fontsize=11,
    )
    axes[0].axis("off")

    axes[1].imshow(p2_img)
    axes[1].set_title(
        f"Experimental (TP:{len(exp_match.true_positives)} FP:{len(exp_match.all_false_positives)} FN:{len(exp_match.false_negatives)})",
        fontsize=11,
    )
    axes[1].axis("off")

    axes[2].imshow(p3_img)
    axes[2].set_title("Overlay: GT (Blue) vs Exp (Cyan)", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(
    tp_scores: List[float],
    fp_scores: List[float],
    output_path: str,
    title: str = "Confidence Score Distribution: True Positives vs False Positives",
) -> None:
    """Plots overlapping histograms of confidence scores for TPs and FPs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)

    if tp_scores:
        ax.hist(tp_scores, bins=bins, alpha=0.6, color="green", label=f"True Positives (n={len(tp_scores)})", edgecolor="black")
    if fp_scores:
        ax.hist(fp_scores, bins=bins, alpha=0.6, color="red", label=f"False Positives (n={len(fp_scores)})", edgecolor="black")

    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_class_wise_ap(
    baseline_ap: Dict[str, float],
    exp_ap: Dict[str, float],
    output_path: str,
    title: str = "Class-Wise AP@0.5: Baseline vs Experimental",
) -> None:
    """Plots grouped horizontal bar chart of per-class AP@0.5."""
    classes = [c for c in baseline_ap.keys() if baseline_ap.get(c, 0) > 0 or exp_ap.get(c, 0) > 0]
    if not classes:
        classes = list(baseline_ap.keys())[:10]

    y = np.arange(len(classes))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.4)))
    base_vals = [baseline_ap.get(c, 0.0) for c in classes]
    exp_vals = [exp_ap.get(c, 0.0) for c in classes]

    ax.barh(y - height / 2, base_vals, height, label="Baseline", color="#4c72b0")
    ax.barh(y + height / 2, exp_vals, height, label="Experimental", color="#55a868")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("AP@0.5", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_density_performance(
    baseline_density: Dict[str, Any],
    exp_density: Dict[str, Any],
    output_path: str,
    title: str = "Object-Density Performance: Baseline vs Experimental",
) -> None:
    """Plots mAP@0.5 across Sparse, Moderate, and Dense image categories."""
    cats = ["sparse", "moderate", "dense"]
    x = np.arange(len(cats))
    width = 0.35

    base_vals = [baseline_density.get("categories", {}).get(c, {}).get("mAP50", 0.0) for c in cats]
    exp_vals = [exp_density.get("categories", {}).get(c, {}).get("mAP50", 0.0) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#4c72b0")
    ax.bar(x + width / 2, exp_vals, width, label="Experimental", color="#dd8452")

    ax.set_xticks(x)
    ax.set_xticklabels(["Sparse (0-3)", "Moderate (4-10)", "Dense (11+)"], fontsize=11)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_latency_vs_accuracy(
    experiment_results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Plots Pareto frontier: Inference Latency (ms) vs mAP@0.5."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for exp in experiment_results:
        name = exp.get("name", "exp")
        lat = exp.get("latency_ms", 0.0)
        map50 = exp.get("mAP50", 0.0)
        ax.scatter(lat, map50, s=120, label=name)
        ax.annotate(name, (lat, map50), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("Inference Latency per Image (ms)", fontsize=12)
    ax.set_ylabel("Detection Accuracy (mAP@0.5)", fontsize=12)
    ax.set_title("SentinelAI: Accuracy vs Inference Latency Trade-off", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_size_stratified_performance(
    size_metrics_by_model: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
) -> None:
    """Plots bar chart comparing mAP across Small, Medium, and Large object categories."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    categories = ["small", "medium", "large"]
    models = list(size_metrics_by_model.keys())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))

    for i, mod in enumerate(models):
        values = [size_metrics_by_model[mod].get(cat, {}).get("mAP50", 0.0) for cat in categories]
        ax.bar(x + i * width, values, width, label=mod)

    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Object Size-Stratified Detection Performance", fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2.0)
    ax.set_xticklabels(["Small (area <= 32²)", "Medium (32² < area <= 96²)", "Large (area > 96²)"], fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
