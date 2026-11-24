"""Evaluation metrics: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, per-class AP, and size-stratified metrics."""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.dataset.constants import DIOR_CLASSES, CLASS_ID_TO_NAME, SIZE_BINS
from sentinelai.postprocessing.fusion import compute_iou_matrix


def compute_ap(recalls: np.ndarray, precisions: np.ndarray, use_11_point: bool = False) -> float:
    """
    Computes Average Precision (AP) using standard continuous all-point interpolation (COCO/VOC 2010+)
    or legacy 11-point interpolation (VOC 2007).
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    if use_11_point:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0.0
            ap += p / 11.0
        return float(ap)
    else:
        # All-point interpolation
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Precision envelope
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Integrate area under curve where recall changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)


def evaluate_detections_at_iou(
    results: List[ImageDetectionResult],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
    size_bin: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates precision, recall, AP for a specific class (or all classes) at a given IoU threshold.
    Optionally filters ground truths and predictions to a specific size bin ('small', 'medium', 'large').
    """
    all_preds: List[Tuple[float, int, str, BoundingBox]] = []  # (score, is_tp, img_id, box)
    n_positives = 0

    # Index ground truths per image
    gt_by_img: Dict[str, List[BoundingBox]] = {}
    for res in results:
        gts = res.ground_truths
        if class_id is not None:
            gts = [g for g in gts if g.class_id == class_id]
        if size_bin is not None and size_bin in SIZE_BINS:
            min_a, max_a = SIZE_BINS[size_bin]
            gts = [g for g in gts if min_a <= g.area < max_a]
        gt_by_img[res.image_id] = gts
        n_positives += len(gts)

    # Gather predictions
    for res in results:
        preds = res.predictions
        if class_id is not None:
            preds = [p for p in preds if p.class_id == class_id]
        if size_bin is not None and size_bin in SIZE_BINS:
            min_a, max_a = SIZE_BINS[size_bin]
            preds = [p for p in preds if min_a <= p.area < max_a]

        for p in preds:
            all_preds.append((p.score, res.image_id, p))

    if n_positives == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_positives": 0,
            "n_predictions": len(all_preds),
            "tp": 0,
            "fp": len(all_preds)
        }

    if not all_preds:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_positives": n_positives,
            "n_predictions": 0,
            "tp": 0,
            "fp": 0
        }

    # Sort predictions descending by score
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Track matched GTs per image
    matched_gt: Dict[str, set] = {img_id: set() for img_id in gt_by_img}

    tp_flags = np.zeros(len(all_preds), dtype=np.float32)
    fp_flags = np.zeros(len(all_preds), dtype=np.float32)

    for i, (score, img_id, p_box) in enumerate(all_preds):
        gts = gt_by_img.get(img_id, [])
        if not gts:
            fp_flags[i] = 1.0
            continue

        p_coords = np.array([p_box.as_xyxy()], dtype=np.float32)
        gt_coords = np.array([g.as_xyxy() for g in gts], dtype=np.float32)

        ious = compute_iou_matrix(p_coords, gt_coords)[0]
        best_gt_idx = int(np.argmax(ious))
        best_iou = ious[best_gt_idx]

        if best_iou >= iou_threshold:
            if best_gt_idx not in matched_gt[img_id]:
                tp_flags[i] = 1.0
                matched_gt[img_id].add(best_gt_idx)
            else:
                fp_flags[i] = 1.0  # Duplicate detection of same GT
        else:
            fp_flags[i] = 1.0

    acc_tp = np.cumsum(tp_flags)
    acc_fp = np.cumsum(fp_flags)

    recalls = acc_tp / float(n_positives)
    precisions = acc_tp / np.maximum(acc_tp + acc_fp, 1e-8)

    ap = compute_ap(recalls, precisions)

    final_tp = int(acc_tp[-1]) if len(acc_tp) > 0 else 0
    final_fp = int(acc_fp[-1]) if len(acc_fp) > 0 else 0
    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "n_positives": n_positives,
        "n_predictions": len(all_preds),
        "tp": final_tp,
        "fp": final_fp,
        "recalls": recalls.tolist(),
        "precisions": precisions.tolist()
    }


def compute_coco_metrics(
    results: List[ImageDetectionResult],
    num_classes: int = len(DIOR_CLASSES)
) -> Dict[str, Any]:
    """
    Computes complete object detection benchmark metrics:
    - mAP@0.5
    - mAP@0.5:0.95 (COCO standard average across 10 IoU thresholds 0.50:0.05:0.95)
    - Per-class AP@0.5 and AP@0.5:0.95
    - Size-stratified AP@0.5 (small, medium, large)
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    per_class_map50: Dict[str, float] = {}
    per_class_map5095: Dict[str, float] = {}

    for c in range(num_classes):
        c_name = CLASS_ID_TO_NAME.get(c, str(c))
        aps_at_ious: List[float] = []

        for iou in iou_thresholds:
            eval_res = evaluate_detections_at_iou(results, iou_threshold=float(iou), class_id=c)
            aps_at_ious.append(eval_res["ap"])

        per_class_map50[c_name] = aps_at_ious[0]  # IoU 0.50
        per_class_map5095[c_name] = float(np.mean(aps_at_ious))

    # Overall mAP
    mean_map50 = float(np.mean(list(per_class_map50.values())))
    mean_map5095 = float(np.mean(list(per_class_map5095.values())))

    # Size-stratified evaluation
    size_metrics: Dict[str, Dict[str, float]] = {}
    for size_bin in SIZE_BINS:
        bin_aps50: List[float] = []
        bin_aps5095: List[float] = []
        for c in range(num_classes):
            c_ious = [
                evaluate_detections_at_iou(results, iou_threshold=float(iou), class_id=c, size_bin=size_bin)["ap"]
                for iou in iou_thresholds
            ]
            bin_aps50.append(c_ious[0])
            bin_aps5095.append(float(np.mean(c_ious)))

        size_metrics[size_bin] = {
            "mAP50": float(np.mean(bin_aps50)),
            "mAP50_95": float(np.mean(bin_aps5095))
        }

    return {
        "mAP50": mean_map50,
        "mAP50_95": mean_map5095,
        "per_class_mAP50": per_class_map50,
        "per_class_mAP50_95": per_class_map5095,
        "size_metrics": size_metrics
    }
