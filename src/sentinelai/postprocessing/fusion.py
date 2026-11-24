r"""Bounding box fusion algorithms: Standard NMS, Soft-NMS, and Weighted Boxes Fusion (WBF).

Algorithmic Comparison & Trade-off Analysis:
--------------------------------------------
1. Standard Greedy Non-Maximum Suppression (NMS):
   - Principle: Iteratively selects the detection with the highest confidence score,
     and completely suppresses (discards) any neighboring candidate detection of the
     same class whose IoU exceeds the overlap threshold `iou_threshold`.
   - Strengths: Highly effective at removing duplicate detections across overlapping tile
     boundaries; computationally fast ($O(N^2)$ worst-case, fast in practice).
   - Limitations: In crowded aerial scenes (e.g. adjacent vehicles in parking lots or ships
     at harbor docks), true neighboring objects with high overlap can be erroneously eliminated,
     leading to false negatives.

2. Soft-NMS (Bodla et al., ICCV 2017):
   - Principle: Instead of hard pruning overlapping candidate detections to zero, Soft-NMS
     decays their confidence scores as a continuous function of overlap (IoU).
     Supports Gaussian decay ($s_i \leftarrow s_i \cdot \exp(-\text{IoU}^2 / \sigma)$) or
     Linear decay ($s_i \leftarrow s_i \cdot (1 - \text{IoU})$ for $\text{IoU} > \text{thresh}$).
   - Strengths: Retains detections in occluded and dense clusters where hard NMS fails.
   - Limitations: Requires careful score thresholding post-decay; in spatial tiling seams,
     it may retain marginal duplicate predictions if decay is insufficiently aggressive.

3. Weighted Boxes Fusion (WBF) (Solovyev et al., 2021):
   - Principle: Rather than discarding or decaying candidates, WBF clusters overlapping
     predictions and computes a fused bounding box whose coordinates are the weighted
     average of all cluster members, weighted by confidence:
       x_fused = (sum_i c_i * x_i) / (sum_i c_i)
   - Strengths: Produces refined, jitter-free bounding box coordinates and leverages
     consensus among overlapping tiles or multi-scale passes.
   - Limitations: Sensitive to outlier detections with high false confidence; incurs higher
     computational complexity during pairwise clustering.

Note on Comparative Validity:
-----------------------------
No algorithm is claimed to be universally superior a priori; performance depends on object
density, sensor GSD (ground sampling distance), tile overlap ratios, and confidence calibration.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from sentinelai.dataset.structures import BoundingBox


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection-over-Union (IoU) between two sets of boxes.

    Args:
        boxes_a: Array of shape [N, 4] with rows [xmin, ymin, xmax, ymax].
        boxes_b: Array of shape [M, 4] with rows [xmin, ymin, xmax, ymax].

    Returns:
        Array of shape [N, M] with IoU values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(0.0, np.minimum(boxes_a[:, 2:3], boxes_b[:, 2]) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0]))
    ih = np.maximum(0.0, np.minimum(boxes_a[:, 3:4], boxes_b[:, 3]) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1]))
    intersection = iw * ih

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-8)
    return np.clip(intersection / union, 0.0, 1.0)


def standard_nms(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.05,
) -> List[BoundingBox]:
    """Class-aware Standard Greedy Non-Maximum Suppression.

    Args:
        boxes: List of candidate bounding boxes.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.
        conf_threshold: Minimum confidence score to consider.

    Returns:
        List of retained bounding boxes sorted descending by confidence score.
    """
    if not boxes:
        return []

    # Filter out boxes below confidence threshold
    filtered = [b for b in boxes if b.score >= conf_threshold and b.width > 0 and b.height > 0]
    if not filtered:
        return []

    classes = set(b.class_id for b in filtered)
    kept_boxes: List[BoundingBox] = []

    for cls_id in classes:
        cls_boxes = [b for b in filtered if b.class_id == cls_id]
        if not cls_boxes:
            continue

        cls_boxes.sort(key=lambda x: x.score, reverse=True)
        box_coords = np.array([b.as_xyxy() for b in cls_boxes], dtype=np.float32)

        indices = np.arange(len(cls_boxes))
        while len(indices) > 0:
            current_idx = indices[0]
            kept_boxes.append(cls_boxes[current_idx])

            if len(indices) == 1:
                break

            current_box = box_coords[current_idx : current_idx + 1]
            remaining_indices = indices[1:]
            remaining_boxes = box_coords[remaining_indices]

            ious = compute_iou_matrix(current_box, remaining_boxes)[0]
            indices = remaining_indices[ious < iou_threshold]

    kept_boxes.sort(key=lambda x: x.score, reverse=True)
    return kept_boxes


def soft_nms(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.50,
    sigma: float = 0.50,
    conf_threshold: float = 0.05,
    method: str = "gaussian",
) -> List[BoundingBox]:
    """Class-aware Soft-NMS with Gaussian or Linear score decay.

    Args:
        boxes: List of candidate bounding boxes.
        iou_threshold: Overlap threshold used in linear decay mode.
        sigma: Variance parameter for Gaussian decay mode.
        conf_threshold: Minimum confidence score required after decay.
        method: Decay function ('gaussian' or 'linear').

    Returns:
        List of bounding boxes with updated confidence scores.
    """
    if not boxes:
        return []

    filtered = [b for b in boxes if b.score >= conf_threshold and b.width > 0 and b.height > 0]
    if not filtered:
        return []

    classes = set(b.class_id for b in filtered)
    results: List[BoundingBox] = []

    for cls_id in classes:
        cls_boxes = [b for b in filtered if b.class_id == cls_id]
        if not cls_boxes:
            continue

        coords = np.array([b.as_xyxy() for b in cls_boxes], dtype=np.float32)
        scores = np.array([b.score for b in cls_boxes], dtype=np.float32)
        n = len(cls_boxes)

        for i in range(n):
            max_pos = i + int(np.argmax(scores[i:]))

            # Swap current i with maximum score candidate
            coords[[i, max_pos]] = coords[[max_pos, i]]
            scores[[i, max_pos]] = scores[[max_pos, i]]
            cls_boxes[i], cls_boxes[max_pos] = cls_boxes[max_pos], cls_boxes[i]

            if i + 1 < n:
                ious = compute_iou_matrix(coords[i : i + 1], coords[i + 1 :])[0]
                if method.lower() == "gaussian":
                    weight = np.exp(-(ious ** 2) / max(1e-6, sigma))
                else:  # linear decay
                    weight = np.ones_like(ious)
                    overlap_mask = ious > iou_threshold
                    weight[overlap_mask] = 1.0 - ious[overlap_mask]

                scores[i + 1 :] = scores[i + 1 :] * weight

        for b, s, c in zip(cls_boxes, scores, coords):
            if s >= conf_threshold:
                results.append(BoundingBox(
                    xmin=float(c[0]),
                    ymin=float(c[1]),
                    xmax=float(c[2]),
                    ymax=float(c[3]),
                    class_id=cls_id,
                    score=float(s),
                    class_name=b.class_name,
                ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def weighted_boxes_fusion(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.55,
    conf_threshold: float = 0.05,
    img_width: int = 800,
    img_height: int = 800,
) -> List[BoundingBox]:
    """Class-aware Weighted Boxes Fusion (WBF).

    Computes fused bounding box coordinates weighted by candidate confidence scores.
    Uses ensemble-boxes library if available; falls back to exact internal numpy implementation.

    Args:
        boxes: List of candidate bounding boxes.
        iou_threshold: Overlap threshold for clustering candidate boxes.
        conf_threshold: Minimum confidence score to retain.
        img_width: Image width for normalization.
        img_height: Image height for normalization.

    Returns:
        List of fused bounding boxes sorted descending by score.
    """
    if not boxes:
        return []

    try:
        from ensemble_boxes import weighted_boxes_fusion as eb_wbf

        norm_boxes = []
        scores = []
        labels = []
        class_name_map = {}

        for b in boxes:
            if b.score >= conf_threshold and b.width > 0 and b.height > 0:
                norm_boxes.append(b.as_normalized_xyxy(img_width, img_height))
                scores.append(b.score)
                labels.append(b.class_id)
                if b.class_id not in class_name_map and b.class_name:
                    class_name_map[b.class_id] = b.class_name

        if not norm_boxes:
            return []

        boxes_list = [norm_boxes]
        scores_list = [scores]
        labels_list = [labels]

        fused_boxes, fused_scores, fused_labels = eb_wbf(
            boxes_list,
            scores_list,
            labels_list,
            weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=conf_threshold,
        )

        results: List[BoundingBox] = []
        for b, s, l in zip(fused_boxes, fused_scores, fused_labels):
            cls_id = int(l)
            results.append(BoundingBox(
                xmin=float(b[0] * img_width),
                ymin=float(b[1] * img_height),
                xmax=float(b[2] * img_width),
                ymax=float(b[3] * img_height),
                class_id=cls_id,
                score=float(s),
                class_name=class_name_map.get(cls_id),
            ).clip(img_width, img_height))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    except Exception:
        return _pure_wbf(boxes, iou_threshold, conf_threshold, img_width, img_height)


def _pure_wbf(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.55,
    conf_threshold: float = 0.05,
    img_width: int = 800,
    img_height: int = 800,
) -> List[BoundingBox]:
    """Pure Python/Numpy Weighted Boxes Fusion implementation."""
    filtered = [b for b in boxes if b.score >= conf_threshold and b.width > 0 and b.height > 0]
    if not filtered:
        return []

    classes = set(b.class_id for b in filtered)
    results: List[BoundingBox] = []

    for cls_id in classes:
        cls_boxes = [b for b in filtered if b.class_id == cls_id]
        cls_boxes.sort(key=lambda x: x.score, reverse=True)

        clusters: List[List[BoundingBox]] = []

        for b in cls_boxes:
            matched = False
            b_coords = np.array([b.as_xyxy()], dtype=np.float32)

            for cluster in clusters:
                total_w = sum(x.score for x in cluster)
                c_coords = np.array([[
                    sum(x.xmin * x.score for x in cluster) / total_w,
                    sum(x.ymin * x.score for x in cluster) / total_w,
                    sum(x.xmax * x.score for x in cluster) / total_w,
                    sum(x.ymax * x.score for x in cluster) / total_w,
                ]], dtype=np.float32)

                iou = compute_iou_matrix(b_coords, c_coords)[0, 0]
                if iou >= iou_threshold:
                    cluster.append(b)
                    matched = True
                    break

            if not matched:
                clusters.append([b])

        for cluster in clusters:
            total_weight = sum(x.score for x in cluster)
            fused_xmin = sum(x.xmin * x.score for x in cluster) / total_weight
            fused_ymin = sum(x.ymin * x.score for x in cluster) / total_weight
            fused_xmax = sum(x.xmax * x.score for x in cluster) / total_weight
            fused_ymax = sum(x.ymax * x.score for x in cluster) / total_weight

            fused_score = min(1.0, total_weight / len(cluster))

            results.append(BoundingBox(
                xmin=float(fused_xmin),
                ymin=float(fused_ymin),
                xmax=float(fused_xmax),
                ymax=float(fused_ymax),
                class_id=cls_id,
                score=float(fused_score),
                class_name=cluster[0].class_name,
            ).clip(img_width, img_height))

    results.sort(key=lambda x: x.score, reverse=True)
    return results
