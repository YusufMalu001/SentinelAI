"""Dataset loaders, parser for YOLO/Pascal VOC formats, and test split utilities."""

import os
import glob
from typing import List, Dict, Tuple, Optional
from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.dataset.constants import DIOR_CLASSES, CLASS_ID_TO_NAME, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def load_yolo_annotations(
    label_path: str,
    img_width: int = DEFAULT_IMAGE_WIDTH,
    img_height: int = DEFAULT_IMAGE_HEIGHT
) -> List[BoundingBox]:
    """Parse YOLO txt format (class_id center_x center_y width height) normalized to pixel coordinates."""
    boxes: List[BoundingBox] = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height

            xmin = max(0.0, cx - w / 2.0)
            ymin = max(0.0, cy - h / 2.0)
            xmax = min(float(img_width), cx + w / 2.0)
            ymax = min(float(img_height), cy + h / 2.0)

            boxes.append(BoundingBox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_id=class_id,
                score=1.0,
                class_name=CLASS_ID_TO_NAME.get(class_id, str(class_id))
            ))
    return boxes


def save_yolo_annotations(
    boxes: List[BoundingBox],
    output_path: str,
    img_width: int = DEFAULT_IMAGE_WIDTH,
    img_height: int = DEFAULT_IMAGE_HEIGHT
) -> None:
    """Save bounding boxes in YOLO txt format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for b in boxes:
            cx = ((b.xmin + b.xmax) / 2.0) / img_width
            cy = ((b.ymin + b.ymax) / 2.0) / img_height
            w = b.width / img_width
            h = b.height / img_height
            f.write(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
