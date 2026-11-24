"""Multi-scale test-time augmentation (TTA) inference pipeline.

Executes detector across multiple image resolutions and fuses candidate predictions
with strict coordinate consistency and runtime profiling.
"""

import time
from typing import List, Union, Optional, Dict, Any
import numpy as np
import cv2

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.inference.detector import BaseDetector
from sentinelai.postprocessing.fusion import standard_nms, soft_nms, weighted_boxes_fusion


class MultiScaleDetector:
    """Multi-Scale Test-Time Augmentation (TTA) detector.

    Predicts across an ensemble of input image scales (e.g., 640, 800, 1024)
    and merges predictions into a unified set using class-aware fusion.
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        scales: Optional[List[int]] = None,
        fusion_method: str = "wbf",
        fusion_iou_threshold: float = 0.55,
    ):
        """Initializes multi-scale detector.

        Args:
            base_detector: Underlying object detector.
            scales: List of integer input image resolutions (e.g., [640, 800, 1024]).
                    Duplicate values are automatically deduplicated.
            fusion_method: Merging algorithm ('nms', 'soft_nms', 'wbf').
            fusion_iou_threshold: Overlap threshold for merging candidate boxes.
        """
        self.base_detector = base_detector
        resolved_scales = scales if scales is not None else [640, 800, 1024]
        # Deduplicate while preserving order
        seen = set()
        self.scales: List[int] = []
        for s in resolved_scales:
            s_int = int(s)
            if s_int not in seen and s_int > 0:
                seen.add(s_int)
                self.scales.append(s_int)

        self.fusion_method = fusion_method.lower()
        self.fusion_iou_threshold = fusion_iou_threshold

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> ImageDetectionResult:
        """Executes multi-scale inference and merges predictions.

        Args:
            image_input: File path or loaded BGR numpy array [H, W, C].
            conf_threshold: Minimum confidence threshold.

        Returns:
            ImageDetectionResult with fused predictions in original pixel coordinates.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image at path: {image_input}")
            img_id = image_input
        else:
            image = image_input
            img_id = "numpy_image"

        img_h, img_w = image.shape[:2]

        all_boxes: List[BoundingBox] = []
        total_inf_time = 0.0
        scale_latencies: Dict[int, float] = {}
        scale_counts: Dict[int, int] = {}

        # If empty image or zero dimensions
        if img_h <= 0 or img_w <= 0:
            return ImageDetectionResult(image_id=img_id, width=img_w, height=img_h)

        # Run inference across each distinct scale
        for scale in self.scales:
            scale_res = self.base_detector.predict(
                image,
                conf_threshold=conf_threshold,
                imgsz=scale,
            )
            total_inf_time += scale_res.inference_time_ms
            scale_latencies[scale] = scale_res.inference_time_ms
            scale_counts[scale] = len(scale_res.predictions)

            # Ensure coordinates are clipped to original image frame
            for b in scale_res.predictions:
                clipped_b = b.clip(img_w, img_h)
                if clipped_b.width > 0 and clipped_b.height > 0:
                    all_boxes.append(clipped_b)

        # Fuse multi-scale candidate predictions
        post_start = time.perf_counter()
        if not all_boxes:
            fused_boxes: List[BoundingBox] = []
        elif self.fusion_method == "wbf":
            fused_boxes = weighted_boxes_fusion(
                all_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
                img_width=img_w,
                img_height=img_h,
            )
        elif self.fusion_method == "soft_nms":
            fused_boxes = soft_nms(
                all_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
            )
        else:
            fused_boxes = standard_nms(
                all_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
            )

        post_latency = (time.perf_counter() - post_start) * 1000.0

        return ImageDetectionResult(
            image_id=img_id,
            image_path=image_input if isinstance(image_input, str) else None,
            width=img_w,
            height=img_h,
            predictions=fused_boxes,
            inference_time_ms=total_inf_time,
            postprocess_time_ms=post_latency,
            metadata={
                "scales": self.scales,
                "fusion_method": self.fusion_method,
                "raw_candidate_count": len(all_boxes),
                "fused_count": len(fused_boxes),
                "per_scale_latency_ms": scale_latencies,
                "per_scale_detection_counts": scale_counts,
            },
        )
