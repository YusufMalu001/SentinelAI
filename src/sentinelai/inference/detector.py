"""Detector interface and YOLOv8 wrapper for SentinelAI."""

import os
import time
from typing import List, Optional, Union, Dict, Any
import numpy as np

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult
from sentinelai.dataset.constants import CLASS_ID_TO_NAME, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


class BaseDetector:
    """Abstract base detector interface supporting single and batched inference."""

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 800,
    ) -> ImageDetectionResult:
        """Runs inference on a single image."""
        raise NotImplementedError

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 800,
    ) -> List[ImageDetectionResult]:
        """Runs inference on a list/batch of images."""
        raise NotImplementedError


class YOLOv8Detector(BaseDetector):
    """Ultralytics YOLOv8 inference wrapper with standardized SentinelAI data structures,

    batched prediction, latency tracking, and GPU memory monitoring.
    """

    def __init__(
        self,
        model_path: str = "runs/detect/yolov8n_epochs50_batch16/weights/best.pt",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        from ultralytics import YOLO
        if not os.path.exists(self.model_path):
            fallback = "yolov8n.pt"
            if os.path.exists(fallback):
                self.model = YOLO(fallback)
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {self.model_path} or {fallback}")
        else:
            self.model = YOLO(self.model_path)

    def warmup(self, imgsz: int = 800, iterations: int = 2) -> float:
        """Warms up the model with dummy inputs to stabilize latency measurements.

        Returns:
            Warmup duration in milliseconds.
        """
        start_t = time.perf_counter()
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            _ = self.model.predict(source=dummy, imgsz=imgsz, device=self.device, verbose=False)
        return (time.perf_counter() - start_t) * 1000.0

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 800,
    ) -> List[ImageDetectionResult]:
        """Runs batched inference on multiple image inputs."""
        if not images:
            return []

        start_t = time.perf_counter()

        # Execute prediction through Ultralytics YOLO
        raw_results = self.model.predict(
            source=images,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        total_latency_ms = (time.perf_counter() - start_t) * 1000.0
        avg_inf_ms = total_latency_ms / len(images)

        detection_results: List[ImageDetectionResult] = []

        for idx, res in enumerate(raw_results):
            img_input = images[idx]
            image_id = f"img_{idx}"
            img_path = None
            if isinstance(img_input, str):
                image_id = os.path.splitext(os.path.basename(img_input))[0]
                img_path = img_input

            img_h, img_w = imgsz, imgsz
            if hasattr(res, "orig_shape") and res.orig_shape is not None:
                img_h, img_w = int(res.orig_shape[0]), int(res.orig_shape[1])

            boxes: List[BoundingBox] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    c_id = int(cls_arr[i])
                    boxes.append(BoundingBox(
                        xmin=float(xyxy[i, 0]),
                        ymin=float(xyxy[i, 1]),
                        xmax=float(xyxy[i, 2]),
                        ymax=float(xyxy[i, 3]),
                        class_id=c_id,
                        score=float(conf[i]),
                        class_name=CLASS_ID_TO_NAME.get(c_id, str(c_id))
                    ).clip(img_w, img_h))

            speed_dict = getattr(res, "speed", {}) or {}
            pre_t = speed_dict.get("preprocess", 0.0)
            inf_t = speed_dict.get("inference", avg_inf_ms)
            post_t = speed_dict.get("postprocess", 0.0)

            detection_results.append(ImageDetectionResult(
                image_id=image_id,
                image_path=img_path,
                width=img_w,
                height=img_h,
                predictions=boxes,
                inference_time_ms=inf_t,
                preprocess_time_ms=pre_t,
                postprocess_time_ms=post_t
            ))

        return detection_results

    def predict(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 800,
    ) -> ImageDetectionResult:
        """Runs inference on a single image."""
        batch_res = self.predict_batch([image], conf_threshold=conf_threshold, iou_threshold=iou_threshold, imgsz=imgsz)
        return batch_res[0]
