"""Spatial tiling sliding-window inference pipeline.

Handles tiled image partitioning, batch tile inference, global coordinate remapping,
border clipping, empty-detection handling, and tile-aware fusion.
"""

import time
from typing import List, Union, Optional, Dict, Any
import numpy as np
import cv2

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult, TileCrop
from sentinelai.inference.detector import BaseDetector
from sentinelai.inference.tiler import SpatialTiler
from sentinelai.postprocessing.fusion import standard_nms, soft_nms, weighted_boxes_fusion


class TiledDetector:
    """Sliding-window tiled detector pipeline for aerial imagery.

    Executes overlapping sliding-window inference over large scenes, re-maps
    tile-local bounding box coordinates to the global image coordinate system,
    and resolves duplicate detections across overlapping seams using class-aware fusion.
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_size: Optional[int] = None,
        overlap_x: Optional[float] = None,
        overlap_y: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
        fusion_method: str = "nms",  # "nms", "soft_nms", "wbf"
        fusion_iou_threshold: float = 0.50,
        include_full_image: bool = True,
        batch_size: int = 1,
    ):
        """Initializes the tiled detector.

        Args:
            base_detector: Detector instance implementing predict / predict_batch.
            tile_width: Width of tile window.
            tile_height: Height of tile window.
            tile_size: Square tile dimension fallback (default: 416).
            overlap_x: Horizontal overlap ratio in [0, 1).
            overlap_y: Vertical overlap ratio in [0, 1).
            overlap_ratio: Overlap ratio fallback in [0, 1) (default: 0.25).
            fusion_method: Bounding-box fusion algorithm ('nms', 'soft_nms', 'wbf').
            fusion_iou_threshold: Overlap threshold for merging boxes.
            include_full_image: Whether to run a whole-image pass in addition to tiles.
            batch_size: Batch size for batched tile inference.
        """
        self.base_detector = base_detector
        resolved_size = 416 if (tile_size is None and tile_width is None and tile_height is None) else tile_size
        self.tiler = SpatialTiler(
            tile_width=tile_width,
            tile_height=tile_height,
            tile_size=resolved_size,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
            overlap_ratio=overlap_ratio,
        )
        self.tile_width = self.tiler.tile_width
        self.tile_height = self.tiler.tile_height
        self.overlap_x = self.tiler.overlap_x
        self.overlap_y = self.tiler.overlap_y
        self.fusion_method = fusion_method.lower()
        self.fusion_iou_threshold = fusion_iou_threshold
        self.include_full_image = include_full_image
        self.batch_size = max(1, int(batch_size))

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        tile_imgsz: Optional[int] = None,
    ) -> ImageDetectionResult:
        """Executes tiled inference on an image and returns full-image detection results.

        Args:
            image_input: File path to image or loaded BGR numpy array [H, W, C].
            conf_threshold: Minimum confidence threshold.
            tile_imgsz: Input size passed to model for tile inference. If None,
                        uses max(tile_width, tile_height).

        Returns:
            ImageDetectionResult containing fused predictions in global pixel coordinates.
        """
        wall_start = time.perf_counter()

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image at path: {image_input}")
            img_id = image_input
        else:
            image = image_input
            img_id = "numpy_image"

        img_h, img_w = image.shape[:2]
        inference_crop_size = tile_imgsz if tile_imgsz is not None else max(self.tile_width, self.tile_height)

        all_raw_boxes: List[BoundingBox] = []
        total_inf_time = 0.0

        # Optional full-image baseline pass (to catch macroscopic objects)
        if self.include_full_image:
            full_res = self.base_detector.predict(
                image,
                conf_threshold=conf_threshold,
                imgsz=max(img_w, img_h),
            )
            total_inf_time += full_res.inference_time_ms
            all_raw_boxes.extend(full_res.predictions)

        # Generate deterministic tile coordinates
        tiles = self.tiler.generate_tiles(img_w, img_h)

        # Extract tile crops
        tile_crops: List[np.ndarray] = []
        valid_tiles: List[TileCrop] = []
        for tile in tiles:
            crop = image[tile.ymin:tile.ymax, tile.xmin:tile.xmax]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                tile_crops.append(crop)
                valid_tiles.append(tile)

        # Execute tile inference in batches
        num_crops = len(tile_crops)
        for i in range(0, num_crops, self.batch_size):
            batch_slice = tile_crops[i : i + self.batch_size]
            batch_tiles = valid_tiles[i : i + self.batch_size]

            batch_results = self.base_detector.predict_batch(
                batch_slice,
                conf_threshold=conf_threshold,
                imgsz=inference_crop_size,
            )

            for crop_res, tile in zip(batch_results, batch_tiles):
                total_inf_time += crop_res.inference_time_ms

                # Coordinate remapping: convert tile-relative coordinates to full-image frame
                for b in crop_res.predictions:
                    global_box = BoundingBox(
                        xmin=b.xmin + tile.xmin,
                        ymin=b.ymin + tile.ymin,
                        xmax=b.xmax + tile.xmin,
                        ymax=b.ymax + tile.ymin,
                        class_id=b.class_id,
                        score=b.score,
                        class_name=b.class_name,
                    ).clip(img_w, img_h)

                    # Only preserve strictly valid boxes (width > 0, height > 0)
                    if global_box.width > 0.0 and global_box.height > 0.0:
                        all_raw_boxes.append(global_box)

        # Detection fusion across tile seams
        post_start = time.perf_counter()
        if not all_raw_boxes:
            fused_boxes: List[BoundingBox] = []
        elif self.fusion_method == "soft_nms":
            fused_boxes = soft_nms(
                all_raw_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
            )
        elif self.fusion_method == "wbf":
            fused_boxes = weighted_boxes_fusion(
                all_raw_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
                img_width=img_w,
                img_height=img_h,
            )
        else:  # default standard nms
            fused_boxes = standard_nms(
                all_raw_boxes,
                iou_threshold=self.fusion_iou_threshold,
                conf_threshold=conf_threshold,
            )

        postprocess_ms = (time.perf_counter() - post_start) * 1000.0

        return ImageDetectionResult(
            image_id=img_id,
            image_path=image_input if isinstance(image_input, str) else None,
            width=img_w,
            height=img_h,
            predictions=fused_boxes,
            inference_time_ms=total_inf_time,
            postprocess_time_ms=postprocess_ms,
            metadata={
                "mode": "tiled" if not self.include_full_image else "tiled_with_full",
                "tile_width": self.tile_width,
                "tile_height": self.tile_height,
                "overlap_x": self.overlap_x,
                "overlap_y": self.overlap_y,
                "num_tiles": len(tiles),
                "raw_detections": len(all_raw_boxes),
                "fused_detections": len(fused_boxes),
                "fusion_method": self.fusion_method,
            },
        )
