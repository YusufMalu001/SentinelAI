"""Experiment runner, pipeline factory, and structured JSON / CSV reporting."""

import os
import sys
import json
import csv
import time
import datetime
import random
from typing import Dict, Any, List, Optional, Union
import numpy as np

from sentinelai.dataset.structures import ImageDetectionResult
from sentinelai.evaluation.metrics import compute_coco_metrics
from sentinelai.evaluation.error_analysis import DetectionErrorAnalyzer
from sentinelai.evaluation.size_analysis import ObjectSizeAnalyzer
from sentinelai.experiments.configs import ExperimentConfig
from sentinelai.inference.detector import YOLOv8Detector, BaseDetector
from sentinelai.inference.tiled_inference import TiledDetector
from sentinelai.inference.multiscale_inference import MultiScaleDetector
from sentinelai.utils.benchmark import get_hardware_profile


class WholeImageWrapper(BaseDetector):
    """Wrapper for baseline whole-image inference matching TiledDetector interface."""

    def __init__(self, detector: YOLOv8Detector, imgsz: int = 800):
        self.detector = detector
        self.imgsz = imgsz

    def predict(self, image: Any, conf_threshold: float = 0.25) -> ImageDetectionResult:
        return self.detector.predict(image, conf_threshold=conf_threshold, imgsz=self.imgsz)


def build_detector_pipeline(config: ExperimentConfig) -> Any:
    """Builds the configured detector pipeline from an ExperimentConfig."""
    base_detector = YOLOv8Detector(model_path=config.model_path, device=config.device)
    inf = config.inference

    mode = inf.mode.lower()
    if mode in ["whole", "baseline", "single"]:
        return WholeImageWrapper(base_detector, imgsz=inf.imgsz)
    elif mode == "multiscale":
        return MultiScaleDetector(
            base_detector=base_detector,
            scales=inf.scales,
            fusion_method=inf.fusion_method,
            fusion_iou_threshold=inf.iou_threshold,
        )
    elif mode == "tiled":
        return TiledDetector(
            base_detector=base_detector,
            tile_width=inf.tile_width,
            tile_height=inf.tile_height,
            tile_size=inf.tile_size,
            overlap_x=inf.overlap_x,
            overlap_y=inf.overlap_y,
            overlap_ratio=inf.overlap,
            fusion_method=inf.fusion_method,
            fusion_iou_threshold=inf.iou_threshold,
            include_full_image=inf.include_full_image,
            batch_size=inf.batch_size,
        )
    else:
        raise ValueError(f"Unknown inference mode: '{mode}'. Supported: 'whole', 'tiled', 'multiscale'")


class ExperimentRunner:
    """Standardized experimental execution harness.

    Manages pipeline instantiation, execution across samples, COCO evaluation,
    size-stratified analysis, error diagnostics, and JSON/CSV summary persistence.
    """

    def __init__(
        self,
        experiment_id: str,
        config: Union[ExperimentConfig, Dict[str, Any]],
        output_dir: str = "runs/experiments",
    ):
        self.experiment_id = experiment_id
        if isinstance(config, ExperimentConfig):
            self.config = config
            self.config_dict = {
                "name": config.name,
                "seed": config.seed,
                "inference": config.inference.__dict__,
                "evaluation": config.evaluation.__dict__,
            }
        else:
            self.config = ExperimentConfig.from_dict(config)
            self.config_dict = config

        self.output_dir = os.path.join(output_dir, self.experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.error_analyzer = DetectionErrorAnalyzer()
        self.size_analyzer = ObjectSizeAnalyzer(use_relative_area=self.config.evaluation.use_relative_size)

        # Set reproducible seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def evaluate_and_record(
        self,
        results: List[ImageDetectionResult],
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculates benchmark metrics, size-stratified stats, errors, and saves JSON & CSV."""
        num_images = max(1, len(results))
        total_inf_time = sum(r.inference_time_ms for r in results)
        total_post_time = sum(r.postprocess_time_ms for r in results)

        avg_latency_ms = (total_inf_time + total_post_time) / num_images
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        coco_metrics = compute_coco_metrics(results)
        error_summary = self.error_analyzer.aggregate_error_breakdown(results)
        size_summary = self.size_analyzer.analyze(results)

        record: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config_dict,
            "environment": get_hardware_profile(),
            "sample_count": len(results),
            "performance": {
                "latency_per_image_ms": round(avg_latency_ms, 2),
                "throughput_fps": round(fps, 2),
                "inference_time_ms": round(total_inf_time / num_images, 2),
                "postprocess_time_ms": round(total_post_time / num_images, 2),
            },
            "metrics": {
                "mAP50": round(coco_metrics["mAP50"], 4),
                "mAP50_95": round(coco_metrics["mAP50_95"], 4),
                "per_class_mAP50": {k: round(v, 4) for k, v in coco_metrics["per_class_mAP50"].items()},
                "per_class_mAP50_95": {k: round(v, 4) for k, v in coco_metrics["per_class_mAP50_95"].items()},
                "size_metrics": coco_metrics["size_metrics"],
            },
            "size_stratified_analysis": size_summary,
            "error_analysis": error_summary,
        }

        if extra_metrics:
            record["extra_metrics"] = extra_metrics

        # Save JSON output
        result_file = os.path.join(self.output_dir, "results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Update Master CSV Table
        self._append_to_summary_csv(record)

        return record

    def _append_to_summary_csv(self, record: Dict[str, Any]) -> None:
        """Appends or updates a row in the master comparison CSV table."""
        summary_csv_path = os.path.join(os.path.dirname(self.output_dir), "experiments_summary.csv")

        csv_headers = [
            "experiment_id",
            "timestamp",
            "mode",
            "resolution_or_tilesize",
            "overlap",
            "fusion_method",
            "latency_ms",
            "throughput_fps",
            "mAP50",
            "mAP50_95",
            "small_mAP50",
            "medium_mAP50",
            "large_mAP50",
            "duplicate_fps",
            "sample_count",
        ]

        inf_meta = self.config_dict.get("inference", {})
        mode = inf_meta.get("mode", "unknown")
        res_or_tile = inf_meta.get("tile_size") if mode == "tiled" else inf_meta.get("imgsz", 800)
        overlap = inf_meta.get("overlap", 0.0) if mode == "tiled" else 0.0
        fusion = inf_meta.get("fusion_method", "none")

        row = {
            "experiment_id": self.experiment_id,
            "timestamp": record["timestamp"],
            "mode": mode,
            "resolution_or_tilesize": res_or_tile,
            "overlap": overlap,
            "fusion_method": fusion,
            "latency_ms": record["performance"]["latency_per_image_ms"],
            "throughput_fps": record["performance"]["throughput_fps"],
            "mAP50": record["metrics"]["mAP50"],
            "mAP50_95": record["metrics"]["mAP50_95"],
            "small_mAP50": record["metrics"]["size_metrics"].get("small", {}).get("mAP50", 0.0),
            "medium_mAP50": record["metrics"]["size_metrics"].get("medium", {}).get("mAP50", 0.0),
            "large_mAP50": record["metrics"]["size_metrics"].get("large", {}).get("mAP50", 0.0),
            "duplicate_fps": record["error_analysis"].get("duplicate_fps", 0),
            "sample_count": record["sample_count"],
        }

        # Read existing rows to allow clean updates
        rows: List[Dict[str, Any]] = []
        if os.path.exists(summary_csv_path):
            try:
                with open(summary_csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r.get("experiment_id") != self.experiment_id:
                            rows.append(r)
            except Exception:
                rows = []

        rows.append(row)

        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(rows)
