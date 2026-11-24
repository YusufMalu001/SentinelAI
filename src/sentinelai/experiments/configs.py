"""Experiment configuration schema, parser, and registry."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import yaml
import os


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    mode: str = "tiled"  # "whole", "tiled", "multiscale"
    imgsz: int = 800
    tile_size: Optional[int] = 416
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    overlap: float = 0.25
    overlap_x: Optional[float] = None
    overlap_y: Optional[float] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.50
    fusion_method: str = "nms"  # "nms", "soft_nms", "wbf"
    scales: List[int] = field(default_factory=lambda: [640, 800, 1024])
    include_full_image: bool = True
    batch_size: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for dataset split and metric evaluation."""
    split: str = "validation"  # "validation", "test"
    iou_threshold: float = 0.50
    use_relative_size: bool = True
    max_samples: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default_experiment"
    seed: int = 42
    model_path: str = "runs/detect/yolov8n_epochs50_batch16/weights/best.pt"
    device: str = "cpu"
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiates ExperimentConfig from a dictionary."""
        exp_meta = data.get("experiment", {})
        inf_meta = data.get("inference", {})
        eval_meta = data.get("evaluation", {})

        name = exp_meta.get("name", data.get("name", "experiment"))
        seed = exp_meta.get("seed", data.get("seed", 42))
        model_path = data.get("model_path", "runs/detect/yolov8n_epochs50_batch16/weights/best.pt")
        device = data.get("device", "cpu")

        inf_cfg = InferenceConfig(
            mode=inf_meta.get("mode", "tiled"),
            imgsz=inf_meta.get("imgsz", 800),
            tile_size=inf_meta.get("tile_size", 416),
            tile_width=inf_meta.get("tile_width"),
            tile_height=inf_meta.get("tile_height"),
            overlap=inf_meta.get("overlap", inf_meta.get("overlap_ratio", 0.25)),
            overlap_x=inf_meta.get("overlap_x"),
            overlap_y=inf_meta.get("overlap_y"),
            confidence_threshold=inf_meta.get("confidence_threshold", 0.25),
            iou_threshold=inf_meta.get("iou_threshold", 0.50),
            fusion_method=inf_meta.get("fusion_method", "nms"),
            scales=inf_meta.get("scales", [640, 800, 1024]),
            include_full_image=inf_meta.get("include_full_image", True),
            batch_size=inf_meta.get("batch_size", 1),
        )

        eval_cfg = EvaluationConfig(
            split=eval_meta.get("split", "validation"),
            iou_threshold=eval_meta.get("iou_threshold", 0.50),
            use_relative_size=eval_meta.get("use_relative_size", True),
            max_samples=eval_meta.get("max_samples"),
        )

        return cls(
            name=name,
            seed=seed,
            model_path=model_path,
            device=device,
            inference=inf_cfg,
            evaluation=eval_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Loads and parses an experiment YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
