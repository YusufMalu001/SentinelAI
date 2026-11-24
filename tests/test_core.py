"""Comprehensive unit and integration test suite for SentinelAI core algorithms."""

import os
import json
import csv
import pytest
import numpy as np

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult, TileCrop
from sentinelai.inference.tiler import SpatialTiler
from sentinelai.postprocessing.fusion import (
    compute_iou_matrix,
    standard_nms,
    soft_nms,
    weighted_boxes_fusion,
    _pure_wbf,
)
from sentinelai.evaluation.metrics import compute_ap, evaluate_detections_at_iou, compute_coco_metrics
from sentinelai.evaluation.error_analysis import DetectionErrorAnalyzer
from sentinelai.evaluation.threshold_optimizer import ThresholdEvaluator
from sentinelai.evaluation.size_analysis import ObjectSizeAnalyzer
from sentinelai.experiments.configs import ExperimentConfig
from sentinelai.experiments.runner import ExperimentRunner


# ==============================================================================
# 1. Bounding Box Geometry & Coordinate Operations
# ==============================================================================

def test_bounding_box_geometry():
    box = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=60.0, class_id=2, score=0.9, class_name="airplane")
    assert box.width == 40.0
    assert box.height == 40.0
    assert box.area == 1600.0
    assert box.center == (30.0, 40.0)
    assert box.as_xyxy() == [10.0, 20.0, 50.0, 60.0]
    assert box.as_normalized_xyxy(100, 100) == [0.1, 0.2, 0.5, 0.6]

    clipped = box.clip(img_w=30, img_h=30)
    assert clipped.xmin == 10.0
    assert clipped.ymin == 20.0
    assert clipped.xmax == 30.0
    assert clipped.ymax == 30.0


# ==============================================================================
# 2. Sliding-Window Tiling & Boundary Handling
# ==============================================================================

def test_spatial_tiler_coverage_square():
    tiler = SpatialTiler(tile_size=400, overlap_ratio=0.20)
    tiles = tiler.generate_tiles(img_width=800, img_height=800)

    assert len(tiles) >= 4
    covered_mask = np.zeros((800, 800), dtype=bool)
    for t in tiles:
        assert t.xmin >= 0 and t.ymin >= 0
        assert t.xmax <= 800 and t.ymax <= 800
        assert t.width <= 400 and t.height <= 400
        covered_mask[t.ymin:t.ymax, t.xmin:t.xmax] = True

    assert np.all(covered_mask), "All pixels must be covered by generated tiles"


def test_spatial_tiler_rectangular_and_edges():
    # Non-square image (1000 x 650) with rectangular tiles (400 x 300)
    tiler = SpatialTiler(tile_width=400, tile_height=300, overlap_x=0.25, overlap_y=0.20)
    tiles = tiler.generate_tiles(img_width=1000, img_height=650)

    assert len(tiles) > 0
    covered_mask = np.zeros((650, 1000), dtype=bool)
    tile_coords = set()

    for t in tiles:
        coord_tuple = (t.xmin, t.ymin, t.xmax, t.ymax)
        assert coord_tuple not in tile_coords, f"Duplicate tile detected: {coord_tuple}"
        tile_coords.add(coord_tuple)

        # Boundary checks
        assert 0 <= t.xmin < t.xmax <= 1000
        assert 0 <= t.ymin < t.ymax <= 650
        assert t.width == 400  # Shifted tiles preserve width
        assert t.height == 300  # Shifted tiles preserve height

        covered_mask[t.ymin:t.ymax, t.xmin:t.xmax] = True

    # Ensure final row and column are completely covered
    assert np.all(covered_mask[:, -1]), "Final column must be covered"
    assert np.all(covered_mask[-1, :]), "Final row must be covered"
    assert np.all(covered_mask), "All pixels in rectangular image must be covered"


def test_spatial_tiler_small_image():
    # Image smaller than tile dimension
    tiler = SpatialTiler(tile_width=512, tile_height=512)
    tiles = tiler.generate_tiles(img_width=250, img_height=300)
    assert len(tiles) == 1
    t = tiles[0]
    assert t.xmin == 0 and t.ymin == 0
    assert t.xmax == 250 and t.ymax == 300


# ==============================================================================
# 3. Coordinate Remapping & Boundary Clipping
# ==============================================================================

def test_coordinate_remapping_from_tile():
    tile = TileCrop(tile_id=1, xmin=200, ymin=300, xmax=600, ymax=700)
    tile_box = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0, class_id=0, score=0.95)

    img_w, img_h = 800, 800
    global_box = BoundingBox(
        xmin=tile_box.xmin + tile.xmin,
        ymin=tile_box.ymin + tile.ymin,
        xmax=tile_box.xmax + tile.xmin,
        ymax=tile_box.ymax + tile.ymin,
        class_id=tile_box.class_id,
        score=tile_box.score,
    ).clip(img_w, img_h)

    assert global_box.xmin == 210.0
    assert global_box.ymin == 320.0
    assert global_box.xmax == 250.0
    assert global_box.ymax == 380.0


# ==============================================================================
# 4. IoU Matrix & Fusion Algorithms
# ==============================================================================

def test_compute_iou_matrix():
    b1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)

    ious = compute_iou_matrix(b1, b2)[0]
    assert pytest.approx(ious[0], 0.01) == 1.0  # Perfect overlap
    assert pytest.approx(ious[1], 0.01) == 25.0 / 175.0
    assert ious[2] == 0.0  # Disjoint


def test_standard_nms():
    boxes = [
        BoundingBox(0, 0, 10, 10, class_id=0, score=0.9),
        BoundingBox(1, 1, 10, 10, class_id=0, score=0.8),
        BoundingBox(50, 50, 60, 60, class_id=0, score=0.7),
        BoundingBox(0, 0, 10, 10, class_id=1, score=0.85),
    ]

    kept = standard_nms(boxes, iou_threshold=0.5, conf_threshold=0.1)
    assert len(kept) == 3
    scores = [b.score for b in kept]
    assert 0.8 not in scores


def test_soft_nms():
    boxes = [
        BoundingBox(0, 0, 10, 10, class_id=0, score=0.9),
        BoundingBox(1, 1, 10, 10, class_id=0, score=0.8),
    ]
    kept = soft_nms(boxes, iou_threshold=0.5, conf_threshold=0.05, method="gaussian")
    assert len(kept) == 2
    assert pytest.approx(kept[0].score, 1e-4) == 0.9
    assert kept[1].score < 0.8


def test_weighted_boxes_fusion():
    boxes = [
        BoundingBox(0, 0, 10, 10, class_id=0, score=0.8, class_name="vehicle"),
        BoundingBox(2, 2, 12, 12, class_id=0, score=0.8, class_name="vehicle"),
    ]
    fused = _pure_wbf(boxes, iou_threshold=0.4, conf_threshold=0.1, img_width=100, img_height=100)
    assert len(fused) == 1
    assert pytest.approx(fused[0].xmin, 0.05) == 1.0
    assert pytest.approx(fused[0].ymin, 0.05) == 1.0
    assert pytest.approx(fused[0].xmax, 0.05) == 11.0
    assert pytest.approx(fused[0].ymax, 0.05) == 11.0


# ==============================================================================
# 5. Empty Detections Handling
# ==============================================================================

def test_empty_detections():
    empty_list = []
    assert standard_nms(empty_list) == []
    assert soft_nms(empty_list) == []
    assert weighted_boxes_fusion(empty_list) == []
    assert _pure_wbf(empty_list) == []


# ==============================================================================
# 6. Confidence Threshold Sweeping
# ==============================================================================

def test_confidence_threshold_evaluator():
    evaluator = ThresholdEvaluator(threshold_range=(0.1, 0.9), step=0.1, default_threshold=0.25)
    assert len(evaluator.thresholds) == 9

    # Synthetic detections with scores [0.2, 0.5, 0.8]
    gt = [
        BoundingBox(0, 0, 20, 20, class_id=0, score=1.0),
        BoundingBox(50, 50, 70, 70, class_id=0, score=1.0),
    ]
    preds = [
        BoundingBox(0, 0, 20, 20, class_id=0, score=0.8),  # High confidence TP
        BoundingBox(50, 50, 70, 70, class_id=0, score=0.4),  # Mid confidence TP
        BoundingBox(200, 200, 220, 220, class_id=0, score=0.15),  # Low confidence FP
    ]
    res = [ImageDetectionResult(image_id="img1", predictions=preds, ground_truths=gt, width=800, height=800)]

    sweep = evaluator.sweep_thresholds(res)
    assert "sweep_curve" in sweep
    assert "optimal_val_threshold" in sweep
    assert sweep["default_threshold"] == 0.25


# ==============================================================================
# 7. Object-Size Categorization & Metrics
# ==============================================================================

def test_object_size_analyzer():
    analyzer = ObjectSizeAnalyzer(use_relative_area=True)

    img_w, img_h = 800, 800
    img_area = 800 * 800  # 640000

    # Small: area <= 0.0016 * 640000 = 1024 -> 30x30 = 900
    small_box = BoundingBox(0, 0, 30, 30, class_id=0, score=0.9)
    # Medium: 50x50 = 2500 -> 2500 / 640000 = 0.0039
    medium_box = BoundingBox(0, 0, 50, 50, class_id=0, score=0.9)
    # Large: 120x120 = 14400 -> 14400 / 640000 = 0.0225
    large_box = BoundingBox(0, 0, 120, 120, class_id=0, score=0.9)

    assert analyzer.categorize_box(small_box, img_w, img_h) == "small"
    assert analyzer.categorize_box(medium_box, img_w, img_h) == "medium"
    assert analyzer.categorize_box(large_box, img_w, img_h) == "large"

    results = [
        ImageDetectionResult(
            image_id="test_size",
            width=img_w,
            height=img_h,
            ground_truths=[small_box, large_box],
            predictions=[small_box],
        )
    ]

    report = analyzer.analyze(results)
    assert "metrics_per_bin" in report
    assert report["metrics_per_bin"]["small"]["ground_truth_count"] == 1
    assert report["metrics_per_bin"]["small"]["prediction_count"] == 1
    assert report["metrics_per_bin"]["large"]["ground_truth_count"] == 1
    assert report["metrics_per_bin"]["large"]["prediction_count"] == 0


# ==============================================================================
# 8. Experiment Config Loading & Result Serialization
# ==============================================================================

def test_experiment_config_loading(tmp_path):
    yaml_content = """
experiment:
  name: test_exp_tiled
  seed: 99

inference:
  mode: tiled
  tile_size: 512
  overlap: 0.20
  confidence_threshold: 0.30
  iou_threshold: 0.45
  fusion_method: wbf

evaluation:
  split: validation
  iou_threshold: 0.50
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = ExperimentConfig.from_yaml(str(config_file))
    assert cfg.name == "test_exp_tiled"
    assert cfg.seed == 99
    assert cfg.inference.mode == "tiled"
    assert cfg.inference.tile_size == 512
    assert cfg.inference.overlap == 0.20
    assert cfg.inference.fusion_method == "wbf"
    assert cfg.evaluation.split == "validation"


def test_experiment_result_serialization(tmp_path):
    out_dir = tmp_path / "runs"
    runner = ExperimentRunner(
        experiment_id="test_serialization_exp",
        config={
            "name": "test_serialization_exp",
            "seed": 42,
            "inference": {"mode": "whole", "imgsz": 800},
            "evaluation": {"split": "validation"},
        },
        output_dir=str(out_dir),
    )

    dummy_res = [
        ImageDetectionResult(
            image_id="img1",
            width=800,
            height=800,
            ground_truths=[BoundingBox(0, 0, 20, 20, class_id=0, score=1.0)],
            predictions=[BoundingBox(0, 0, 20, 20, class_id=0, score=0.9)],
            inference_time_ms=50.0,
            postprocess_time_ms=2.0,
        )
    ]

    record = runner.evaluate_and_record(dummy_res)
    # Class 0 AP is 1.0; mAP averaged across all 20 DIOR classes is 1.0 / 20 = 0.05
    assert record["metrics"]["per_class_mAP50"]["Expressway-Service-area"] == 1.0
    assert record["metrics"]["mAP50"] == 0.05

    # Verify JSON existence
    json_path = out_dir / "test_serialization_exp" / "results.json"
    assert json_path.exists()
    with open(json_path) as f:
        loaded_json = json.load(f)
    assert loaded_json["experiment_id"] == "test_serialization_exp"
    assert "metrics" in loaded_json

    # Verify CSV summary existence and content
    csv_path = out_dir / "experiments_summary.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 1
    assert reader[-1]["experiment_id"] == "test_serialization_exp"
    assert float(reader[-1]["mAP50"]) == 0.05
