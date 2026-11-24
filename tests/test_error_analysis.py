"""Unit tests for error analysis, density stratification, comparative visualization, and report generation."""

import os
import pytest
import numpy as np

from sentinelai.dataset.structures import BoundingBox, ImageDetectionResult, TileCrop
from sentinelai.evaluation.error_analysis import DetectionErrorAnalyzer
from sentinelai.evaluation.density_analysis import ObjectDensityAnalyzer
from sentinelai.evaluation.report_generator import ExperimentReportGenerator
from sentinelai.visualization.plots import (
    plot_comparative_diagnostics,
    plot_confidence_distribution,
    plot_class_wise_ap,
    plot_density_performance,
)


def test_detection_matching_archetypes():
    analyzer = DetectionErrorAnalyzer(iou_threshold=0.50, bg_iou_threshold=0.10)

    gt = [
        BoundingBox(100, 100, 140, 140, class_id=0, score=1.0),  # Will be TP + Duplicate
        BoundingBox(300, 300, 340, 340, class_id=1, score=1.0),  # Will be Class Confusion
        BoundingBox(500, 500, 540, 540, class_id=2, score=1.0),  # Will be Missed (FN)
    ]

    preds = [
        BoundingBox(100, 100, 140, 140, class_id=0, score=0.95),  # TP
        BoundingBox(102, 102, 142, 142, class_id=0, score=0.80),  # Duplicate FP
        BoundingBox(300, 300, 340, 340, class_id=0, score=0.75),  # Class Confusion (wrong class)
        BoundingBox(700, 700, 740, 740, class_id=0, score=0.60),  # Background FP
    ]

    res = ImageDetectionResult(image_id="match_test", width=800, height=800, ground_truths=gt, predictions=preds)
    m = analyzer.match_image(res)

    assert len(m.true_positives) == 1
    assert len(m.duplicate_fps) == 1
    assert len(m.class_confusions) == 1
    assert len(m.background_fps) == 1
    assert len(m.false_negatives) == 2  # GT 1 (confused) and GT 2 (missed)


def test_iou_sensitivity_and_boundary_analysis():
    analyzer = DetectionErrorAnalyzer(iou_threshold=0.50, boundary_margin=20)
    tiles = [TileCrop(tile_id=0, xmin=0, ymin=0, xmax=400, ymax=400)]

    gt = [
        BoundingBox(5, 5, 25, 25, class_id=0, score=1.0),       # Image boundary (missed)
        BoundingBox(390, 200, 410, 220, class_id=0, score=1.0), # Tile boundary (detected)
        BoundingBox(200, 200, 240, 240, class_id=0, score=1.0), # Interior (detected)
    ]
    preds = [
        BoundingBox(390, 200, 410, 220, class_id=0, score=0.9),
        BoundingBox(200, 200, 240, 240, class_id=0, score=0.9),
    ]
    res = [ImageDetectionResult(image_id="b_test", width=800, height=800, ground_truths=gt, predictions=preds)]

    bound = analyzer.analyze_boundary_performance(res, tiles=tiles)
    assert bound["image_boundary"]["gt_count"] == 1
    assert bound["image_boundary"]["missed"] == 1
    assert bound["tile_boundary"]["gt_count"] == 1
    assert bound["tile_boundary"]["missed"] == 0
    assert bound["interior"]["gt_count"] == 1
    assert bound["interior"]["missed"] == 0

    sens = analyzer.analyze_iou_sensitivity(res, iou_thresholds=[0.3, 0.5, 0.75])
    assert len(sens) == 3
    assert sens[0]["tp"] == 2


def test_object_density_analyzer():
    analyzer = ObjectDensityAnalyzer(density_bins={"sparse": (0, 2), "moderate": (3, 5), "dense": (6, 100)})

    r_sparse = ImageDetectionResult(
        image_id="s1", width=800, height=800,
        ground_truths=[BoundingBox(10, 10, 30, 30, class_id=0, score=1.0)],
        predictions=[BoundingBox(10, 10, 30, 30, class_id=0, score=0.9)],
    )
    r_dense = ImageDetectionResult(
        image_id="d1", width=800, height=800,
        ground_truths=[BoundingBox(i * 50, 10, i * 50 + 20, 30, class_id=0, score=1.0) for i in range(7)],
        predictions=[BoundingBox(0, 10, 20, 30, class_id=0, score=0.9)],
    )

    report = analyzer.analyze([r_sparse, r_dense])
    assert report["categories"]["sparse"]["image_count"] == 1
    assert report["categories"]["sparse"]["recall"] == 1.0
    assert report["categories"]["dense"]["image_count"] == 1
    assert report["categories"]["dense"]["gt_count"] == 7
    assert pytest.approx(report["categories"]["dense"]["recall"], 0.01) == 1.0 / 7.0


def test_comparative_visualizations_and_report(tmp_path):
    base_res = ImageDetectionResult(
        image_id="vis1", width=400, height=400,
        ground_truths=[BoundingBox(50, 50, 100, 100, class_id=0, score=1.0)],
        predictions=[BoundingBox(200, 200, 250, 250, class_id=0, score=0.6)],
    )
    exp_res = ImageDetectionResult(
        image_id="vis1", width=400, height=400,
        ground_truths=[BoundingBox(50, 50, 100, 100, class_id=0, score=1.0)],
        predictions=[BoundingBox(50, 50, 100, 100, class_id=0, score=0.9)],
    )

    comp_path = tmp_path / "comp.png"
    conf_path = tmp_path / "conf.png"
    cls_path = tmp_path / "cls.png"
    dens_path = tmp_path / "dens.png"

    plot_comparative_diagnostics("nonexistent.jpg", base_res, exp_res, str(comp_path))
    plot_confidence_distribution([0.9, 0.85], [0.4, 0.3], str(conf_path))
    plot_class_wise_ap({"vehicle": 0.5, "ship": 0.8}, {"vehicle": 0.6, "ship": 0.85}, str(cls_path))
    plot_density_performance(
        {"categories": {"sparse": {"mAP50": 0.5}, "moderate": {"mAP50": 0.4}, "dense": {"mAP50": 0.3}}},
        {"categories": {"sparse": {"mAP50": 0.6}, "moderate": {"mAP50": 0.5}, "dense": {"mAP50": 0.45}}},
        str(dens_path),
    )

    assert comp_path.exists()
    assert conf_path.exists()
    assert cls_path.exists()
    assert dens_path.exists()

    # Test automated report generator
    gen = ExperimentReportGenerator()
    b_rec = {
        "experiment_id": "base", "sample_count": 10, "timestamp": "2026-11-25",
        "performance": {"latency_per_image_ms": 100.0, "throughput_fps": 10.0},
        "metrics": {"mAP50": 0.50, "mAP50_95": 0.30, "per_class_mAP50": {"vehicle": 0.40}, "size_metrics": {"small": {"mAP50": 0.20}, "medium": {"mAP50": 0.50}, "large": {"mAP50": 0.70}}},
        "error_analysis": {"tp": 10, "localization_errors": 2, "class_confusions": 1, "duplicate_fps": 0, "background_fps": 3, "missed_gts": 5},
    }
    e_rec = {
        "experiment_id": "exp", "sample_count": 10, "timestamp": "2026-11-25",
        "performance": {"latency_per_image_ms": 400.0, "throughput_fps": 2.5},
        "metrics": {"mAP50": 0.55, "mAP50_95": 0.34, "per_class_mAP50": {"vehicle": 0.48}, "size_metrics": {"small": {"mAP50": 0.30}, "medium": {"mAP50": 0.52}, "large": {"mAP50": 0.68}}},
        "error_analysis": {"tp": 12, "localization_errors": 1, "class_confusions": 1, "duplicate_fps": 2, "background_fps": 2, "missed_gts": 3},
    }
    rep_path = tmp_path / "report.md"
    text = gen.generate_comparison_report(b_rec, e_rec, str(rep_path))
    assert rep_path.exists()
    assert "Accuracy-Latency Trade-off" in text
    assert "Scale Trade-off" in text
    assert "Duplicate Detections" in text
