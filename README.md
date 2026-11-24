# SentinelAI: Small-Object Detection in Aerial and Satellite Imagery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 18 Passing](https://img.shields.io/badge/tests-18%20passed-green.svg)](tests/)

---

## 1. Research Overview

**SentinelAI** is an applied computer vision research repository investigating small-object detection in optical remote sensing imagery (DIOR dataset) using YOLOv8-based detection and controlled inference-time algorithmic modifications.

Rather than modifying model weights or building an application UI, this project isolates and evaluates whether inference-time strategies—**high-resolution inference**, **spatial sliding-window tiling**, **multi-scale test-time augmentation**, **alternative bounding-box fusion** (Standard NMS, Soft-NMS, Weighted Boxes Fusion), **confidence-threshold optimization**, and **object-size/density-aware evaluation**—improve detection quality on small aerial targets and how those gains trade off against computational cost.

---

## 2. Research Question

> *Can spatial tiling, multi-scale inference, and improved detection fusion improve small-object detection performance in aerial imagery while maintaining a reasonable computational cost?*

---

## 3. Motivation

In high-altitude optical remote sensing imagery ($800 \times 800\text{ px}$), objects such as vehicles, ships, storage tanks, and windmills frequently occupy $\le 32^2\text{ px}^2$ ($< 0.16\%$ of image area). Standard single-pass YOLOv8n inference downsamples the input by a factor of $32\times$ at the deepest feature pyramid level ($P_5$), reducing a $20 \times 20\text{ px}$ vehicle to $< 1\text{ feature pixel}$.

In the inherited baseline (`runs/detect/yolov8n_val_on_test3`), this manifests as severe recall degradation on small or elongated classes:
- `vehicle` (21,144 instances): Recall = **0.383**, mAP@0.5 = **0.520**
- `bridge` (2,061 instances): Recall = **0.330**, mAP@0.5 = **0.425**
- `storagetank` (18,322 instances): Recall = **0.526**, mAP@0.5 = **0.713**

---

## 4. Existing Baseline

The repository was initialized from an existing YOLOv8n + DIOR notebook (`dior_object_detection_satellite_imagery_yolov8.ipynb`) and its saved training artifacts (`runs/detect/yolov8n_epochs50_batch16/weights/best.pt`).

- **Inherited Training Configuration**: YOLOv8n trained from scratch (`yolov8n.yaml`, `pretrained: false`) for 50 epochs, batch size 16, `imgsz: 800`, SGD (`lr0: 0.01`, `momentum: 0.937`), on 11,725 DIOR training images.
- **Inherited Reported Test Results** (`runs/detect/yolov8n_val_on_test3`, $N = 9,391$ images, $98,630$ instances, Quadro P5000 GPU):
  - **Precision**: `0.817` | **Recall**: `0.632` | **mAP@0.5**: `0.710` | **mAP@0.5:0.95**: `0.485` | **GPU Inference**: `4.6 ms/img`

---

## 5. Original Contributions Implemented in This Repository

1. **Modular Research Pipeline** (`src/sentinelai/`): Decoupled dataset protocols, inference wrappers, postprocessing fusion, evaluation metrics, error analysis, and experiment runner.
2. **Deterministic Spatial Tiling** (`SpatialTiler`, `TiledDetector`): Configurable rectangular tile dimensions, independent horizontal/vertical overlap ratios, backward boundary-shift coverage, batched crop inference, and global coordinate remapping with clipping.
3. **Tile-Aware Detection Fusion** (`fusion.py`): Class-aware implementations of Standard Greedy NMS, Soft-NMS (Gaussian and Linear decay), and Weighted Boxes Fusion (WBF) with empty-detection handling.
4. **Multi-Scale Inference** (`MultiScaleDetector`): Configurable scale ensembles with deduplication and per-scale latency profiling.
5. **Validation-Isolated Threshold Optimization** (`ThresholdEvaluator`): Precision/Recall/F1 sweeping and PR-curve generation strictly on validation data.
6. **Size- & Density-Stratified Evaluation** (`ObjectSizeAnalyzer`, `ObjectDensityAnalyzer`): Normalized-area size bins and object-count density bins.
7. **Diagnostic Error Analysis & Automated Reporting** (`DetectionErrorAnalyzer`, `ExperimentReportGenerator`, `plots.py`): Class-aware IoU matching, 6 error archetypes, boundary/IoU sensitivity, comparative 3-panel visualizations, and automated Markdown trade-off reports.
8. **Fair Benchmarking Harness** (`LatencyBenchmark`): In-memory image preloading (excluding disk I/O), warmup passes, mean/median/p95 latency, and hardware/VRAM logging.

---

## 6. Experimental Methodology

- **Controlled Sample Evaluation**: All comparative experiments (Exp 0–4) are evaluated on identical image sets with fixed random seeds (`seed: 42`).
- **Split Isolation**: Confidence threshold optimization (Exp 5) is restricted to validation data; test evaluation uses fixed thresholds.
- **Fair Latency Profiling**: Models are warmed up (`warmup_runs: 2`) and evaluated on preloaded in-memory image buffers to separate inference/fusion compute from disk I/O.

---

## 7. Dataset

- **Dataset**: DIOR (Optical Remote Sensing Dataset, 20 object classes, $800 \times 800\text{ px}$).
- **Local Repository Status**: The full DIOR `JPEGImages` and XML/YOLO annotation directories are not committed to the repository. Local verification and preliminary benchmarks use the 33 sample test images staged in `data/benchmark_eval/` (sourced from `runs/detect/predict/` and `runs/detect/predict3/`).

---

## 8. Model & Baseline Configuration

- **Checkpoint**: `runs/detect/yolov8n_epochs50_batch16/weights/best.pt`
- **Default Inference**: `imgsz: 800`, `conf_threshold: 0.25`, `iou_threshold: 0.70`
- **Experiment Configs**: Stored as YAML files in `configs/` (`baseline_whole_800.yaml`, `tiled_416_overlap_25.yaml`, `tiled_wbf.yaml`, `multiscale_ensemble.yaml`).

---

## 9. Algorithms Investigated

| Algorithm | Module | Mechanism | Key Trade-off |
| :--- | :--- | :--- | :--- |
| **Whole-Image Baseline** | `YOLOv8Detector` | Single forward pass at $800\text{px}$ | Lowest latency; loses sub-32px targets |
| **Resolution Sweep** | `YOLOv8Detector` | Single pass at $640, 800, 1024\text{px}$ | $1024\text{px}$ increases latency without retraining |
| **Spatial Tiling** | `SpatialTiler` + `TiledDetector` | Overlapping sliding-window crops + remapping | Preserves small-object pixels; $3\text{--}4\times$ latency |
| **Standard NMS** | `standard_nms` | Hard greedy suppression above IoU threshold | Low duplicate FPs; can suppress crowded neighbors |
| **Soft-NMS** | `soft_nms` | Gaussian/Linear score decay by IoU | Retains crowded objects; higher seam duplicate FPs |
| **Weighted Boxes Fusion** | `weighted_boxes_fusion` | Confidence-weighted coordinate averaging | Smooths box coordinates across tile seams |
| **Multi-Scale TTA** | `MultiScaleDetector` | Ensemble across `[640, 800, 1024]` + fusion | Multi-scale consensus; $3.6\times$ latency |

---

## 10. Evaluation Metrics

- **Detection Accuracy**: All-point interpolated AP, mAP@0.5, COCO mAP@0.5:0.95, Per-class AP.
- **Size-Stratified Metrics** (`ObjectSizeAnalyzer`):
  - **Small**: Normalized area $A_{rel} \le 0.0016$ ($\le 32^2\text{ px}^2$ at $800 \times 800$)
  - **Medium**: $0.0016 < A_{rel} \le 0.0144$ ($32^2 < \text{area} \le 96^2$)
  - **Large**: $A_{rel} > 0.0144$ ($> 96^2$)
- **Density-Stratified Metrics** (`ObjectDensityAnalyzer`): Sparse ($0\text{--}3$), Moderate ($4\text{--}10$), Dense ($11+$ objects/image).
- **Computational Cost**: Mean/Median/P95 latency (ms/img), throughput (FPS), peak GPU VRAM (when CUDA is available).

---

## 11. Experimental Results

> **Provenance Notice**:
> - **Original Reported Baseline**: From `runs/detect/yolov8n_val_on_test3` on the full DIOR test split ($N = 9,391$ images, Quadro P5000 GPU).
> - **Preliminary Local Benchmarks**: Executed on CPU (`torch 2.15.0+cpu`) over 10 sample images from `data/benchmark_eval/` using baseline detections ($\text{conf} \ge 0.35$) as reference targets (`runs/experiments/master_benchmark.json`). Full DIOR test-set evaluation of experimental methods is **pending** dataset mounting.

| Method | Provenance | mAP@0.5 | mAP@0.5:0.95 | Small mAP@0.5 | Duplicate FPs | Latency (ms/img) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Original Baseline (GPU)** | Inherited (`val_on_test3`) | `0.710` | `0.485` | — | — | `4.6` (GPU) | Full DIOR test ($N=9,391$) |
| **Exp 0: Baseline 800px** | Preliminary (CPU, $N=10$) | `0.3000` | `0.3000` | `0.0500` | `0` | `166.49` | Whole-image inference |
| **Exp 1a: Low-Res 640px** | Preliminary (CPU, $N=10$) | `0.2417` | `0.1956` | — | — | `101.37` | Reduced input resolution |
| **Exp 1b: High-Res 1024px** | Preliminary (CPU, $N=10$) | `0.2875` | `0.2437` | — | — | `273.78` | Increased input resolution |
| **Exp 2a: Tiled 416 (15% ov)** | Preliminary (CPU, $N=10$) | `0.2475` | `0.2365` | `0.0250` | — | `610.91` | Sliding window + NMS |
| **Exp 2b: Tiled 416 (25% ov)** | Preliminary (CPU, $N=10$) | `0.2750` | `0.2400` | `0.0500` | — | `613.01` | Sliding window + NMS |
| **Exp 2c: Tiled 512 (20% ov)** | Preliminary (CPU, $N=10$) | `0.2917` | `0.2383` | `0.0500` | — | `496.25` | 4-tile Pareto compromise |
| **Exp 3a: Tiled + NMS** | Preliminary (CPU, $N=10$) | `0.2750` | `0.2400` | `0.0500` | `1` | `631.53` | 416px, 25% overlap |
| **Exp 3b: Tiled + Soft-NMS** | Preliminary (CPU, $N=10$) | `0.2600` | `0.2250` | `0.0500` | `5` | `660.49` | Higher seam duplicates |
| **Exp 3c: Tiled + WBF** | Preliminary (CPU, $N=10$) | `0.2667` | `0.2158` | `0.0500` | `2` | `748.31` | Weighted box consensus |
| **Exp 4: Multi-Scale + WBF** | Preliminary (CPU, $N=10$) | `0.3000` | `0.2817` | `0.0500` | — | `601.21` | Scales `[640, 800, 1024]` |

---

## 12. Ablation Studies, Error Analysis & Accuracy-vs-Cost

- **Ablation & Error Analysis**: See [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) and [docs/COMPARATIVE_ANALYSIS_REPORT.md](docs/COMPARATIVE_ANALYSIS_REPORT.md).
- **Generated Figures** (`docs/figures/`):
  - `accuracy_vs_latency.png`, `size_stratified_performance.png`, `density_performance.png`, `class_wise_ap.png`, `confidence_distribution.png`, `comparative_diagnostics.png`, `sample_tiled_wbf_comparison.png`.

---

## 13. Repository Structure

```
SentinelAI/
├── configs/                  # YAML experiment configurations
├── data/benchmark_eval/      # Staged sample evaluation images (33 images)
├── docs/
│   ├── architecture.md       # System architecture & component diagram
│   ├── EXPERIMENTS.md        # Experimental protocols & CLI guide
│   ├── RESEARCH_REPORT.md    # Research report & baseline audit
│   ├── COMPARATIVE_ANALYSIS_REPORT.md
│   └── figures/              # Generated research plots
├── runs/
│   ├── detect/               # Inherited baseline weights & training logs
│   └── experiments/          # JSON results & experiments_summary.csv
├── src/sentinelai/
│   ├── dataset/              # Constants, structures, YOLO loader
│   ├── inference/            # YOLOv8Detector, SpatialTiler, TiledDetector, MultiScaleDetector
│   ├── postprocessing/       # standard_nms, soft_nms, weighted_boxes_fusion
│   ├── evaluation/           # metrics, size_analysis, density_analysis, error_analysis, threshold_optimizer, report_generator
│   ├── experiments/          # configs.py, runner.py
│   ├── visualization/        # plots.py
│   ├── utils/                # benchmark.py
│   └── cli.py                # CLI entrypoints
├── tests/                    # 18 unit & integration tests (test_core.py, test_error_analysis.py)
└── pyproject.toml
```

---

## 14. Installation & Dataset Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To evaluate on the full DIOR dataset, place images and YOLO-format `.txt` labels under `datasets/images/{train,val,test}` and `datasets/labels/{train,val,test}` matching `config.yaml` and `test_config.yaml`.

---

## 15. Verified Commands (Tests, Audit, Evaluation, Experiments)

```bash
# Run all 18 unit and integration tests
.venv/bin/pytest tests/

# Audit inherited baseline metrics and failure modes
.venv/bin/python3 -m sentinelai.cli audit

# List registered experiment recipes
.venv/bin/python3 -m sentinelai.cli list-experiments

# Run experiments from YAML configs
.venv/bin/python3 -m sentinelai.cli run-exp --config configs/baseline_whole_800.yaml
.venv/bin/python3 -m sentinelai.cli run-exp --config configs/tiled_416_overlap_25.yaml
.venv/bin/python3 -m sentinelai.cli run-exp --config configs/tiled_wbf.yaml
.venv/bin/python3 -m sentinelai.cli run-exp --config configs/multiscale_ensemble.yaml

# Run fair in-memory latency benchmark
.venv/bin/python3 -m sentinelai.cli benchmark --mode tiled --tile-size 416 --overlap 0.25 --fusion wbf

# Sweep confidence thresholds on validation images
.venv/bin/python3 -m sentinelai.cli sweep-thresholds --images-dir data/benchmark_eval --out-plot docs/figures/threshold_curves.png
```

---

## 16. Limitations & Future Work

- **Dataset Availability**: Full DIOR test-set ground-truth annotations are not bundled in this repository; local comparative runs use high-confidence baseline detections as reference targets.
- **CPU Benchmarking**: Local latency numbers were measured on CPU; CUDA GPU throughput and `peak_gpu_memory_mb` require execution on a GPU host.
- **Future Work**: Execute full-dataset DIOR evaluation on GPU, investigate SAHI-style adaptive slicing, and evaluate small-object-focused training augmentations (e.g., mosaic scaling).

---

## 17. References & Attribution

- **Original Repository**: Adapted from [YusufMalu001/SentinelAI](https://github.com/YusufMalu001/SentinelAI) (`dior_object_detection_satellite_imagery_yolov8.ipynb`, baseline weights `runs/detect/yolov8n_epochs50_batch16/weights/best.pt`).
- **Dataset**: Li et al., *Object Detection in Optical Remote Sensing Images: A Survey and a New Benchmark (DIOR)*, ISPRS Journal of Photogrammetry and Remote Sensing, 2020.
- **Model Framework**: Ultralytics YOLOv8 ([ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)).
- **Fusion Algorithms**:
  - Bodla et al., *Soft-NMS — Improving Object Detection With One Line of Code*, ICCV 2017.
  - Solovyev et al., *Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models*, Image and Vision Computing, 2021.
