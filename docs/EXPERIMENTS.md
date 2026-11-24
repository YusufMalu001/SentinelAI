# SentinelAI Experimental Protocols & Execution Guide

This document details the experimental protocols, configurations, execution commands, and reproducibility rules for the SentinelAI research suite.

---

## 1. Experimental Protocols Overview

| Experiment ID | Title | Objective | Key Variables |
| :--- | :--- | :--- | :--- |
| **Exp 0** | Baseline Reproduction | Audit and establish baseline YOLOv8n single-scale inference metrics | Model weights: `best.pt`, Input size: $800 \times 800$, Split: Test |
| **Exp 1** | Resolution Sensitivity | Evaluate whole-image inference accuracy and latency across image resolutions | Resolutions: 640, 800, 1024, 1280 |
| **Exp 2** | Spatial Tiling & Overlap | Quantify small-object sensitivity and boundary truncation across tile sizes & overlaps | Tile sizes: 416, 512; Overlaps: 15%, 25%, 35% |
| **Exp 3** | Detection Fusion Comparison | Compare seam deduplication and box accuracy between NMS, Soft-NMS, and WBF | Fusion methods: NMS, Soft-NMS, WBF; Overlap IoU: 0.45 to 0.60 |
| **Exp 4** | Multi-Scale Inference | Evaluate test-time scale ensembling against single-scale baselines | Scales: [640, 800, 1024] |
| **Exp 5** | Threshold Optimization | Tune confidence thresholds on validation split to maximize F1 score without test leakage | Confidence threshold sweep: 0.05 to 0.85 in steps of 0.05 |
| **Exp 6** | Granular Error Analysis | Classify detection failures into 6 diagnostic archetypes across size bins | Object sizes: Small ($\le 32^2$), Medium, Large |

---

## 2. Configuration Schema

Every experiment is defined via a YAML file matching the schema below:

```yaml
experiment:
  name: tiled_416_overlap_25  # Unique experiment identifier
  seed: 42                    # Pseudo-random seed for reproducibility

inference:
  mode: tiled                 # Modes: 'whole', 'tiled', 'multiscale'
  imgsz: 800                  # Whole image inference resolution
  tile_size: 416              # Tile crop dimension
  tile_width: 416             # Optional rectangular tile width
  tile_height: 416            # Optional rectangular tile height
  overlap: 0.25               # Overlap ratio between adjacent tiles
  confidence_threshold: 0.25  # Minimum detection confidence
  iou_threshold: 0.50         # NMS / Fusion IoU threshold
  fusion_method: wbf          # Fusion algorithm: 'nms', 'soft_nms', 'wbf'
  include_full_image: true    # Whether to run a concurrent whole-image pass
  batch_size: 1               # Batch size for tile processing
  scales: [640, 800, 1024]    # Scale list for multiscale mode

evaluation:
  split: validation           # 'validation' or 'test'
  iou_threshold: 0.50         # Benchmark IoU threshold for AP calculation
  use_relative_size: true     # Use relative area vs pixel area for size analysis
  max_samples: 20             # Maximum evaluation samples (optional)
```

---

## 3. Step-by-Step Execution Commands

### 3.1 Audit Inherited Baseline
Inspect inherited baseline configurations, training hyperparameters, and per-class failure modes:
```bash
python3 -m sentinelai.cli audit
```

### 3.2 List All Registered Experiments
```bash
python3 -m sentinelai.cli list-experiments
```

### 3.3 Execute Standard Whole-Image Baseline (Exp 0)
```bash
python3 -m sentinelai.cli run-exp --config configs/baseline_whole_800.yaml
```

### 3.4 Execute Tiled Inference with NMS (Exp 2)
```bash
python3 -m sentinelai.cli run-exp --config configs/tiled_416_overlap_25.yaml
```

### 3.5 Execute Tiled Inference with Weighted Boxes Fusion (Exp 3)
```bash
python3 -m sentinelai.cli run-exp --config configs/tiled_wbf.yaml
```

### 3.6 Execute Multi-Scale Test-Time Augmentation (Exp 4)
```bash
python3 -m sentinelai.cli run-exp --config configs/multiscale_ensemble.yaml
```

### 3.7 Run Confidence Threshold Sweep & Curve Generation (Exp 5)
```bash
python3 -m sentinelai.cli sweep-thresholds --images-dir data/benchmark_eval --out-plot docs/figures/threshold_curves.png
```

### 3.8 Execute Pure Latency & Throughput Benchmark
Measures in-memory inference speed with warmups (excluding disk I/O):
```bash
# Whole-image benchmark
python3 -m sentinelai.cli benchmark --mode whole

# Tiled benchmark (416px, 25% overlap, WBF)
python3 -m sentinelai.cli benchmark --mode tiled --tile-size 416 --overlap 0.25 --fusion wbf

# Multi-scale benchmark
python3 -m sentinelai.cli benchmark --mode multiscale
```

---

## 4. Scientific Assumptions & Methodological Validity

1. **Split Isolation (Leakage Prevention)**:
   - Confidence threshold sweeps and hyperparameter selections (tile size, overlap ratio) are executed **exclusively on the validation split**.
   - The test set is evaluated only once with the chosen configuration.
2. **Deterministic Tile Generation**:
   - Tile boundaries are calculated deterministically without random offsets or jitter.
   - When the image size is larger than the tile size, the final column and row are shifted backwards to preserve identical receptive fields without padding.
3. **Identical Sample Control**:
   - All comparative experiments (Exp 0 through Exp 4) are evaluated on the exact same sample set.
   - Ground truth annotations and prediction inputs are identical across fusion methods.
4. **Fair Benchmarking Controls**:
   - All models are warmed up with 2 dummy passes before timing begins.
   - Timing measurements profile in-memory image buffers to isolate deep learning and fusion operations from disk read latencies.
   - Hardware specifications (OS, CPU model, RAM, CUDA device name, peak VRAM) are logged alongside results.

---

## 5. Output Artifacts

Execution produces structured artifacts under `runs/experiments/`:
- `runs/experiments/<name>/results.json`: Full machine-readable record including configuration, hardware profile, COCO metrics, per-class AP, size-stratified analysis, and error breakdowns.
- `runs/experiments/experiments_summary.csv`: Aggregated comparison table logging latency, throughput, mAP@0.5, mAP@0.5:0.95, small-object mAP, and duplicate false positive counts.
- `docs/figures/`: High-resolution figures including accuracy-vs-latency Pareto frontiers, size-stratified bar charts, and side-by-side detection comparisons.
