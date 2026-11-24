# Automated Experiment Analysis Report: `analysis_tiled_wbf` vs `analysis_baseline`

## 1. Configuration & Experimental Context
- **Baseline ID**: `analysis_baseline`
- **Experimental ID**: `analysis_tiled_wbf`
- **Dataset Split**: `validation`
- **Sample Count**: 10 images
- **Timestamp**: 2026-11-24T19:27:22.230041+00:00

## 2. Overall Metrics & Computational Cost
| Metric | Baseline | Experimental | Delta |
| :--- | :---: | :---: | :---: |
| **mAP@0.5** | 0.3000 | 0.2667 | -0.0333 |
| **mAP@0.5:0.95** | 0.3000 | 0.2158 | -0.0842 |
| **Latency (ms/img)** | 175.36 | 698.77 | 3.98x |
| **Throughput (FPS)** | 5.70 | 1.43 | - |

## 3. Trade-off Analysis
- Computational Overhead without Global Gain: Latency increased by 3.98x (175.4ms -> 698.8ms) while overall mAP@0.5 changed by -0.0333.
- Duplicate Detections: Duplicate false positives increased from 0 to 2, indicating seam-fusion tuning may be required.

## 4. Class-Wise Performance Shifts
- **Classes Improved**: None (> 0.005)
- **Classes Degraded**: bridge (-0.1667), overpass (-0.5000)

## 5. Object-Size Analysis (mAP@0.5)
| Size Bin | Baseline | Experimental | Delta |
| :--- | :---: | :---: | :---: |
| **Small** | 0.0500 | 0.0500 | +0.0000 |
| **Medium** | 0.1000 | 0.1000 | +0.0000 |
| **Large** | 0.2000 | 0.1750 | -0.0250 |

## 6. Error Analysis Breakdown
| Error Archetype | Baseline | Experimental |
| :--- | :---: | :---: |
| **True Positives** | 12 | 12 |
| **Localization Errors** | 0 | 11 |
| **Class Confusions** | 1 | 2 |
| **Duplicate FPs** | 0 | 2 |
| **Background FPs** | 4 | 18 |
| **Missed GTs (FN)** | 0 | 0 |

## 7. Object-Density Analysis (mAP@0.5)
| Density Category | Baseline | Experimental |
| :--- | :---: | :---: |
| **Sparse** | 1.0000 | 0.6457 |
| **Moderate** | 1.0000 | 1.0000 |
| **Dense** | 0.0000 | 0.0000 |

## 8. Visual Samples & Research Plots
- **Comparative Diagnostics**: `docs/figures/comparative_diagnostics.png`
- **Confidence Distribution**: `docs/figures/confidence_distribution.png`
- **Class-Wise AP**: `docs/figures/class_wise_ap.png`
- **Density Performance**: `docs/figures/density_performance.png`
