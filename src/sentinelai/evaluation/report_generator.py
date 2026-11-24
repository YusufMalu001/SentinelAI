"""Automated comparative experiment analysis report generator."""

import os
from typing import Dict, Any, List, Optional


class ExperimentReportGenerator:
    """Generates rigorous Markdown analysis reports comparing an experiment against a baseline."""

    def generate_comparison_report(
        self,
        baseline_record: Dict[str, Any],
        exp_record: Dict[str, Any],
        output_path: str,
        baseline_density: Optional[Dict[str, Any]] = None,
        exp_density: Optional[Dict[str, Any]] = None,
        figure_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generates a structured Markdown report with nuanced trade-off analysis."""
        b_id = baseline_record.get("experiment_id", "baseline")
        e_id = exp_record.get("experiment_id", "experiment")

        b_met = baseline_record.get("metrics", {})
        e_met = exp_record.get("metrics", {})

        b_perf = baseline_record.get("performance", {})
        e_perf = exp_record.get("performance", {})

        b_err = baseline_record.get("error_analysis", {})
        e_err = exp_record.get("error_analysis", {})

        # Compute deltas
        d_map50 = e_met.get("mAP50", 0.0) - b_met.get("mAP50", 0.0)
        d_map5095 = e_met.get("mAP50_95", 0.0) - b_met.get("mAP50_95", 0.0)
        b_lat = b_perf.get("latency_per_image_ms", 1.0)
        e_lat = e_perf.get("latency_per_image_ms", 1.0)
        lat_ratio = e_lat / max(1e-6, b_lat)

        # Class-wise deltas
        b_cls = b_met.get("per_class_mAP50", {})
        e_cls = e_met.get("per_class_mAP50", {})
        improved_classes: List[str] = []
        degraded_classes: List[str] = []
        for cls_name in b_cls:
            diff = e_cls.get(cls_name, 0.0) - b_cls.get(cls_name, 0.0)
            if diff > 0.005:
                improved_classes.append(f"{cls_name} (+{diff:.4f})")
            elif diff < -0.005:
                degraded_classes.append(f"{cls_name} ({diff:.4f})")

        # Size-stratified deltas
        b_size = b_met.get("size_metrics", {})
        e_size = e_met.get("size_metrics", {})
        d_small = e_size.get("small", {}).get("mAP50", 0.0) - b_size.get("small", {}).get("mAP50", 0.0)
        d_large = e_size.get("large", {}).get("mAP50", 0.0) - b_size.get("large", {}).get("mAP50", 0.0)

        # Trade-off observations
        tradeoffs: List[str] = []
        if d_map50 > 0 and lat_ratio > 1.5:
            tradeoffs.append(
                f"Accuracy-Latency Trade-off: Overall mAP@0.5 changed by {d_map50:+.4f}, "
                f"while inference latency increased by {lat_ratio:.2f}x ({b_lat:.1f}ms -> {e_lat:.1f}ms)."
            )
        elif d_map50 <= 0 and lat_ratio > 1.5:
            tradeoffs.append(
                f"Computational Overhead without Global Gain: Latency increased by {lat_ratio:.2f}x "
                f"({b_lat:.1f}ms -> {e_lat:.1f}ms) while overall mAP@0.5 changed by {d_map50:+.4f}."
            )

        if d_small > 0 and d_large < 0:
            tradeoffs.append(
                f"Scale Trade-off: Small-object mAP@0.5 improved ({d_small:+.4f}), "
                f"but large-object mAP@0.5 degraded ({d_large:+.4f})."
            )
        elif d_small > 0:
            tradeoffs.append(
                f"Small-Object Recovery: Small-object mAP@0.5 improved by {d_small:+.4f} without large-object regression."
            )

        b_dup = b_err.get("duplicate_fps", 0)
        e_dup = e_err.get("duplicate_fps", 0)
        if e_dup > b_dup:
            tradeoffs.append(
                f"Duplicate Detections: Duplicate false positives increased from {b_dup} to {e_dup}, "
                f"indicating seam-fusion tuning may be required."
            )

        if not tradeoffs:
            tradeoffs.append("No severe precision/latency regressions observed under current evaluation protocol.")

        lines = [
            f"# Automated Experiment Analysis Report: `{e_id}` vs `{b_id}`",
            "",
            "## 1. Configuration & Experimental Context",
            f"- **Baseline ID**: `{b_id}`",
            f"- **Experimental ID**: `{e_id}`",
            f"- **Dataset Split**: `{exp_record.get('config', {}).get('evaluation', {}).get('split', 'validation')}`",
            f"- **Sample Count**: {exp_record.get('sample_count', 0)} images",
            f"- **Timestamp**: {exp_record.get('timestamp', 'N/A')}",
            "",
            "## 2. Overall Metrics & Computational Cost",
            "| Metric | Baseline | Experimental | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **mAP@0.5** | {b_met.get('mAP50', 0.0):.4f} | {e_met.get('mAP50', 0.0):.4f} | {d_map50:+.4f} |",
            f"| **mAP@0.5:0.95** | {b_met.get('mAP50_95', 0.0):.4f} | {e_met.get('mAP50_95', 0.0):.4f} | {d_map5095:+.4f} |",
            f"| **Latency (ms/img)** | {b_lat:.2f} | {e_lat:.2f} | {lat_ratio:.2f}x |",
            f"| **Throughput (FPS)** | {b_perf.get('throughput_fps', 0.0):.2f} | {e_perf.get('throughput_fps', 0.0):.2f} | - |",
            "",
            "## 3. Trade-off Analysis",
        ]
        for t in tradeoffs:
            lines.append(f"- {t}")

        lines.extend([
            "",
            "## 4. Class-Wise Performance Shifts",
            f"- **Classes Improved**: {', '.join(improved_classes) if improved_classes else 'None (> 0.005)'}",
            f"- **Classes Degraded**: {', '.join(degraded_classes) if degraded_classes else 'None (< -0.005)'}",
            "",
            "## 5. Object-Size Analysis (mAP@0.5)",
            "| Size Bin | Baseline | Experimental | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Small** | {b_size.get('small', {}).get('mAP50', 0.0):.4f} | {e_size.get('small', {}).get('mAP50', 0.0):.4f} | {d_small:+.4f} |",
            f"| **Medium** | {b_size.get('medium', {}).get('mAP50', 0.0):.4f} | {e_size.get('medium', {}).get('mAP50', 0.0):.4f} | {e_size.get('medium', {}).get('mAP50', 0.0) - b_size.get('medium', {}).get('mAP50', 0.0):+.4f} |",
            f"| **Large** | {b_size.get('large', {}).get('mAP50', 0.0):.4f} | {e_size.get('large', {}).get('mAP50', 0.0):.4f} | {d_large:+.4f} |",
            "",
            "## 6. Error Analysis Breakdown",
            "| Error Archetype | Baseline | Experimental |",
            "| :--- | :---: | :---: |",
            f"| **True Positives** | {b_err.get('tp', 0)} | {e_err.get('tp', 0)} |",
            f"| **Localization Errors** | {b_err.get('localization_errors', 0)} | {e_err.get('localization_errors', 0)} |",
            f"| **Class Confusions** | {b_err.get('class_confusions', 0)} | {e_err.get('class_confusions', 0)} |",
            f"| **Duplicate FPs** | {b_dup} | {e_dup} |",
            f"| **Background FPs** | {b_err.get('background_fps', 0)} | {e_err.get('background_fps', 0)} |",
            f"| **Missed GTs (FN)** | {b_err.get('missed_gts', 0)} | {e_err.get('missed_gts', 0)} |",
        ])

        if baseline_density and exp_density:
            lines.extend([
                "",
                "## 7. Object-Density Analysis (mAP@0.5)",
                "| Density Category | Baseline | Experimental |",
                "| :--- | :---: | :---: |",
            ])
            for cat in ["sparse", "moderate", "dense"]:
                b_d = baseline_density.get("categories", {}).get(cat, {}).get("mAP50", 0.0)
                e_d = exp_density.get("categories", {}).get(cat, {}).get("mAP50", 0.0)
                lines.append(f"| **{cat.capitalize()}** | {b_d:.4f} | {e_d:.4f} |")

        if figure_paths:
            lines.extend(["", "## 8. Visual Samples & Research Plots"])
            for label, fpath in figure_paths.items():
                lines.append(f"- **{label}**: `{fpath}`")

        report_text = "\n".join(lines) + "\n"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_text
