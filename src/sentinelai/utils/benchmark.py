"""Benchmarking and computational profiling harness for SentinelAI.

Ensures scientifically fair latency and throughput benchmarks:
- Isolates pure inference time from disk I/O loading
- Executes pre-benchmark warmup passes
- Records mean, median, standard deviation, and 95th percentile latency
- Captures system hardware, CPU model, RAM, CUDA availability, and peak GPU VRAM
- Tracks preprocessing, model execution, and fusion postprocessing stages
"""

import time
import platform
import psutil
import numpy as np
from typing import List, Dict, Any, Union, Optional
import cv2

from sentinelai.dataset.structures import ImageDetectionResult


def get_hardware_profile() -> Dict[str, Any]:
    """Retrieves system hardware and environment specs for reproducibility."""
    profile: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
        "peak_gpu_memory_mb": None,
    }

    try:
        import torch
        profile["torch_version"] = torch.__version__
        profile["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            profile["cuda_device_count"] = torch.cuda.device_count()
            profile["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return profile


def get_current_peak_gpu_memory() -> Optional[float]:
    """Returns peak GPU allocated memory in Megabytes if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def reset_peak_gpu_memory() -> None:
    """Resets CUDA max memory tracker if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class LatencyBenchmark:
    """Profiles detector latency and throughput on preloaded in-memory image buffers."""

    def __init__(self, warmup_runs: int = 2):
        self.warmup_runs = max(0, warmup_runs)

    def benchmark_pipeline(
        self,
        detector: Any,
        image_inputs: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        benchmark_name: str = "pipeline_benchmark",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs a fair latency benchmark over pre-loaded images.

        Args:
            detector: Model or detector pipeline with .predict() method.
            image_inputs: List of file paths or numpy arrays.
            conf_threshold: Inference threshold.
            benchmark_name: Identifier for this benchmark run.
            metadata: Optional pipeline metadata (e.g. tile_size, scales, fusion).

        Returns:
            Dictionary containing comprehensive timing and hardware metrics.
        """
        # Pre-load all images into memory to exclude disk I/O from inference timing
        preloaded_images: List[np.ndarray] = []
        for inp in image_inputs:
            if isinstance(inp, str):
                img = cv2.imread(inp)
                if img is not None:
                    preloaded_images.append(img)
            else:
                preloaded_images.append(inp)

        if not preloaded_images:
            raise ValueError("No valid images provided for benchmarking.")

        # Warmup passes to prime torch JIT / memory allocations
        warmup_start = time.perf_counter()
        if self.warmup_runs > 0:
            sample_img = preloaded_images[0]
            for _ in range(self.warmup_runs):
                _ = detector.predict(sample_img, conf_threshold=conf_threshold)
        warmup_time_ms = (time.perf_counter() - warmup_start) * 1000.0

        # Reset GPU peak tracking
        reset_peak_gpu_memory()

        # Timed execution loop
        inference_latencies: List[float] = []
        postprocess_latencies: List[float] = []
        total_latencies: List[float] = []
        detection_counts: List[int] = []

        overall_start = time.perf_counter()

        for img in preloaded_images:
            img_start = time.perf_counter()
            res = detector.predict(img, conf_threshold=conf_threshold)
            img_wall_ms = (time.perf_counter() - img_start) * 1000.0

            inference_latencies.append(res.inference_time_ms)
            postprocess_latencies.append(res.postprocess_time_ms)
            total_latencies.append(img_wall_ms)
            detection_counts.append(len(res.predictions))

        overall_wall_sec = time.perf_counter() - overall_start
        num_images = len(preloaded_images)

        # Statistical summaries
        lat_arr = np.array(total_latencies, dtype=np.float32)
        inf_arr = np.array(inference_latencies, dtype=np.float32)
        post_arr = np.array(postprocess_latencies, dtype=np.float32)

        peak_gpu_mb = get_current_peak_gpu_memory()

        return {
            "benchmark_name": benchmark_name,
            "num_images": num_images,
            "warmup_runs": self.warmup_runs,
            "warmup_time_ms": round(warmup_time_ms, 2),
            "throughput_fps": round(num_images / max(1e-6, overall_wall_sec), 2),
            "wall_clock_time_sec": round(overall_wall_sec, 4),
            "timing_statistics_ms": {
                "total_wall_mean": round(float(np.mean(lat_arr)), 2),
                "total_wall_median": round(float(np.median(lat_arr)), 2),
                "total_wall_std": round(float(np.std(lat_arr)), 2),
                "total_wall_p95": round(float(np.percentile(lat_arr, 95)), 2),
                "inference_mean": round(float(np.mean(inf_arr)), 2),
                "inference_median": round(float(np.median(inf_arr)), 2),
                "postprocess_mean": round(float(np.mean(post_arr)), 2),
            },
            "average_detections_per_image": round(float(np.mean(detection_counts)), 2),
            "peak_gpu_memory_mb": peak_gpu_mb,
            "pipeline_metadata": metadata or {},
            "hardware": get_hardware_profile(),
        }
