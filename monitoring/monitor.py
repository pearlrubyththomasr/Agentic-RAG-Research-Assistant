from __future__ import annotations

import os
import time
from typing import Dict


def _safe_import(name: str):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None


def get_system_metrics() -> Dict[str, object]:
    """Return lightweight system metrics. Optional dependencies: psutil, GPUtil."""
    psutil = _safe_import("psutil")
    gputil = _safe_import("GPUtil")

    metrics: Dict[str, object] = {
        "timestamp": int(time.time()),
        "ollama_base": os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3:latest"),
    }

    if psutil:
        try:
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            proc = psutil.Process()
            mem_info = proc.memory_info()
            metrics.update(
                {
                    "cpu_percent": cpu,
                    "memory_total": vm.total,
                    "memory_used": vm.used,
                    "memory_percent": vm.percent,
                    "process_rss": getattr(mem_info, "rss", None),
                    "process_vms": getattr(mem_info, "vms", None),
                }
            )
        except Exception:
            pass

    if gputil:
        try:
            gpus = gputil.getGPUs()
            gpu_stats = []
            for g in gpus:
                gpu_stats.append({"id": g.id, "name": g.name, "memoryTotal": g.memoryTotal, "memoryUsed": g.memoryUsed, "load": g.load})
            metrics["gpus"] = gpu_stats
        except Exception:
            pass

    return metrics
