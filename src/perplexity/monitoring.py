"""
GPU hardware monitoring utility.

This module provides background monitoring of GPU metrics like power draw,
utilization, and VRAM usage during evaluation runs.
"""

import threading
import time
import logging
import torch
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HardwareMonitor:
    """
    Background monitor for GPU hardware metrics.
    
    Samples power draw, utilization, and memory in a separate thread.
    """

    def __init__(self, device_index: int = 0, sample_interval: float = 0.05):
        """
        Initialize the monitor.
        
        Args:
            device_index: Index of the CUDA device to monitor.
            sample_interval: Time between samples in seconds.
        """
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.power_samples: List[float] = []
        self.utilization_samples: List[float] = []
        self.allocated_mem_samples: List[float] = []
        self.reserved_mem_samples: List[float] = []
        self.system_mem_samples: List[float] = []
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._is_available = torch.cuda.is_available()

    def _monitor_loop(self):
        """Internal sampling loop."""
        while not self._stop_event.is_set():
            try:
                if self._is_available:
                    # Sample power draw (mW)
                    if hasattr(torch.cuda, "power_draw"):
                        p = torch.cuda.power_draw()
                        self.power_samples.append(float(p))
                    
                    # Sample utilization (%)
                    if hasattr(torch.cuda, "utilization"):
                        u = torch.cuda.utilization(self.device_index)
                        self.utilization_samples.append(float(u))

                    # Sample VRAM (MB)
                    self.allocated_mem_samples.append(torch.cuda.memory_allocated(self.device_index) / (1024 * 1024))
                    self.reserved_mem_samples.append(torch.cuda.memory_reserved(self.device_index) / (1024 * 1024))
                    
                    # System-wide VRAM (Total - Free)
                    free, total = torch.cuda.mem_get_info(self.device_index)
                    self.system_mem_samples.append((total - free) / (1024 * 1024))
            except Exception as e:
                logger.debug(f"Error sampling hardware metrics: {e}")
            
            time.sleep(self.sample_interval)

    def start(self):
        """Start background sampling."""
        if not self._is_available:
            logger.debug("CUDA not available, hardware monitor will not start.")
            return

        self.power_samples = []
        self.utilization_samples = []
        self.allocated_mem_samples = []
        self.reserved_mem_samples = []
        self.system_mem_samples = []
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("Hardware monitor started.")

    def stop(self) -> Dict[str, Optional[float]]:
        """
        Stop sampling and return statistics.
        
        Returns:
            Dictionary with avg/max for power, utilization, and memory.
        """
        if self._monitor_thread is None:
            return {
                **{f"{k}_{suffix}": None for k in ["power_draw_mw", "gpu_utilization", "memory_allocated_mb", "memory_reserved_mb", "memory_used_system_mb"] for suffix in ["avg", "max"]},
                "num_hw_samples": 0
            }

        self._stop_event.set()
        self._monitor_thread.join(timeout=2.0)
        logger.debug("Hardware monitor stopped.")

        def get_stats(samples: List[float]):
            if not samples:
                return None, None
            return sum(samples) / len(samples), max(samples)

        p_avg, p_max = get_stats(self.power_samples)
        u_avg, u_max = get_stats(self.utilization_samples)
        a_avg, a_max = get_stats(self.allocated_mem_samples)
        r_avg, r_max = get_stats(self.reserved_mem_samples)
        s_avg, s_max = get_stats(self.system_mem_samples)

        stats = {
            "avg_power_draw_mw": p_avg,
            "max_power_draw_mw": p_max,
            "avg_gpu_utilization": u_avg,
            "max_gpu_utilization": u_max,
            "avg_memory_allocated_mb": a_avg,
            "max_memory_allocated_mb": a_max,
            "avg_memory_reserved_mb": r_avg,
            "max_memory_reserved_mb": r_max,
            "avg_memory_used_system_mb": s_avg,
            "max_memory_used_system_mb": s_max,
            "num_hw_samples": len(self.power_samples)
        }
        return stats
