# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import psutil
import threading

import torch
from accelerate.utils import is_xpu_available

def byte2gb(x):
    return int(x / 2**30)
# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()   # reset the peak gauge to zero
            self.begin = byte2gb(torch.xpu.memory_allocated())
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            self.end = byte2gb(torch.xpu.memory_allocated())
            self.peak = byte2gb(torch.xpu.max_memory_allocated())
            xpu_info = torch.xpu.memory_stats()
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.malloc_retries = xpu_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.m_ooms = xpu_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.xpu.max_memory_reserved())
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = byte2gb(torch.cuda.memory_allocated())
            self.peak = byte2gb(torch.cuda.max_memory_allocated())
            cuda_info = torch.cuda.memory_stats()
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.malloc_retries = cuda_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.m_ooms = cuda_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
        
    def print_stats(self):
        device_str = None
        if is_xpu_available():
            device_str = "XPU"
        elif torch.cuda.is_available():
            device_str = "CUDA"
            
        if device_str:
            print(f"Max {device_str} memory allocated was {self.peak} GB")
            print(f"Max {device_str} memory reserved was {self.max_reserved} GB")
            print(f"Peak active {device_str} memory was {self.peak_active_gb} GB")
            print(f"{device_str} Malloc retries : {self.malloc_retries}")
        print(f"CPU Total Peak Memory consumed during the train (max): {self.cpu_peaked + self.cpu_begin} GB")
        

def get_memory_stats(device: torch.device, reset_stats: bool = True) -> dict:
    """
    Computes a memory summary for the passed in device. If ``reset_stats`` is ``True``, this will
    also reset CUDA's peak memory tracking. This is useful to get data around relative use of peak
    memory (e.g. peak memory during model init, during forward, etc) and optimize memory for
    individual sections of training.

    Args:
        device (torch.device): Device to get memory summary for. Only CUDA devices are supported.
        reset_stats (bool): Whether to reset CUDA's peak memory tracking.

    Returns:
        Dict[str, float]: A dictionary containing the peak memory active, peak memory allocated,
        and peak memory reserved. This dict is useful for logging memory stats.

    Raises:
        ValueError: If the passed-in device is not CUDA.
    """
    if device.type != "cuda":
        raise ValueError(
            f"Logging memory stats is only supported on CUDA devices, got {device}"
        )

    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)

    memory_stats = {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_mem_alloc,
        "peak_memory_reserved": peak_mem_reserved,
    }
    return memory_stats