#!/usr/bin/env python3
"""
GPU Memory Cleanup Script

Clears PyTorch GPU cache and resets memory allocations.
Useful for running sequential analysis on different datasets.
"""

import torch
import gc
import sys

def clear_gpu_memory():
    """
    Clear all GPU memory and cache.
    """
    print("=" * 60)
    print("GPU MEMORY CLEANUP")
    print("=" * 60)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA not available. No GPU to clean.")
        return
    
    print(f"\nGPU Device: {torch.cuda.get_device_name(0)}")
    
    # Print memory before cleanup
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated() / (1024**3)
    reserved_before = torch.cuda.memory_reserved() / (1024**3)
    print(f"\nMemory BEFORE cleanup:")
    print(f"  Allocated: {allocated_before:.2f} GB")
    print(f"  Reserved:  {reserved_before:.2f} GB")
    
    # Empty cache
    print("\nClearing GPU cache...")
    torch.cuda.empty_cache()
    
    # Force garbage collection
    print("Running garbage collection...")
    gc.collect()
    
    # Clear again after GC
    torch.cuda.empty_cache()
    
    # Print memory after cleanup
    allocated_after = torch.cuda.memory_allocated() / (1024**3)
    reserved_after = torch.cuda.memory_reserved() / (1024**3)
    print(f"\nMemory AFTER cleanup:")
    print(f"  Allocated: {allocated_after:.2f} GB")
    print(f"  Reserved:  {reserved_after:.2f} GB")
    
    freed = allocated_before + reserved_before - (allocated_after + reserved_after)
    print(f"\nMemory freed: {freed:.2f} GB")
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    available = total_memory - allocated_after
    print(f"\nGPU Available: {available:.2f} GB / {total_memory:.2f} GB")
    
    print("\n" + "=" * 60)
    print("GPU cleanup complete! Ready for next run.")
    print("=" * 60)

if __name__ == "__main__":
    clear_gpu_memory()
