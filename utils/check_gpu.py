#!/usr/bin/env python
"""
Quick check if GPU acceleration is available for lattice simulation.
"""
import sys

print("=" * 70)
print("GPU AVAILABILITY CHECK")
print("=" * 70)

# Check CuPy
print("\n1. Checking CuPy installation...")
try:
    import cupy as cp

    print("   ✅ CuPy is installed")
    print(f"   Version: {cp.__version__}")
except ImportError:
    print("   ❌ CuPy not found")
    print("\n   To install:")
    print("   pip install cupy-cuda11x  # for CUDA 11.x")
    print("   pip install cupy-cuda12x  # for CUDA 12.x")
    sys.exit(1)

# Check CUDA
print("\n2. Checking CUDA availability...")
try:
    cuda_available = cp.cuda.is_available()
    if cuda_available:
        print("   ✅ CUDA is available")
    else:
        print("   ❌ CUDA not available")
        print("   Check: nvidia-smi")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error checking CUDA: {e}")
    sys.exit(1)

# Get GPU info
print("\n3. GPU Information:")
try:
    device = cp.cuda.Device()
    print(f"   GPU Name: {device.name}")
    print(f"   Compute Capability: {device.compute_capability}")

    # Memory info
    mempool = cp.get_default_memory_pool()
    total_bytes = device.mem_info[1]
    free_bytes = device.mem_info[0]
    used_bytes = total_bytes - free_bytes

    print(f"   Total Memory: {total_bytes / 1e9:.2f} GB")
    print(f"   Free Memory: {free_bytes / 1e9:.2f} GB")
    print(f"   Used Memory: {used_bytes / 1e9:.2f} GB")

    # CUDA version
    cuda_version = cp.cuda.runtime.runtimeGetVersion()
    major = cuda_version // 1000
    minor = (cuda_version % 1000) // 10
    print(f"   CUDA Version: {major}.{minor}")

except Exception as e:
    print(f"   ⚠️  Could not get full GPU info: {e}")

# Quick benchmark
print("\n4. Running quick benchmark...")
try:
    import time
    import numpy as np

    # Small test
    size = (1024, 1024)
    print(f"   Test size: {size[0]}×{size[1]}")

    # CPU
    a_cpu = np.random.randn(*size).astype(np.float32)
    b_cpu = np.random.randn(*size).astype(np.float32)
    t0 = time.time()
    c_cpu = a_cpu + b_cpu
    _ = c_cpu.sum()
    t_cpu = time.time() - t0

    # GPU
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    cp.cuda.Stream.null.synchronize()  # Ensure transfer complete

    t0 = time.time()
    c_gpu = a_gpu + b_gpu
    _ = cp.asnumpy(c_gpu.sum())
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - t0

    speedup = t_cpu / t_gpu
    print(f"   CPU time: {t_cpu*1000:.2f} ms")
    print(f"   GPU time: {t_gpu*1000:.2f} ms")
    print(f"   Speedup: {speedup:.1f}×")

except Exception as e:
    print(f"   ⚠️  Benchmark failed: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n✅ GPU acceleration is READY!")
print("\nYou can now run:")
print("   python simulation/latticeSimeRescale_gpu.py")
print("\nExpected speedup for 128×128 lattice:")
print("   ~100-200× faster than original scipy code")
print("   ~5-10× faster than Numba CPU-optimized code")
print("=" * 70)
