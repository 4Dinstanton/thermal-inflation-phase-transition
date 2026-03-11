#!/usr/bin/env python
"""
Check if Metal (Apple GPU) acceleration is available via PyTorch MPS.
"""
import sys

print("=" * 70)
print("METAL (Apple GPU) AVAILABILITY CHECK")
print("=" * 70)

# Check PyTorch
print("\n1. Checking PyTorch installation...")
try:
    import torch
    print(f"   ✅ PyTorch is installed")
    print(f"   Version: {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch not found")
    print("\n   To install:")
    print("   pip install torch")
    sys.exit(1)

# Check MPS (Metal Performance Shaders)
print("\n2. Checking Metal/MPS availability...")
try:
    if torch.backends.mps.is_available():
        print("   ✅ Metal (MPS) is available")
        if torch.backends.mps.is_built():
            print("   ✅ MPS backend is built")
        else:
            print("   ⚠️  MPS backend not built")
    else:
        print("   ❌ Metal (MPS) not available")
        print("   Note: MPS requires macOS 12.3+ and Apple Silicon (M1/M2/M3)")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Error checking MPS: {e}")
    sys.exit(1)

# Get device info
print("\n3. Device Information:")
import platform
import subprocess

print(f"   macOS version: {platform.mac_ver()[0]}")
print(f"   Python: {platform.python_version()}")
print(f"   Architecture: {platform.machine()}")

# Try to get GPU info
try:
    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                          capture_output=True, text=True, timeout=5)
    if 'Apple' in result.stdout:
        for line in result.stdout.split('\n'):
            if 'Chipset Model' in line or 'GPU' in line:
                print(f"   {line.strip()}")
except:
    print("   (Could not get detailed GPU info)")

# Quick benchmark
print("\n4. Running quick benchmark...")
try:
    import time
    import numpy as np

    size = (1024, 1024)
    print(f"   Test size: {size[0]}×{size[1]}")

    # CPU
    a_cpu = torch.randn(*size, dtype=torch.float32)
    b_cpu = torch.randn(*size, dtype=torch.float32)
    t0 = time.time()
    c_cpu = a_cpu + b_cpu
    _ = c_cpu.sum().item()
    t_cpu = time.time() - t0

    # MPS (Apple GPU)
    a_mps = a_cpu.to('mps')
    b_mps = b_cpu.to('mps')
    torch.mps.synchronize()  # Ensure transfer complete

    t0 = time.time()
    c_mps = a_mps + b_mps
    _ = c_mps.sum().item()
    torch.mps.synchronize()
    t_mps = time.time() - t0

    speedup = t_cpu / t_mps
    print(f"   CPU time: {t_cpu*1000:.2f} ms")
    print(f"   MPS (GPU) time: {t_mps*1000:.2f} ms")
    print(f"   Speedup: {speedup:.1f}×")

    if speedup > 1.5:
        print("   ✅ GPU acceleration is working!")
    else:
        print("   ⚠️  GPU not significantly faster (overhead for small ops)")

except Exception as e:
    print(f"   ⚠️  Benchmark failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if torch.backends.mps.is_available():
    print("\n✅ Metal (Apple GPU) acceleration is READY!")
    print("\nYou can use GPU acceleration on your Mac with:")
    print("   python latticeSimeRescale_metal.py")
    print("\nExpected speedup for 128×128 lattice:")
    print("   ~2-5× faster than CPU-optimized Numba")
    print("   Note: Apple GPU is not as fast as NVIDIA for this workload,")
    print("   but the Numba CPU version is already very fast on Apple Silicon!")
    print("\n💡 Recommendation:")
    print("   Your CPU-optimized Numba version is likely already excellent")
    print("   on Apple Silicon (M1/M2/M3). Try Metal for comparison, but")
    print("   the CPU version with multi-threading may be just as fast.")
else:
    print("\n⚠️  Metal GPU not available.")
    print("   Stick with the Numba CPU-optimized version")
    print("   (which is already very fast on Apple Silicon!)")

print("=" * 70)

