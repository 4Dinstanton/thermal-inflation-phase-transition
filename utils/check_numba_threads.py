#!/usr/bin/env python
"""
Quick diagnostic to check Numba threading configuration.
Run this to see if your parallel loops will actually use multiple cores.
"""
import numba as nb
import numpy as np
import time
import os

print("=" * 70)
print("NUMBA THREADING DIAGNOSTICS")
print("=" * 70)

# Check environment
print("\n1. Environment Variables:")
print(f"   NUMBA_NUM_THREADS = {os.environ.get('NUMBA_NUM_THREADS', 'NOT SET')}")
print(f"   OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")

# Check Numba config
print("\n2. Numba Configuration:")
print(f"   Version: {nb.__version__}")
print(f"   Threading layer: {nb.config.THREADING_LAYER}")
print(f"   Configured threads: {nb.config.NUMBA_NUM_THREADS}")
print(f"   Active threads: {nb.get_num_threads()}")

# Simple benchmark
print("\n3. Running quick parallel benchmark...")


@nb.njit(parallel=True, fastmath=True)
def parallel_sum(arr):
    total = 0.0
    for i in nb.prange(arr.size):
        total += np.sin(arr[i]) * np.cos(arr[i])
    return total


# Warmup
x = np.random.randn(10_000_000)
_ = parallel_sum(x)

# Time it
t0 = time.time()
result = parallel_sum(x)
t1 = time.time()

elapsed = (t1 - t0) * 1000

print(f"   Array size: {x.size:,}")
print(f"   Time: {elapsed:.2f} ms")
print(f"   Result: {result:.6e}")

print("\n4. Interpretation:")
threads = nb.get_num_threads()
if threads == 1:
    print("   ⚠️  WARNING: Only using 1 thread!")
    print("   → Your parallel loops will NOT be parallelized!")
    print("   → Set NUMBA_NUM_THREADS before running:")
    print("      export NUMBA_NUM_THREADS=8  # (use your CPU core count)")
elif threads <= 2:
    print(f"   ⚠️  Using only {threads} threads (likely suboptimal)")
    print("   → Consider increasing NUMBA_NUM_THREADS to your core count")
else:
    print(f"   ✅ Using {threads} threads - good!")
    print("   → Parallel loops will use multiple cores")

print("\n5. To get maximum performance:")
print("   Run: ./scripts/run_lattice_fast.sh")
print("   Or manually: export NUMBA_NUM_THREADS=<your_cores>")
print("=" * 70)
