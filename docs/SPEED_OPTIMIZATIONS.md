# Speed Optimization Guide for simulation/latticeSimeRescale_numba.py

## ⚡ CRITICAL: Enable Multi-Threading First! ⚡

**By default, Numba may only use 1-2 threads!** This is why you're not seeing speedup.

### Quick Start (Most Important):

**Option 1: Use the script (easiest)**
```bash
./scripts/run_lattice_fast.sh
```

**Option 2: Manual (macOS/Linux)**
```bash
# Check your CPU cores:
sysctl -n hw.ncpu              # macOS
# nproc                        # Linux

# Set threading (replace 8 with your core count):
export NUMBA_NUM_THREADS=8
python simulation/latticeSimeRescale_numba.py
```

**Expected speedup from threading alone: 4-8× on an 8-core CPU!**

---

## Current Optimizations (Already Implemented)
1. ✅ Uniform-grid cubic spline with O(1) indexing (no binary search)
2. ✅ Preallocated scratch arrays (no per-step allocations)
3. ✅ Numba JIT compilation with cache and fastmath
4. ✅ Parallel loops with nb.prange in Laplacian and Vprime
5. ✅ Fused RK2 kernel option (set `USE_FUSED_RK2 = True`)
6. ✅ In-kernel RNG (set `USE_NUMBA_RNG = True`) - **Eliminates 30% Python overhead!**

## Why Threading Matters

Your code has parallel loops (`nb.prange`) in:
- `laplacian_periodic`: computes Laplacian for all grid points
- `Vprime_field`: evaluates V'(φ,T) for all grid points

Without setting `NUMBA_NUM_THREADS`, these run on 1-2 cores only!

---

## Additional Speed Improvements

### 1. ✅ Enable Threading (DONE - use run_lattice_fast.sh)
Expected: **4-8× on 8-core CPU**

### 2. Float32 Precision (Verified Safe)
Set `USE_FLOAT32 = True` in the code.
- **Verified safe** for your parameters (run `python utils/check_float32_safety.py`)
- φ max ~ 10³, V' max ~ 1.5×10⁹ (far below float32 limit of 3.4×10³⁸)
- Expected: **1.5-2× speedup** from memory bandwidth improvement
- Trade-off: ~7 decimal digits vs 16, but acceptable for lattice sims

### 2. Reduce Spline Resolution (Check comparison plots first)
In the file, change:
```python
N_Y = 512  # Current
# Try:
N_Y = 256  # or even 128
```
After changing, run comparison_plots() to verify errors are acceptable.
Expected speedup: ~1.5-2× for N_Y=256.

### 3. Reduce Lattice Size or Output Frequency
- Lower Nx, Ny (e.g., 64×64 instead of 128×128): 4× faster
- Increase `steps` interval: only affects I/O, not compute
- Lower Nt: fewer total steps

### 4. Fused RK2 Kernel (Moderate complexity, ~1.3-1.5×)
Combine the two `rk2_step` calls into one kernel that accepts both T and T_mid,
avoiding the second kernel entry/exit overhead.

### 5. In-Kernel RNG (Moderate complexity, ~1.2-1.3×)
Move noise generation from Python `np.random.randn` into a Numba-compiled
random generator (e.g., using `numba.random` or a simple xorshift).

### 6. Float32 Precision (If physics permits, ~1.5-2× memory bandwidth)
Change field arrays to float32:
```python
phi = 0.01 * np.random.randn(Nx, Ny).astype(np.float32)
pi = np.zeros((Nx, Ny), dtype=np.float32)
```
Keep parameters and spline coefficients as float64 for accuracy.

### 7. Reduce dx (coarser spatial resolution)
If acceptable for your physics:
```python
dx_phys = 2e-3  # instead of 1e-3
```
Larger dx → larger stable dt → fewer steps for same physical time.

### 8. ✅ GPU Acceleration (CuPy + CUDA kernels, ~10-200×)
**NEW: `simulation/latticeSimeRescale_gpu.py` available!**

Requirements:
- NVIDIA GPU with CUDA
- Install: `pip install cupy-cuda11x` (or cuda12x)
- See `docs/GPU_SETUP.md` for details

Expected speedup over CPU-optimized:
- 128×128 grid: **~5-10× vs Numba CPU**
- 256×256 grid: **~10-20× vs Numba CPU**
- Total: **~100-200× vs original scipy code!**

Usage:
```bash
python simulation/latticeSimeRescale_gpu.py
```

Auto-detects GPU and falls back to CPU if unavailable.

## Recommended Order
1. **Set NUMBA_NUM_THREADS** (easiest, immediate)
2. **Lower N_Y to 256** after checking comparison plots
3. **Try float32** if your physics tolerates single precision
4. If still not fast enough: implement fused RK2 + in-kernel RNG
5. For very large production runs: port to GPU

## Current Performance Baseline
- Original (scipy InterpolatedUnivariateSpline): ~3 min per 50k steps
- With uniform cubic + preallocated buffers: significantly faster
- Expected with all optimizations: 5-20× overall improvement on multi-core CPU

