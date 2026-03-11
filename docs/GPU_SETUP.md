# GPU Acceleration Setup Guide

## Requirements

1. **NVIDIA GPU** with CUDA support (compute capability ≥ 3.5)
2. **CUDA Toolkit** installed
3. **CuPy** Python library

## Installation

### Step 1: Check your CUDA version

```bash
nvcc --version
# or
nvidia-smi
```

### Step 2: Install CuPy

**For CUDA 11.x:**
```bash
pip install cupy-cuda11x
```

**For CUDA 12.x:**
```bash
pip install cupy-cuda12x
```

**For other CUDA versions:**
See https://docs.cupy.dev/en/stable/install.html

### Step 3: Verify installation

```bash
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'GPU: {cp.cuda.Device().name}')"
```

## Usage

### Run GPU-accelerated simulation:

```bash
python simulation/latticeSimeRescale_gpu.py
```

The script will:
- ✅ Auto-detect if GPU is available
- ✅ Fall back to CPU if CuPy not installed
- ✅ Print GPU info at startup

### Toggle GPU on/off:

Edit `simulation/latticeSimeRescale_gpu.py`:
```python
USE_GPU = False  # Force CPU even if GPU available
```

## Expected Performance

| Configuration | Grid Size | Speed | vs Original |
|---------------|-----------|-------|-------------|
| **Original CPU** (scipy) | 128×128 | 1× | baseline |
| **Numba CPU** (all opts) | 128×128 | ~20-30× | 20-30× faster |
| **GPU** (float32) | 128×128 | **~100-200×** | **100-200× faster!** |
| **GPU** (float32) | 256×256 | **~200-400×** | scales well |

### Why GPU is so fast:

1. **Parallel**: Processes all 16,384 points (128×128) simultaneously
2. **Memory bandwidth**: Float32 + GPU memory is 10× faster than CPU RAM
3. **Latency hiding**: GPU overlaps computation with memory access
4. **Scales better**: Larger grids see even bigger speedups

## Typical Speedup Breakdown

For 128×128 grid, 100k steps:

| Method | Time | Speedup |
|--------|------|---------|
| Original (scipy) | ~50 minutes | 1× |
| Numba (optimized) | ~2 minutes | 25× |
| **GPU (CuPy)** | **~15-30 seconds** | **100-200×** |

## Troubleshooting

### "CuPy not found"
```bash
pip install cupy-cuda11x  # or cuda12x
```

### "CUDA not available"
- Check `nvidia-smi` works
- Verify CUDA toolkit installed
- Check GPU compute capability ≥ 3.5

### "Out of memory"
- Reduce grid size (Nx, Ny)
- Use float32 (already default in GPU version)
- Monitor GPU memory: `nvidia-smi -l 1`

### Slower than expected
- First run includes JIT compilation (warmup)
- Small grids (<64×64) have overhead; use ≥128×128
- Check GPU isn't being used by other processes

## GPU vs CPU Decision Tree

```
Do you have NVIDIA GPU?
├─ No → Use simulation/latticeSimeRescale_numba.py (CPU optimized)
└─ Yes
   ├─ Grid < 64×64 → CPU might be faster (overhead)
   ├─ Grid 64×128 → GPU ~10-30× faster
   ├─ Grid 128×128 → GPU ~50-200× faster
   └─ Grid ≥256×256 → GPU ~100-500× faster (recommended!)
```

## Memory Requirements

| Grid Size | Float32 Memory | Float64 Memory |
|-----------|----------------|----------------|
| 64×64 | ~80 KB | ~160 KB |
| 128×128 | ~320 KB | ~640 KB |
| 256×256 | ~1.3 MB | ~2.6 MB |
| 512×512 | ~5 MB | ~10 MB |
| 1024×1024 | ~20 MB | ~40 MB |

Even budget GPUs (2-4 GB) can handle very large grids!

## Advanced: Multi-GPU

For multiple GPUs, use:
```python
import cupy as cp
cp.cuda.Device(0).use()  # Select GPU 0, 1, 2, etc.
```

Run multiple simulations in parallel on different GPUs.

