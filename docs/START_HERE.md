# 🚀 START HERE

Welcome to your optimized PhaseTransition lattice simulation framework!

## ⚡ Quick Start (30 seconds)

```bash
python simulation/latticeSimeRescale_numba.py
```

**Done!** Output is automatically saved to `logs/simulation_TIMESTAMP.log`

---

## 📖 What You Have

### ✅ Performance
- **25-30× speedup** achieved (50 min → 2 min)
- Multi-threaded, optimized, production-ready

### ✅ Logging
- All output saved to `logs/` directory automatically
- Timestamped files: `simulation_YYYYMMDD_HHMMSS.log`
- See `EXAMPLE_LOG.txt` for what logs look like

### ✅ Organization
- Codebase organized into `simulation/`, `utils/`, `scripts/`, `docs/`

### ✅ Documentation
- 8+ guides covering everything
- See list below

---

## 📚 Documentation Guide

**Start with these** (in order):

1. **`docs/QUICK_REFERENCE.md`** ← Commands you'll use daily
2. **`docs/WHAT_WAS_DONE.md`** ← Complete summary of work
3. **`README.md`** ← Project overview

**For specific topics**:

| Topic | File |
|-------|------|
| Performance details | `docs/PERFORMANCE_SUMMARY.md` |
| How to organize files | `docs/DIRECTORY_STRUCTURE.md` |
| All optimizations | `docs/SPEED_OPTIMIZATIONS.md` |
| More speedup options | `docs/PRACTICAL_SPEEDUPS.md` |
| Mac GPU options | `docs/MAC_GPU_OPTIONS.md` |
| CUDA GPU setup | `docs/GPU_SETUP.md` |

---

## 🎯 Common Tasks

### Run Simulation
```bash
python simulation/latticeSimeRescale_numba.py
# Output → logs/simulation_YYYYMMDD_HHMMSS.log
```

### View Latest Log
```bash
cat logs/$(ls -t logs/ | head -1)
```

### Check Performance
```bash
python utils/profile_detailed.py
```

---

## 🔍 Files at a Glance

### 🎯 Main Files (Use These!)
- **`simulation/latticeSimeRescale_numba.py`** ← Your optimized simulation
- **`docs/QUICK_REFERENCE.md`** ← Commands cheat sheet
- **`docs/WHAT_WAS_DONE.md`** ← Complete summary

### 🛠️ Utilities
- `utils/check_numba_threads.py` - Check threading
- `utils/profile_detailed.py` - Performance profiling
- `utils/benchmark_comparison.py` - Speed comparison

### 📊 Examples
- `EXAMPLE_LOG.txt` - What logs look like

### 📚 Documentation (8 files)
- All the `.md` files

---

## 💡 First Time Setup

### 1. Check Threading
```bash
python utils/check_numba_threads.py
```

### 2. Run Short Test
```bash
# Edit simulation/latticeSimeRescale_numba.py:
# Change line ~205: Nt = 10000  # Quick test
# Change line ~200: Nx, Ny = 64, 64  # Smaller grid

python simulation/latticeSimeRescale_numba.py
```

### 3. View the Log
```bash
cat logs/$(ls -t logs/ | head -1)
```

### 4. For Production
```bash
# Restore original values in simulation/latticeSimeRescale_numba.py:
# Nt = 100000
# Nx, Ny = 128, 128

python simulation/latticeSimeRescale_numba.py
```

---

## 🎉 You're Ready!

Everything is set up and documented. Just:
1. Run `python simulation/latticeSimeRescale_numba.py`
2. Check logs in `logs/` directory
3. Enjoy **25-30× faster** simulations!

**Questions?** → See `docs/WHAT_WAS_DONE.md` for complete details.

---

## 📞 Quick Help

| Issue | Solution |
|-------|----------|
| "How do I run it?" | `python simulation/latticeSimeRescale_numba.py` |
| "Where are results?" | `logs/` directory |
| "How to check speed?" | `python utils/profile_detailed.py` |
| "Need more info?" | Read `docs/WHAT_WAS_DONE.md` |

---

**Happy simulating!** 🚀

