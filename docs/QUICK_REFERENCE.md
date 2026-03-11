# Quick Reference Card

## 🚀 Run Simulation

```bash
python simulation/latticeSimeRescale_numba.py
```

Output automatically saved to: `logs/simulation_YYYYMMDD_HHMMSS.log`

## 📊 View Results

```bash
# List all logs (newest first)
ls -lt logs/

# View latest log
cat logs/$(ls -t logs/ | head -1)

# Follow latest log in real-time
tail -f logs/$(ls -t logs/ | head -1)
```

## 🔧 Configuration

Edit `simulation/latticeSimeRescale_numba.py` (lines 20-25):

```python
USE_FUSED_RK2 = True   # Keep True
USE_NUMBA_RNG = True   # Keep True
USE_FLOAT32 = True     # Keep True
N_Y = 256              # Spline resolution
```

## ⚡ Performance

| Action | Command |
|--------|---------|
| Check threading | `python utils/check_numba_threads.py` |
| Profile performance | `python utils/profile_detailed.py` |
| Compare vs original | `python utils/benchmark_comparison.py` |

## 📚 Documentation

| Topic | File |
|-------|------|
| Complete overview | `docs/PERFORMANCE_SUMMARY.md` |
| This summary | `docs/WHAT_WAS_DONE.md` |
| Project intro | `README.md` |
| Organization | `docs/DIRECTORY_STRUCTURE.md` |
| All optimizations | `docs/SPEED_OPTIMIZATIONS.md` |
| Mac-specific | `docs/MAC_GPU_OPTIONS.md` |

## 🎯 Key Numbers

- **Speedup achieved**: 25-30×
- **Original time**: ~50 min (100k steps)
- **Current time**: ~2-3 min
- **Threading**: 8 cores (auto-detected)
- **Grid**: 128×128 (configurable)

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow performance | `python utils/check_numba_threads.py` |
| Want GPU | See `docs/MAC_GPU_OPTIONS.md` (Mac) or `docs/GPU_SETUP.md` |
| Need more speed | See `docs/PRACTICAL_SPEEDUPS.md` |
| Physics verification | Comparison plots in first run |

## 💡 Tips

### Faster Development Cycles
```python
# In simulation/latticeSimeRescale_numba.py, change:
Nx, Ny = 64, 64      # line ~200 (4× faster)
Nt = 10_000          # line ~205 (10× faster)
```

### More Threads
```bash
export NUMBA_NUM_THREADS=16  # your core count
python simulation/latticeSimeRescale_numba.py
```

### Clean Old Logs
```bash
rm -f logs/*.log
```

### Find Large Files
```bash
du -sh figs/*  # Check figure sizes
```

## 📞 Help

1. **Read first**: `docs/WHAT_WAS_DONE.md`
2. **Performance**: `docs/PERFORMANCE_SUMMARY.md`
3. **Specific issue**: See relevant `.md` file
4. **Check setup**: Run diagnostic tools

---

**Everything is documented and ready to use!** 🎉

