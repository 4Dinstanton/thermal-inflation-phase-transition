# PhaseTransition - Directory Organization Guide

## Directory Structure

```
PhaseTransition/
├── README.md
├── potential/               # Physics: potential definitions
│   ├── __init__.py
│   ├── Potential.py
│   └── flatonPotential.py
│
├── simulation/              # Main simulation scripts
│   ├── latticeSimeRescale_numba.py  ⭐ Primary (CPU-optimized)
│   ├── latticeSimComplex_numba.py   (Complex scalar field)
│   ├── latticeSimeRescale_gpu.py    (For CUDA GPUs)
│   ├── latticeSim_3D_torch.py       (3D PyTorch version)
│   ├── latticeSimRescale.py         (Original - reference)
│   └── latticeSim.py                (Original - reference)
│
├── analysis/                # Tunneling, action & potential analysis
│   ├── getTunneling.py              (CosmoTransitions tunneling)
│   ├── drawPotential.py             (Potential visualization)
│   ├── drawAction.py                (Action plots)
│   ├── drawAction_long.py           (Extended action plots)
│   ├── drawFittingAnalysis.py       (Fitting analysis)
│   ├── drawPhiEscOnPotential.py     (Escape field on potential)
│   ├── findCriticalTemperatures.py  (T_c and T_sp calculation)
│   ├── scanCouplingTemp.py          (Coupling-temperature scan)
│   ├── plotCouplingComparison.py    (Coupling comparison)
│   ├── compareBounceProfiles.py     (Bounce profile comparison)
│   ├── analyzeBarrierAndGamma.py    (Barrier & nucleation rate)
│   └── tunneling_Kerem.py           (Reference implementation)
│
├── postprocess/             # Post-simulation processing
│   ├── revisualize_snapshots.py     (Re-render simulation snapshots)
│   └── make_gif.py                  (Create GIF animations)
│
├── utils/                   # Diagnostic & profiling tools
│   ├── check_numba_threads.py       (Check threading setup)
│   ├── profile_detailed.py          (Performance profiling)
│   ├── benchmark_comparison.py      (Speed comparison)
│   ├── check_float32_safety.py      (Precision analysis)
│   ├── check_metal.py               (Mac GPU check)
│   └── check_gpu.py                 (CUDA GPU check)
│
├── scripts/                 # Helper shell scripts
│   ├── run_lattice_fast.sh          (Set up threading & run)
│   └── run_complex_fast.sh          (Run complex field simulation)
│
├── docs/                    # Documentation
│   ├── DIRECTORY_STRUCTURE.md       (This file)
│   ├── PERFORMANCE_SUMMARY.md       (Complete overview)
│   ├── SPEED_OPTIMIZATIONS.md       (All optimizations)
│   ├── PRACTICAL_SPEEDUPS.md        (Additional tips)
│   ├── MAC_GPU_OPTIONS.md           (macOS-specific)
│   ├── GPU_SETUP.md                 (CUDA setup)
│   ├── REVISUALIZATION_GUIDE.md     (Snapshot re-rendering)
│   ├── FIXED_COLORBAR_GUIDE.md      (Colorbar guide)
│   ├── QUICK_REFERENCE.md           (Quick reference)
│   ├── START_HERE.md                (Getting started)
│   └── WHAT_WAS_DONE.md             (Optimization history)
│
├── data/                    # All data outputs
│   ├── lattice/             # Lattice simulation results
│   │   └── set6/            # Parameter set directories
│   │       └── <run_dirs>/  # Individual simulation runs
│   ├── tunneling/           # CosmoTransitions results
│   │   ├── set1/ ... set7/  # Parameter set directories
│   │   └── (CSVs, JSONs, analysis results)
│   └── csv/toyModel/        # Reference/toy model data
│
├── figs/                    # Generated figures
└── logs/                    # Simulation logs (auto-created)
```

## Quick Start

### Run a Simulation

```bash
# Option 1: Direct run (simplest)
python simulation/latticeSimeRescale_numba.py

# Option 2: With optimal threading (recommended)
export NUMBA_NUM_THREADS=8  # your core count
python simulation/latticeSimeRescale_numba.py

# Option 3: Use the helper script
./scripts/run_lattice_fast.sh

# All output is automatically saved to logs/simulation_TIMESTAMP.log
```

### Check Performance

```bash
# See detailed profiling
python utils/profile_detailed.py

# Compare original vs optimized
python utils/benchmark_comparison.py

# Check if threading is working
python utils/check_numba_threads.py
```

### Read Documentation

```bash
# Overview of all improvements
cat docs/PERFORMANCE_SUMMARY.md

# Specific optimization details
cat docs/SPEED_OPTIMIZATIONS.md

# Mac-specific guidance
cat docs/MAC_GPU_OPTIONS.md
```

## Log Files

Every simulation run creates a log file in `logs/` directory:

**Log file name**: `simulation_YYYYMMDD_HHMMSS.log`

**Contains**:
- System configuration (threads, precision, etc.)
- Real-time progress (steps/sec, ETA)
- Time breakdown (RK2, noise, I/O)
- Final summary (total time, parameters, results)

## Key Files Reference

| Purpose | File |
|---------|------|
| **Run simulation** | `simulation/latticeSimeRescale_numba.py` |
| **Complex field sim** | `simulation/latticeSimComplex_numba.py` |
| **Check speed** | `utils/profile_detailed.py` |
| **Re-render snapshots** | `postprocess/revisualize_snapshots.py` |
| **Create GIFs** | `postprocess/make_gif.py` |
| **Read overview** | `docs/PERFORMANCE_SUMMARY.md` |
| **View logs** | `logs/simulation_*.log` |

## Data Organization

| Directory | Contents |
|-----------|----------|
| `data/lattice/` | Lattice simulation results (field states, snapshots, CSVs) |
| `data/tunneling/` | CosmoTransitions outputs (T-S CSVs, coupling scans, parameters) |
| `data/csv/toyModel/` | Reference toy model data |

## Tips

### Finding Logs
```bash
# List all logs
ls -lt logs/

# View latest log
cat logs/$(ls -t logs/ | head -1)

# View latest log in real-time
tail -f logs/$(ls -t logs/ | head -1)
```
