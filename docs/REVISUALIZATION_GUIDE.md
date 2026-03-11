# Field State Saving & Revisualization Guide

## Overview

Your simulation now saves **field states** at every snapshot, allowing you to:
- ✅ Revisualize with proper colorbar scaling
- ✅ Try different colormaps
- ✅ Analyze field evolution after simulation completes
- ✅ Create custom visualizations without re-running

## What Gets Saved

### During Simulation

At each snapshot (every `steps` iterations), the simulation saves:

**Field State Files**: `field_states/state_step_NNNNNNNNNN.npz`
- `phi` - Field configuration (2D array)
- `pi` - Momentum field (2D array)
- `step` - Iteration number
- `time` - Physical time
- `temperature` - Temperature at this step
- `phi_min`, `phi_max` - Field range

**Metadata File**: `simulation_metadata.npz`
- All simulation parameters
- Grid size, physical constants
- VEV for normalization
- Timing information

### File Structure

```
figs/latticeSim_rescaled_numba/set6/T0_7330.../
├── field_states/                    # Field data (NEW!)
│   ├── state_step_0000000000.npz
│   ├── state_step_0050000.npz
│   ├── state_step_0100000.npz
│   └── ...
├── simulation_metadata.npz          # Parameters (NEW!)
├── t_0.0.png                        # Original snapshots
├── t_0.5.png
└── ...
```

## The Colorbar Problem (Now Fixed!)

### The Issue ⚠️

Your original snapshots used:
```python
vmin=-1000, vmax=1000
```

But your true vacuum is at:
```python
v = sqrt(mphi^2 / lambda) ≈ 10^11
```

**Result**: Bubble nucleation completely invisible! (0.00001% of actual range)

### The Solution ✅

Use **normalized field** (φ/v):
- False vacuum (φ ≈ 0) → φ/v ≈ 0 (green)
- True vacuum (φ ≈ ±v) → φ/v ≈ ±1 (red/blue)
- Bubble walls → Sharp color transitions

## Using the Revisualization Script

### Basic Usage

```bash
# Revisualize with proper normalization (recommended!)
python postprocess/revisualize_snapshots.py <simulation_directory>

# Example:
python postprocess/revisualize_snapshots.py figs/latticeSim_rescaled_numba/set6/T0_7330_dt_1e-05_a_0.001_interval_50000
```

This creates: `revisualized_normalized/` directory with corrected plots.

### Colorbar Modes

**1. Normalized (Recommended for phase transitions)**
```bash
python postprocess/revisualize_snapshots.py <sim_dir> --mode normalized
```
- Shows φ/v (fraction of VEV)
- Range: [-1.5, 1.5]
- **Best for seeing bubble nucleation!**

**2. Auto (Data-driven)**
```bash
python postprocess/revisualize_snapshots.py <sim_dir> --mode auto
```
- Uses actual field range
- Adapts to data
- Good for exploratory analysis

**3. Symmetric (Percentile-based)**
```bash
python postprocess/revisualize_snapshots.py <sim_dir> --mode symmetric
```
- Symmetric around zero
- 99th percentile range
- Robust to outliers

**4. Fixed (Original)**
```bash
python postprocess/revisualize_snapshots.py <sim_dir> --mode fixed
```
- Fixed [-1000, 1000]
- Same as original (will miss bubbles!)

### Different Colormaps

```bash
# Seismic colormap
python postprocess/revisualize_snapshots.py <sim_dir> --cmap seismic

# Viridis
python postprocess/revisualize_snapshots.py <sim_dir> --cmap viridis

# RdBu (red-blue)
python postprocess/revisualize_snapshots.py <sim_dir> --cmap RdBu_r
```

Popular colormaps: `coolwarm`, `seismic`, `RdBu_r`, `bwr`, `PiYG`

### Compare Modes

See all colorbar modes side-by-side:
```bash
python postprocess/revisualize_snapshots.py <sim_dir> --compare
```

Creates `colorbar_comparison.png` showing all modes.

## Complete Workflow

### 1. Run Simulation (saves field states automatically)

```bash
python simulation/latticeSimeRescale_numba.py
```

Output:
```
Field states will be saved to: figs/.../field_states
...
Metadata saved to: figs/.../simulation_metadata.npz
Field states saved to: figs/.../field_states/
Use postprocess/revisualize_snapshots.py to replot with different settings
```

### 2. Compare Colorbar Modes

```bash
python postprocess/revisualize_snapshots.py figs/latticeSim_rescaled_numba/set6/T0_7330_dt_1e-05_a_0.001_interval_50000 --compare
```

Check `colorbar_comparison.png` to see which mode shows bubbles best.

### 3. Revisualize All Snapshots

```bash
# With proper normalization
python postprocess/revisualize_snapshots.py figs/latticeSim_rescaled_numba/set6/T0_7330_dt_1e-05_a_0.001_interval_50000 --mode normalized

# Or with auto scaling
python postprocess/revisualize_snapshots.py figs/latticeSim_rescaled_numba/set6/T0_7330_dt_1e-05_a_0.001_interval_50000 --mode auto
```

### 4. Check Results

```bash
ls figs/.../revisualized_normalized/
# snapshot_step_0000000000.png
# snapshot_step_0050000.png
# ...
```

## What You Should See

### With Proper Normalization (φ/v):

**Early times (T > T_c)**:
- φ/v ≈ 0 everywhere (green/white)
- Thermal fluctuations visible
- No bubbles yet

**Nucleation event**:
- Local region with φ/v → ±1 (red or blue)
- Bubble appears clearly!
- Sharp boundary visible

**Bubble expansion**:
- Growing region of true vacuum (red/blue)
- False vacuum outside (green/white)
- Bubble wall clearly defined

**Late times (T < T_c)**:
- Most of space in true vacuum (φ/v ≈ ±1)
- Possible bubble collisions
- Domain walls if multiple bubbles

### With Original [-1000, 1000] Range:

- Everything looks the same (thermal noise)
- **Bubbles invisible!** (field goes to ~10^11, saturates colorbar)
- Wasted 28 hours of simulation time! ⚠️

## Storage Space

**Field states** are saved compressed (`.npz` format):

| Grid Size | Steps | Snapshots | Storage |
|-----------|-------|-----------|---------|
| 64×64 | 100k | 2 | ~20 KB |
| 128×128 | 100k | 2 | ~80 KB |
| 128×128 | 100M | 100 | ~4 MB |
| 256×256 | 100M | 100 | ~16 MB |

**Very manageable!** Much smaller than PNG images.

## Advanced: Custom Analysis

Load field states programmatically:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load metadata
metadata = np.load("figs/.../simulation_metadata.npz")
vev = float(metadata['vev'])

# Load a field state
state = np.load("figs/.../field_states/state_step_0000050000.npz")
phi = state['phi']
T = state['temperature']

# Analyze
phi_normalized = phi / vev
bubble_fraction = np.sum(np.abs(phi_normalized) > 0.5) / phi.size

print(f"Temperature: {T:.1f}")
print(f"Bubble fraction: {bubble_fraction*100:.1f}%")

# Custom plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original field
axes[0].imshow(phi, cmap='coolwarm')
axes[0].set_title("Field φ")

# Normalized
axes[1].imshow(phi_normalized, cmap='coolwarm', vmin=-1.5, vmax=1.5)
axes[1].set_title("Normalized φ/v")

plt.savefig("custom_analysis.png")
```

## Tips

### For 100M Step Run

With `steps = 1_000_000`, you'll save ~100 snapshots:
- Total storage: ~5-20 MB (compressed)
- Revisualization: < 1 minute
- Can try different colorbars instantly!

### Recommended Workflow

1. **Run simulation** (auto-saves field states)
2. **Check colorbar comparison** first
3. **Revisualize** with best mode
4. **Share** clean visualizations in papers

### If Storage is Tight

Reduce snapshot frequency:
```python
steps = 5_000_000  # Save every 5M instead of 1M
# 20 snapshots instead of 100
```

Or delete original PNG snapshots (keep field states only).

## Troubleshooting

### "No field state files found"

**Problem**: Simulation ran before field state saving was added.

**Solution**: Re-run simulation (states will be saved automatically).

### "Metadata file not found"

**Problem**: Old simulation without metadata.

**Solution**: Re-run, or manually provide VEV:
```python
# In postprocess/revisualize_snapshots.py, set manually:
vev = 1e11  # Your calculated VEV
```

### Colorbars still look wrong

**Problem**: Using 'fixed' mode or wrong VEV.

**Solution**:
1. Check metadata has correct VEV
2. Use 'normalized' or 'auto' mode
3. Run comparison to verify

## Summary

✅ **Field states saved automatically** during simulation
✅ **Revisualize anytime** with proper colorbar
✅ **Compare modes** to find best visualization
✅ **No re-running** needed for better plots
✅ **Minimal storage** (~5-20 MB for 100 snapshots)

**Most important**: Use **normalized mode** (φ/v) to actually see your bubble nucleation! 🎯

