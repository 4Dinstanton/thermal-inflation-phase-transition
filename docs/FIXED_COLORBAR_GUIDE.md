# Fixed Colorbar for Bubble Detection - Important Update!

## The Problem You Identified ✅

**Old behavior**: Each snapshot calculated its own colorbar range
- Snapshot 1: `vmin/vmax` based on data at t=0
- Snapshot 2: `vmin/vmax` based on data at t=1
- Snapshot 3: `vmin/vmax` based on data at t=2
- ...

**Result**: Colorbar kept changing → **impossible to detect rare events!**

## The Fix 🎯

**New behavior**: ALL snapshots use the SAME fixed colorbar range
- Calculate ONE global range before processing
- Apply this SAME range to ALL snapshots
- Rare bubble nucleation will stand out dramatically!

## How It Works Now

### Mode: `normalized` (RECOMMENDED)

```python
# Fixed range for ALL snapshots
vmin, vmax = -1.5, 1.5  # φ/v scale

# What you'll see:
- False vacuum (φ ≈ 0)      → φ/v ≈ 0  (green)
- True vacuum (φ ≈ ±10^11)  → φ/v ≈ ±1 (red/blue)
- Bubble nucleation         → Dramatic color change!
```

### Mode: `auto`

```python
# Scan ALL snapshots first
global_min = min(phi_min from all snapshots)
global_max = max(phi_max from all snapshots)

# Then use this FIXED range for all snapshots
vmin, vmax = -1.1*max(|global_min|, |global_max|), +1.1*max(...)
```

### Mode: `vev_based`

```python
# Based on physics, not data
vev = sqrt(mphi^2 / lambda)
vmin, vmax = -1.2*vev, +1.2*vev  # FIXED for all snapshots
```

## Usage

### Basic (recommended)

```bash
# Uses normalized mode with FIXED colorbar
python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350_dt_1e-05_a_0.001_interval_500000
```

Output:
```
Found 100 field states to revisualize
Colorbar mode: normalized
Colormap: coolwarm

Physics parameters:
  VEV (v) = 1.00e+11
  mphi = 1.00e+03

Calculating GLOBAL colorbar range for all 100 snapshots...
  Mode: Normalized (φ/v)
  Range: [-1.5, 1.5] (FIXED)
  → False vacuum (φ≈0) appears as φ/v≈0 (green)
  → True vacuum (φ≈±v) appears as φ/v≈±1 (red/blue)

======================================================================
IMPORTANT: ALL snapshots will use the SAME colorbar range!
  vmin = -1.50e+00
  vmax = 1.50e+00
This fixed scale is essential for detecting rare bubble nucleation.
======================================================================
```

### Compare modes first

```bash
# Creates colorbar_comparison.png showing all 4 modes
python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --compare
```

### Other modes

```bash
# VEV-based (physical scale, not normalized)
python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --mode vev_based

# Auto (global data range)
python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --mode auto

# Symmetric (global percentile)
python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --mode symmetric
```

## Why This Matters for Your Research

### With OLD per-snapshot colorbar:
```
Step 0:   φ ∈ [-10, +10]       → colorbar [-10, +10]
Step 50k: φ ∈ [-100, +100]     → colorbar [-100, +100]  (rescaled!)
Step 100k: φ ∈ [-10^11, +10^11] → colorbar [-10^11, +10^11]  (rescaled!)
```
**Problem**: Everything looks "green" in every snapshot because colorbar adapts!

### With NEW fixed colorbar:
```
Step 0:   φ ∈ [-10, +10]       → colorbar [-1.5, +1.5] φ/v  (≈0, green)
Step 50k: φ ∈ [-100, +100]     → colorbar [-1.5, +1.5] φ/v  (≈0, green)
Step 100k: φ ∈ [-10^11, +10^11] → colorbar [-1.5, +1.5] φ/v  (→±1, RED/BLUE!)
```
**Success**: Bubble nucleation jumps out as dramatic color change!

## Workflow

1. **Run simulation** (saves field states automatically)
   ```bash
   python simulation/latticeSimeRescale_numba.py
   ```

2. **Compare colorbar modes** (once, to decide which mode to use)
   ```bash
   python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --compare
   # Check colorbar_comparison.png
   ```

3. **Revisualize all with fixed colorbar** (use chosen mode)
   ```bash
   python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350... --mode normalized
   # Creates revisualized_normalized/ directory
   ```

4. **Check for bubbles!**
   ```bash
   ls revisualized_normalized/
   # Look for dramatic color changes in sequence
   ```

## What To Look For

### Thermal Fluctuations (Early)
- Small variations around green (φ/v ≈ 0)
- Colorbar: all snapshots greenish
- **This is expected** - just noise

### Bubble Nucleation (Rare Event!)
- Sudden appearance of red or blue region (φ/v ≈ ±1)
- Localized at first
- **This is what you want to catch!**

### Bubble Growth
- Red/blue region expands
- Sharp boundary (bubble wall)
- Eventually fills space

### Bubble Collision
- Multiple bubbles meet
- Domain walls form
- Complex patterns

## Technical Details

### How Global Range is Calculated

**Normalized mode**:
```python
# Simply fixed based on physics
vmin, vmax = -1.5, 1.5  # No calculation needed
```

**Auto mode**:
```python
# Scan all N snapshots
for each snapshot in all_snapshots:
    global_min = min(global_min, snapshot.phi.min())
    global_max = max(global_max, snapshot.phi.max())

# Use global extrema
phi_range = max(abs(global_min), abs(global_max))
vmin, vmax = -1.1 * phi_range, 1.1 * phi_range
```

**Symmetric mode**:
```python
# Sample 20 snapshots, gather all |φ| values
all_abs_phi = []
for snapshot in sampled_snapshots:
    all_abs_phi.extend(abs(snapshot.phi))

# Use 99.5th percentile
phi_range = percentile(all_abs_phi, 99.5)
vmin, vmax = -phi_range, phi_range
```

### Output Structure

```
data/lattice/set6/T0_7350.../
├── field_states/                    # Saved during simulation
│   ├── state_step_0000000000.npz
│   ├── state_step_0500000.npz
│   └── ...
├── simulation_metadata.npz          # Saved during simulation
├── colorbar_comparison.png          # Created by --compare
└── revisualized_normalized/         # Created by revisualization
    ├── snapshot_step_0000000000.png
    ├── snapshot_step_0500000.png
    └── ...
```

## Summary

✅ **Fixed**: Colorbar now SAME for all snapshots
✅ **Better**: Rare events stand out dramatically
✅ **Recommended**: Use `normalized` mode (φ/v)
✅ **Workflow**: Run sim → compare → revisualize
✅ **Result**: Actually see your bubble nucleation!

The key insight: **Fixed colorbar = consistent reference frame = visible rare events!** 🎯

