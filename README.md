# PhaseTransition - Lattice Simulation Framework

High-performance lattice simulation for finite-temperature scalar field theory, studying first-order cosmological phase transitions, false vacuum decay, bubble nucleation, and topological defect formation (cosmic strings).

## Note on AI-Assisted Development

Most of the code in this repository was written or substantially aided by AI (Claude / Cursor), including this readme file. This is shared intentionally -- I encourage students to leverage AI tools for scientific computing. AI is excellent at accelerating boilerplate, optimization, and visualization work, letting you focus on the physics. That said, always verify the physics and numerics yourself. Use AI as a collaborator, not a black box.

## Physics Overview

This framework solves the real-time classical field equations on a 2D or 3D lattice, coupled to a thermal bath via stochastic Langevin dynamics. The physical setup is a scalar field theory with a finite-temperature effective potential that exhibits a first-order phase transition.

### What the simulation does

1. **Finite-temperature effective potential**: The scalar potential includes tree-level, one-loop Coleman-Weinberg, and thermal corrections (bosonic and/or fermionic). At high temperature the field sits in the symmetric (false) vacuum at the origin. As the universe cools, a second minimum develops and eventually becomes the true vacuum.

2. **Bubble nucleation**: The field tunnels from the false vacuum to the true vacuum by nucleating bubbles. The simulation seeds thermal fluctuations and lets the field evolve via the Langevin equation, naturally producing bubble nucleation events.

3. **Bubble collisions and cosmic strings**: With a complex scalar field and U(1) or Z_N symmetry, colliding bubbles can produce topological defects -- cosmic strings. The simulation tracks winding numbers to identify and visualize string loops.

4. **Observables**: The code outputs 2D/3D snapshots of the field configuration, bubble identification, cosmic string detection and counting, and the time evolution of these quantities.

### Key parameters

| Parameter | Meaning | Typical value |
|-----------|---------|---------------|
| `T0` | Initial temperature (GeV) | 6770 -- 7350 |
| `Nx, Ny, Nz` | Lattice dimensions | 256 -- 1024 |
| `dx` | Lattice spacing (GeV^-1) | 0.001 -- 0.005 |
| `dt_phys` | Physical time step (GeV^-1) | 0.00025 |
| `steps` | Snapshot interval | 4000 -- 100000 |
| `λb, λf` | Boson/fermion Yukawa couplings to the thermal bath | ~1.09 |

### Tunneling analysis (CosmoTransitions)

The `analysis/` folder contains scripts that interface with [CosmoTransitions](https://github.com/clwainwright/CosmoTransitions) to compute the bounce action S_3/T, critical temperatures (T_c, T_sp), and nucleation rates. Results are stored in `data/tunneling/`.

## Quick Start

```bash
# Run the real scalar field simulation
python simulation/latticeSimeRescale_numba.py

# Or the complex scalar field simulation (supports U(1) / Z_N symmetry breaking)
python simulation/latticeSimComplex_numba.py
```

## Running Simulations with Shell Scripts

The `scripts/` folder provides shell scripts that automatically detect your CPU core count and set optimal Numba threading before launching the simulation. **Run them from the project root:**

### Real scalar field

```bash
# Basic run (auto-detects CPU cores)
./scripts/run_lattice_fast.sh

# Custom lattice size and temperature
./scripts/run_lattice_fast.sh --Nx 512 --Ny 512 --T0 6770

# Long run with frequent snapshots
./scripts/run_lattice_fast.sh --Nt 50000000 --steps 4000
```

### Complex scalar field (with cosmic strings)

```bash
# Basic run
./scripts/run_complex_fast.sh

# Custom lattice and temperature
./scripts/run_complex_fast.sh --Nx 1024 --Ny 1024 --T0 6770

# Enable Z_N symmetry breaking
./scripts/run_complex_fast.sh --zn_order 6 --zn_strength 1e20 --zn_turn_on_T 7300
```

All CLI arguments are passed through to the Python simulation script. The shell scripts handle the `NUMBA_NUM_THREADS` environment variable automatically, so you don't need to set it yourself.

## Project Structure

```
potential/       Physics: potential definitions
simulation/      Main simulation scripts
analysis/        Tunneling, action & potential analysis
postprocess/     Revisualization & GIF creation
utils/           Diagnostic & profiling tools
scripts/         Helper shell scripts (run from project root)
docs/            Complete documentation
data/            Simulation results & tunneling data
figs/            Generated figures
logs/            Simulation logs (auto-created)
```

See `docs/DIRECTORY_STRUCTURE.md` for full details.

## Setup

### Prerequisites

- Python 3.9+
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for managing environments)

### Installation

It is recommended to create a dedicated conda environment:

```bash
conda create -n phase_transition python=3.10
conda activate phase_transition
```

Install the required packages:

```bash
pip install numpy scipy matplotlib numba sympy pandas Pillow cosmoTransitions
```

### Package Summary

| Package | Purpose | Required by |
|---------|---------|-------------|
| `numpy` | Array operations, linear algebra | All scripts |
| `scipy` | Interpolation, optimization, integration | Simulation, analysis, postprocess |
| `matplotlib` | Plotting and visualization | Simulation, analysis, postprocess |
| `numba` | JIT compilation for performance | `simulation/latticeSimeRescale_numba.py`, `latticeSimComplex_numba.py` |
| `sympy` | Symbolic math (potential definitions) | `potential/`, some analysis scripts |
| `pandas` | CSV data handling | Analysis scripts |
| `Pillow` | GIF creation from PNG snapshots | `postprocess/make_gif.py` |
| `cosmoTransitions` | Tunneling calculation, thermal integrals | `potential/`, `simulation/`, `analysis/` |

### Optional

**For GPU acceleration (Linux/Windows with NVIDIA GPU only):**

```bash
pip install cupy-cuda11x  # for CUDA 11.x
# or
pip install cupy-cuda12x  # for CUDA 12.x
```

**For 3D lattice simulation with PyTorch:**

```bash
pip install torch
```

GPU acceleration and PyTorch are not required. The CPU-optimized Numba version is already excellent, especially on Apple Silicon.

## Post-Processing

After running a simulation, you can re-render snapshots or create animations:

```bash
# Re-render snapshots with different visualization settings
python postprocess/revisualize_snapshots.py data/lattice/set6/<run_dir>

# Detect and visualize cosmic strings (complex field only)
python postprocess/revisualize_snapshots.py data/lattice/set6/<run_dir> --strings

# Create a GIF animation
python postprocess/make_gif.py data/lattice/set6/<run_dir>
```

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/DIRECTORY_STRUCTURE.md` | File organization guide |
| `docs/PERFORMANCE_SUMMARY.md` | Complete performance overview |
| `docs/SPEED_OPTIMIZATIONS.md` | All optimization details |
| `docs/PRACTICAL_SPEEDUPS.md` | Additional speedup options |
| `docs/REVISUALIZATION_GUIDE.md` | Re-rendering & cosmic string visualization |
| `docs/MAC_GPU_OPTIONS.md` | macOS-specific guidance |
| `docs/GPU_SETUP.md` | CUDA GPU setup |

## Performance

- **Original implementation**: ~50 minutes (100k steps, 128x128 grid)
- **Current optimized**: ~2-3 minutes (~25x faster)

Optimizations: multi-core threading, Numba JIT, uniform-grid cubic splines (O(1) thermal integral lookup), in-kernel RNG, fused RK2 integrator, float32 precision.

## Diagnostic Tools

```bash
python utils/check_numba_threads.py    # Check threading setup
python utils/profile_detailed.py       # Profile performance
python utils/benchmark_comparison.py   # Compare implementations
python utils/check_float32_safety.py   # Verify float32 safety
```

## Configuration

Edit `simulation/latticeSimeRescale_numba.py` to adjust:

```python
USE_FUSED_RK2 = True   # Fused integrator
USE_NUMBA_RNG = True   # In-kernel random numbers
USE_FLOAT32 = True     # Memory-optimized precision

Nx, Ny = 128, 128      # Grid size
Nt = 100_000           # Total steps
T0 = 3000.0            # Initial temperature (GeV)
steps = 50_000         # Snapshot interval
```

## Data Organization

| Directory | Contents |
|-----------|----------|
| `data/lattice/` | Lattice simulation results (field states `.npz`, snapshots `.png`, bubble/string CSVs) |
| `data/tunneling/` | CosmoTransitions outputs (T-S action curves, coupling scans, bounce profiles) |
| `data/csv/toyModel/` | Reference toy model data |
| `figs/` | Generated analysis figures (potential plots, action curves, etc.) |

## License

This project is licensed under the [MIT License](LICENSE).
