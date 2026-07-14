# CosmoLattice thermal-inflation extension

Project-specific extension to [CosmoLattice](https://github.com/cosmolattice/cosmolattice)
(added as the submodule `external/cosmolattice`) implementing:

- A **single real scalar** with the **full** finite-temperature effective
  potential `V_correct = V_tree + V_thermal + V_radiation + V_CW`
  (matching `potential/Potential.py`), via tabulated thermal integrals.
- **Inflation-style temperature** evolution `T = T0 / a` (legacy), or optional
  **staged post-PT expansion** (`ti → md → rd`) via `--expansion_mode staged`.
- A **custom stochastic evolver** (`stochasticrk`) adding Langevin friction and
  fluctuation-dissipation (FDT) thermal noise, so bubble nucleation by thermal
  fluctuations appears on the lattice — mirroring
  `simulation/latticeSimeRescale_numba.py` (`rk2_fused`).

## Files

| File | Role |
|------|------|
| `models/thermal_tables.hpp` | Standalone loader/evaluator of `V`, `V'`, `V''` from the binary J-tables (no CosmoLattice deps; unit-testable). |
| `models/thermal_force.h` | TempLat unary operators wrapping the per-site table lookup (like `exp()`), so the T-dependent potential plugs into CosmoLattice's symbolic EOM. |
| `models/thermal_inflation.h` | CosmoLattice model (`NScalars=2`, `n_scalars=1|2`), thermal potential + optional Z_N, snapshots, GW ghost refresh. |
| `evolvers/stochasticrk.h` | Custom `StochasticRK` evolver: fused-RK2 + friction + FDT noise + `T=T0/a` + on-lattice GW (`kickGWs`/`driftGWs`). |
| `measurements/field_snapshot.hpp` | 3D field snapshots (single or two-component) to `field_states/*.raw`. |
| `parameter-files/ti_boson_fermion.in`, `ti_fermion_only.in` | Example inputs (Set B / Set C). |
| `tests/test_thermal_tables.cpp` | Standalone C++ check of `thermal_tables.hpp`. |

The thermal tables are produced by `tools/export_thermal_splines.py` and written
to `data/thermal_splines/thermal_tables.bin`. Run that first.

## One-time setup

`simulation/run_cosmolattice.py --install` performs these steps automatically
(idempotent). They are listed here for reference / manual use.

### 1. Tables
```bash
python tools/export_thermal_splines.py        # -> data/thermal_splines/thermal_tables.bin
```

### 2. Register the model

CosmoLattice selects a model with `-DMODEL=<name>` and includes
`src/models/<name>.h`. Make the model and its helper headers visible:

```bash
ln -s ../../../cosmolattice_ext/models/thermal_inflation.h external/cosmolattice/src/models/
ln -s ../../../cosmolattice_ext/models/thermal_tables.hpp  external/cosmolattice/src/models/
ln -s ../../../cosmolattice_ext/models/thermal_force.h     external/cosmolattice/src/models/
```

### 3. Register the custom evolver (two small upstream edits)

The evolver dispatch is hard-coded in CosmoLattice, so two files need a few
lines added. The installer applies these between `// >>> thermal-inflation` /
`// <<< thermal-inflation` marker comments (idempotent).

`src/include/CosmoInterface/evolvers/evolvertype.h` — extend the enum and parser:
```cpp
enum EvolverType {LF, VV2, VV4, VV6, VV8, VV10, VV6_2, RK2, RK3_4, RK3_4_A,
                  STOCHASTICRK /* >>> thermal-inflation <<< */};
// ... in operator>> :
else if(tmp=="stochasticrk"||tmp=="STOCHASTICRK") eType=STOCHASTICRK; // >>> thermal-inflation <<<
```

`src/include/CosmoInterface/evolvers/evolver.h` — include, member, dispatch:
```cpp
#include "CosmoInterface/evolvers/stochasticrk.h"     // >>> thermal-inflation <<<
// constructor:
else if(type == STOCHASTICRK) srk = std::make_shared<StochasticRK<T>>(model, rPar); // >>> thermal-inflation
// evolve(): else if(type == STOCHASTICRK) srk->evolve(model, tMinust0);
// sync():   else if(type == STOCHASTICRK) srk->sync(model, tMinust0);
// member:   std::shared_ptr<StochasticRK<T>> srk;
```
and copy/symlink the evolver header:
```bash
ln -s ../../../../../../cosmolattice_ext/evolvers/stochasticrk.h \
      external/cosmolattice/src/include/CosmoInterface/evolvers/
```

## Build & run

```bash
cd external/cosmolattice && mkdir -p build && cd build
cmake -DMODEL=thermal_inflation ../
make cosmolattice
./thermal_inflation input=../../../cosmolattice_ext/parameter-files/ti_boson_fermion.in
```

Or simply use the wrapper:
```bash
python simulation/run_cosmolattice.py --install --build \
    --Nx 64 --T0 7350 --potential_type V_correct --gamma 4.1667e-4 --param_set set8
```

### MPI (multi-core / multi-process)

CosmoLattice scales via **MPI domain decomposition** (not OpenMP in the evolver).

**macOS:** Prefer **MPICH**. Homebrew OpenMPI 5’s `mpirun` (PRRTE) often segfaults on Apple Silicon.
Also rebuild FFTW against MPICH (Homebrew’s `fftw` bottle is OpenMPI-ABI):

```bash
brew unlink open-mpi          # if linked
brew install mpich && brew link mpich
# one-time: FFTW linked to MPICH → external/fftw_mpich/
python simulation/run_cosmolattice.py --install --build --mpi --dry_run
```

Run with MPI (`--np` must divide `Nx`; auto-picks largest valid rank count ≤ CPU count).
For `np=1` the binary is launched directly; for `np>1` uses a working `mpirun` (MPICH preferred):

```bash
python simulation/run_cosmolattice.py --mpi --np 8 \
  --Nx 256 --T0 1230 --tMax 1520 --param_set set8
```

Binary: `external/cosmolattice/build_mpi/thermal_inflation` (NOMPI build stays in `build/`).

## Staged post-PT expansion CLI

```bash
# Force ti→md at T=1200 GeV, then md→rd at T_rh=800 GeV
python simulation/run_cosmolattice.py \
  --Nx 64 --T0 1230 --tMax 400 --param_set staged_smoke \
  --expansion_mode staged --expansion_T_switch 1200 --T_rh 800 \
  --no_snapshots --dry_run

# Fraction-triggered MD (default f_switch=1e-5), no reheating
python simulation/run_cosmolattice.py \
  --Nx 256 --T0 1230 --tMax 1520 --param_set set8 \
  --expansion_mode staged --expansion_f_switch 1e-5 --expansion_phi_esc 1e4
```

## Notes / caveats

- **Full potential.** The model uses `V_correct` (tree + thermal + radiation +
  Coleman-Weinberg). `V_p_correct` alone is only tree+thermal. CW is included by
  default; set `include_cw = 0` for parity against the numba EOM (which omits CW).
- **Radiation double-count.** The radiation term `pi^2/30 g* T^4` in the
  potential and the `T^4/chi_g^2` term in the Friedmann `H` describe the same
  radiation energy. Use either CosmoLattice self-consistent expansion *or* the
  evolver's `H(T,delV)` (the latter mirrors numba) — not both. See
  `g_star_pot` vs `g_star_hubble`.
- **FDT noise amplitude** is `sqrt(2 eta_eff T dt / (mu^2 dx_phys^3))`, with
  `eta_eff = eta + 3H/mu` and `T = T0/a`, applied as half per RK2 half-step
  (identical structure to the numba `rk2_fused` kernel).
- The Langevin **algorithm** is validated independently of the C++ build in
  `tools/validate_langevin.py` (FDT/equipartition + numba parity).
- **Inflation expansion.** With self-consistent expansion the program-unit field
  energy gives a negligible `H`; the `StochasticRK` evolver therefore drives `a`
  with the prescribed `H(T, delV)` so `T = T0/a` cools as in the numba run.
- **Staged post-PT expansion** (`--expansion_mode staged`). After thermal
  inflation the background can switch:
  - **ti**: `H^2 = (T^4/χ_g² + ΔV)/(3 M_Pl²)`, `T = T0/a` (same as legacy).
  - **md**: at switch dump `ρ_m = ΔV`, then `ρ_m ∝ a⁻³`,
    `H = √(ρ_m/(3 M_Pl²))`, `T = T_sw (a_sw/a)^{3/2}`.
    Trigger: `T ≤ --expansion_T_switch` if that is >0, else false-vacuum
    fraction `≤ --expansion_f_switch` with escape `|φ|/ρ ≤ --expansion_phi_esc`.
  - **rd**: when bath `T ≤ --T_rh` (`T_rh=0` disables), dump `ρ_m → ρ_r` and
    set `T = (ρ_r / ((π²/30) g_*))^{1/4}`, then continue with radiation-only
    `H(T)` and `T ∝ 1/a`. (`T_rh` is the decay trigger, not the post-dump T.)
  Snapshot `manifest.csv` records `expansion_stage` (0=ti,1=md,2=rd) and `rho_m`.
- **Newer CMake.** If `cmake` rejects the upstream policy version, pass
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (the wrapper does this automatically).
- **Snapshots.** 3D phi snapshots are written to `field_states/*.raw` during the
  run (no HDF5 required), then converted to numba-compatible NPZ by
  `tools/export_cl_snapshots.py` (called automatically by `run_cosmolattice.py`
  when `--steps` is set). Use `postprocess/revisualize_snapshots.py` on the run
  directory as with numba output.

## Field snapshots and revisualization

```bash
# Run with snapshots every 50000 lattice steps
python simulation/run_cosmolattice.py \
  --Nx 64 --T0 1150 --tMax 400 --steps 50000 \
  --dx_phys 1e-3 --dt_phys 1e-4 --param_set set8

# Re-export NPZ from an existing run (without re-simulating)
python simulation/run_cosmolattice.py --export_only --run_dir data/lattice/set8/<run_dir>

# Revisualize (same tool as numba)
python postprocess/revisualize_snapshots.py data/lattice/set8/<run_dir> --mode normalized
python postprocess/revisualize_snapshots.py data/lattice/set8/<run_dir> --view3d --escape_phi 1e14
```

## Gravitational waves (`--with_gws`)

Enable CosmoLattice's built-in GW sector (anisotropic stress → auxiliary `fldGWs`):

```bash
python simulation/run_cosmolattice.py --with_gws --tOutputInfreq 50 \
  --Nx 64 --uniform_phi 1e11 --no_hubble --tMax 200 --param_set set8
```

Spectra are written to `spectra_gws.txt` and `energy_gws.txt` on infrequent measurement
steps. Plot with:

```bash
python postprocess/plot_cl_gw_spectrum.py data/lattice/set8/<run_dir>
```

**Caveat:** `StochasticRK` uses prescribed `H(T, delV)` for expansion while
`GWsPowerSpectrumMeasurer` normalizes with `Energies::rho(model)` — spectra may
need a consistency pass if amplitudes look off.

## Cosmic strings (postprocess)

CosmoLattice does **not** compute winding numbers in-simulation. For global U(1)
strings, run with `--n_scalars 2` (and optional `--zn_order`, `--zn_strength`,
`--zn_turn_on_T`). Export snapshots, then:

```bash
python tools/export_cl_snapshots.py data/lattice/set8/<run_dir>
python postprocess/revisualize_snapshots.py data/lattice/set8/<run_dir> --strings
# or batch:
python tools/compute_strings_cl.py data/lattice/set8/<run_dir>
```

With `n_scalars=1` (default nucleation runs), only a real scalar is evolved; string
detection is a no-op until two-component snapshots exist.

Input parameters: `save_snapshots = 1`, `snapshot_steps` (coarse), optional
`snapshot_steps_dense` + `phi_threshold` (GeV) for dense mode after nucleation.

Dense snapshots mirror numba: when `max|phi|` exceeds `phi_threshold`, the writer
switches from `snapshot_steps` to `snapshot_steps_dense` for the rest of the run.

Unit conversion on export: `phi_GeV = fldS_program * fStar`.

Output folder names mirror numba (`latticeSimeRescale_numba.py` / `latticeSimComplex_numba.py`) with a `_CL` suffix, e.g.:
`64x64x64_T0_7350_dx_0.001_dtphys_0.0001_interval_100000_3D_hubble_eta_7350_gb_1.09_gf_1.09_nb_20_nf_20_stochasticrk_V_correct_CL`

Complex-field runs (`--n_scalars 2`) insert `_complex` after `T0_*` (and `_ZN{n}` when `--zn_order > 0`), e.g.:
`64x64x64_T0_1230_complex_ZN6_dx_0.001_..._stochasticrk_V_correct_CL`

## End-to-end smoke test

```bash
python tools/smoke_test_cosmolattice.py --Nx 64 --T0 1150 --tMax 400
```
Starts just at the Set-B spinodal (~1150 GeV), cools through it via inflation
expansion with FDT noise on, and writes `figs/cosmolattice_smoke.png`
(`T(t)`, `a(t)`, `<phi>(t)`, `rms(phi)(t)`) plus a spectrum plot. A successful
run shows `rms(phi)` growing toward `phi0` as true-vacuum bubbles nucleate and
grow (`<phi>` stays near zero while both Z2 vacua are populated).
