# CosmoLattice Langevin runs & changing the potential

Short guide for running CosmoLattice (`thermal_inflation`) with the Langevin
scheme, full CLI reference for `simulation/run_cosmolattice.py`, and which files
to edit for a different potential.

See also: [`README.md`](README.md) (architecture, install details, GW notes).

---

## 1. One-time setup

```bash
# From repo root
python tools/export_thermal_splines.py          # thermal J-tables → data/thermal_splines/
python simulation/run_cosmolattice.py --install --build
# MPI production (recommended for ≥256³):
python simulation/run_cosmolattice.py --install --build --mpi
```

Outputs land under `data/lattice/<param_set>/<run_dirname>_CL/`.

---

## 2. Minimal Langevin run

Langevin = custom evolver **`stochasticrk`**: friction η + FDT thermal noise
(plus optional Hubble `3H`).

```bash
python simulation/run_cosmolattice.py \
  --Nx 256 --T0 1230 --mphi 1000 --gamma 4.1667e-4 \
  --dx_phys 0.001 --dt_phys 0.0001 --tMax 940 \
  --n_scalars 2 --potential_type V_correct \
  --evolver stochasticrk --stochastic_scheme numba \
  --eta_follows_T --thermal_noise 1 \
  --expansion_mode staged --expansion_f_switch 0.01 \
  --expansion_phi_esc 50000 --T_rh 1000 \
  --snapshot_format hdf5 \
  --steps 4000 --phi_threshold 50000 --steps_dense 100 \
  --param_set set8 --mpi --np 8
```

### Langevin shutoff (optional)

| Goal | Flags |
|------|--------|
| Keep Langevin on whole run (default) | omit langoff flags, or `--langevin_off_mode none` |
| Global off after percolation | `--langevin_off_mode global --langevin_off_f_switch 0.01` |
| Per-site off | `--langevin_off_mode per_site --langevin_off_phi_esc 1e5` (requires `--stochastic_scheme numba`) |
| Legacy enable | `--langevin_off_after_nucleation` → same as global unless mode is set |

Global mode zeros η + FDT on the **whole box** when false-vac fraction ≤ `f_switch`.
Hubble damping `3H` is **never** turned off by Langevin shutoff.

**Recommendation:** prefer **global late shutoff** (`f_switch ~ 0.01`) over per-site
with a tiny `phi_esc ≪ φ₀` (that kills damping inside young bubbles and amplifies
wake ringing). Or omit shutoff and use `--eta_follows_T` so η cools with the bath.

---

## 3. HDF5 field dumps

```bash
--snapshot_format hdf5          # default; field_snapshot.h5 / MPI slabs
--steps 4000                    # coarse interval (also enables snapshots)
--phi_threshold 50000           # switch to dense when max|Φ| (GeV) exceeds this
--steps_dense 100               # dense interval after threshold
```

| Flag | Effect |
|------|--------|
| `--snapshot_format hdf5` | CosmoLattice HDF5 field dumps (default; safe at large N) |
| `--snapshot_format raw` | Gather to `field_states/snapshot_*.raw` (OK at 256³) |
| `--save_snapshots` | Force 3D dumps on even without `--steps` |
| `--no_snapshots` | Disable 3D field dumps |
| `--tBackupFreq` / `--backup_steps` | Checkpoint `thermal_inflation_backup.h5` (separate) |
| `--tOutputRareFreq` | Rare CosmoLattice 3D **energy** HDF5 (independent of `--steps`) |

Binary must be built with HDF5 (`--build`, usually with `--mpi`). If HDF5 is off,
HDF5 dumps no-op — use `--snapshot_format raw` or rebuild with HDF5.

---

## 4. Full CLI reference

All flags are from `simulation/run_cosmolattice.py`.

### Lattice & time

| Flag | Default | Description |
|------|---------|-------------|
| `--Nx` | `64` | Cubic lattice size |
| `--Ny`, `--Nz` | — | Unused (numba parity); CosmoLattice is cubic |
| `--dx_phys` | `1e-3` | Physical spacing (GeV⁻¹) |
| `--dt_phys` | `1e-4` | Physical time step (GeV⁻¹) |
| `--tMax` | `2000` | Max **program** time |
| `--tOutputFreq` | `10` | Frequent diagnostics interval (program time) |
| `--tOutputInfreq` | `100` | Infrequent: spectra / GW spectra |
| `--tOutputRareFreq` | `1000*dt` | Rare: 3D energy HDF5 |
| `--tBackupFreq` | off (`-1`) | Checkpoint interval (program time) |
| `--backup_steps` | — | Checkpoint every N lattice steps → sets `tBackupFreq` |

Program units: `omegaStar = mphi`, so
`dx_tilde = mphi·dx_phys`, `dt_tilde = mphi·dt_phys`.

### Snapshots & I/O

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | — | Coarse field-snapshot interval (lattice steps) |
| `--phi_threshold` | — | GeV; switch to dense dumps when `max|Φ|` exceeds this |
| `--steps_dense` | — | Dense interval after threshold |
| `--save_snapshots` | off† | Enable 3D φ dumps (†on automatically if `--steps` set) |
| `--no_snapshots` | — | Disable 3D dumps |
| `--snapshot_format` | `hdf5` | `hdf5` or `raw` |
| `--export_only` | — | Only convert existing dumps → NPZ (no sim) |
| `--run_dir` | — | Directory for `--export_only` |
| `--keep_raw` | — | Keep leftover `.raw` after NPZ export |
| `--no_export` | — | Skip automatic `.raw`→NPZ after run |
| `--out` | — | Explicit output directory |
| `--run_name` | — | Suffix / leaf name under `data/lattice/<param_set>/` |
| `--param_set` | `auto` | Folder under `data/lattice/`; `auto` from `--gamma` |
| `--force_param_set` | — | Do not remap `param_set` when it conflicts with γ |
| `--dry_run` | — | Write `.in`, print command; do not execute |
| `--install` | — | Symlink headers + register evolver in submodule |
| `--build` | — | `cmake` + `make` |
| `--mpi` | — | MPI build / `mpirun` |
| `--np` | CPU count | MPI ranks |

### Physics / potential parameters (CLI only)

| Flag | Default | Description |
|------|---------|-------------|
| `--T0` | `7350` | Initial bath temperature (GeV) |
| `--mphi` | `1000` | Scalar mass / μ (GeV); sets `omegaStar` |
| `--gamma` | `4.1667e-4` | `φ₀ = γ M_Pl`; sets `λ = mφ²/φ₀²` and default `ΔV` |
| `--potential_type` | `V_correct` | `V_correct` or `fermion_only` |
| `--nb`, `--nf` | `20` | Boson / fermion multiplicities |
| `--boson_coupling`, `--fermion_coupling` | `1.09` | Yukawa-like couplings |
| `--gauge` | `1.05` | Gauge coupling (boson & fermion) |
| `--include_cw` | `1` | Coleman–Weinberg force (`0` = numba-parity off) |

Tree potential used by the lattice (unchanged shape):

\[
V_{\rm tree} = \frac{\lambda}{4}\phi^4 - \frac{m_\phi^2}{2}\phi^2
\quad(+\;\Delta V\;\text{for Hubble})
\]

with \(\phi_0 = \sqrt{m_\phi^2/\lambda}\). Override vacuum energy with
`delV` in `input.in` if needed (set via parser when generating `.in`).

### Langevin / expansion

| Flag | Default | Description |
|------|---------|-------------|
| `--evolver` | `stochasticrk` | Evolver name |
| `--stochastic_scheme` | `numba` | `numba`, `fused_rk2`, `rk2_fused`, `fdt`, `nonfused_rk2`, `fused` |
| `--eta_phys` | `T0` | Langevin friction (GeV) |
| `--eta_follows_T` | off | `η(t) = η_phys · T(t)/T0` |
| `--thermal_noise` | `1` | FDT noise on/off |
| `--langevin_off_after_nucleation` | off | Enable shutoff (legacy → global) |
| `--langevin_off_mode` | `none`† | `none` / `global` / `per_site` (†global if legacy flag alone) |
| `--langevin_off_f_switch` | `0.99` | Global: off when false-vac frac ≤ this |
| `--langevin_off_phi_esc` | `expansion_phi_esc` | Escape \|Φ\| (GeV) for false-vac / per-site |
| `--noise_seed` | `1` | RNG seed for noise |
| `--no_hubble` | off | Fixed `T=T0`, no expansion |
| `--expansion_mode` | `legacy` | `legacy` or `staged` (ti→md→rd) |
| `--expansion_T_switch` | `0` | Enter MD when `T ≤` this (GeV); `0` = use fraction |
| `--expansion_f_switch` | `1e-5` | Enter MD when false-vac frac ≤ this |
| `--expansion_phi_esc` | `1e4` | Escape \|Φ\| for false-vac fraction |
| `--T_rh` | `0` | Reheating T (GeV); `0` = stay in MD |

### Initial conditions

| Flag | Default | Description |
|------|---------|-------------|
| `--kCutOff` | `4` | Spectral IC cutoff (CosmoLattice IC) |
| `--cosmolattice_ic` | off | Use CL spectral IC (default: numba-like tiny φ) |
| `--baseSeed` | `1` | IC seed |
| `--uniform_phi` | `0` | Uniform φ (GeV), π=0 (roll test); `0` = off |
| `--bubble_seed_phi` | `0` | Seed centre patch to this φ (GeV); `0` = off |
| `--bubble_seed_bg` | `0` | Background φ outside seed patch |
| `--bubble_seed_radius` | `0` | Patch half-width in cells |

### Complex field / strings / GW

| Flag | Default | Description |
|------|---------|-------------|
| `--n_scalars` | `1` | `1` = real; `2` = complex (φ₁+iφ₂) for strings |
| `--zn_order` | `0` | Z_N breaking order (`0` = pure U(1)) |
| `--zn_strength` | `0` | Z_N potential strength |
| `--zn_turn_on_T` | `0` | Activate Z_N below this T; `0` = always if order>0 |
| `--with_gws` | off | On-lattice GW sector |
| `--PS_type` | `1` | Power-spectrum normalization |
| `--PS_version` | `1` | PS algorithm version |
| `--GWprojectorType` | `2` | GW TT projector |
| `--deltaKBin` | `1` | Spectral bin width |

---

## 5. Changing the potential

### A. Same shape, different numbers → CLI only

| Quantity | Flag |
|----------|------|
| \(m_\phi\) (\(m^2\)) | `--mphi` |
| \(\lambda\) | `--gamma` (preferred) or set `lambda` in generated `.in` |
| \(\Delta V\) / \(V_0\) | derived from γ; override `delV` in `.in` if needed |
| Thermal content | `--potential_type`, `--nb`, `--nf`, couplings |

No C++ edit, no rebuild (unless you never built before).

### B. New functional form → edit potential code, then rebuild

Example: replace \(\phi^4\) with \(\phi^6\) (flaton-like
\(V = V_0 - \tfrac12 m^2\phi^2 + \lambda_6\phi^6\)).

| File | What to change |
|------|----------------|
| `cosmolattice_ext/models/thermal_tables.hpp` | Tree \(V\), \(V'\), \(V''\) (what the lattice force uses) |
| `cosmolattice_ext/models/thermal_inflation.h` | New params (e.g. `lambdaSix`), pass into table struct, `delV` / VEV / `fStar` logic |
| `cosmolattice_ext/models/thermal_force.h` | Only if TempLat force wiring changes |
| `simulation/run_cosmolattice.py` | New CLI flags → write into `input.in` |
| `potential/Potential.py` | Optional: keep Python / analysis in sync |
| `tools/export_thermal_splines.py` | Rebuild tables if thermal/CW pieces change |

Then:

```bash
python tools/export_thermal_splines.py   # if tables changed
python simulation/run_cosmolattice.py --build [--mpi]
```

**Do not edit for potential-only changes:**
`cosmolattice_ext/evolvers/stochasticrk.h` (Langevin EOM). Reuse the same Langevin
API (`etaPhys`, `maybeDisableLangevin()`, `langevinSiteWeightProg()`, …) from
`thermal_inflation.h`.

Python reference for a \(\phi^6\) tree: `potential/flatonPotential.py`
(analysis / formulas only; the running lattice reads C++).

---

## 6. Post-processing

```bash
# RAW → NPZ (if snapshot_format=raw or leftover raw)
python tools/export_cl_snapshots.py <run_dir>

# Strings from NPZ
python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --from-npz

# Or from HDF5 (when field_snapshot.h5 / per-step h5 exist)
python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --workers 8

# Revisualize
python postprocess/revisualize_snapshots.py <run_dir> --strings
```

---

## 7. File map (quick)

| Path | Role |
|------|------|
| `simulation/run_cosmolattice.py` | CLI → `input.in` → launch binary |
| `cosmolattice_ext/models/thermal_inflation.h` | Model + Langevin shutoff + expansion staging |
| `cosmolattice_ext/models/thermal_tables.hpp` | \(V,V',V''\) evaluator |
| `cosmolattice_ext/models/thermal_force.h` | TempLat force operators |
| `cosmolattice_ext/evolvers/stochasticrk.h` | Langevin RK + FDT noise |
| `cosmolattice_ext/measurements/field_snapshot.hpp` | 3D field dumps |
| `data/thermal_splines/thermal_tables.bin` | Tabulated thermal integrals |
