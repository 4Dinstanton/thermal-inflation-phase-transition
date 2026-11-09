# Thermal-inflation lattice simulation: HPC run manual

Minimal instructions for reproducing our **standard production run (set8)**, the
configuration we ran most often on KISTI Nurion, at another supercomputing center.
For the full CLI reference see
[`cosmolattice_ext/RUNNING_AND_POTENTIAL.md`](../cosmolattice_ext/RUNNING_AND_POTENTIAL.md).

---

## 0. Standard configuration (this is the run to do)

| Item | Value |
|------|-------|
| Lattice | `Nx=1024` (1024³), `dx_phys=1e-3`, `dt_phys=1e-4` GeV⁻¹, `tMax=1000` (= 10,000 steps) |
| Potential | `V_correct`, `nb=nf=20`, `T0=1230` GeV, `γ=4.1667e-4` (`param_set=set8`) |
| Fields | complex scalar (`--n_scalars 2`), on-lattice GWs (`--with_gws`) |
| Langevin | `stochasticrk` evolver (numba scheme), friction `η ∝ T` (`--eta_follows_T`), switched off globally once the false-vacuum fraction ≤ 0.01 |
| Expansion | `staged` thermal inflation → matter → radiation, `f_switch=0.01`, `φ_esc=5e4` GeV, `T_rh=1000` GeV |
| Output | averages / spectra / GW text every 10 steps, 3D energy HDF5 every 200 steps, checkpoint every 1000 steps, 3D field snapshots every 4000 steps (every 100 steps once max\|φ\| > 5e4 GeV) |

These are the defaults of `simulation/run_tipt_gamma.sh`; you only set the
environment variables `NX` and `NP`.

---

## 1. Prerequisites

- C++14 compiler + MPI (KISTI: `gcc/8.3.0` + `openmpi/3.1.0`)
- **Parallel** HDF5 (KISTI: `hdf5-parallel/1.10.2`) and FFTW3 built with MPI (KISTI: `fftw_mpi/3.3.7`)
- `cmake` ≥ 3.12, `python3` ≥ 3.9 (the run scripts use only the standard library; post-processing needs `requirements.txt`)

```bash
git clone --recursive https://github.com/4Dinstanton/thermal-inflation-phase-transition.git PhaseTransition
cd PhaseTransition
git submodule update --init      # external/cosmolattice @ a5654d64 (pinned; our patches anchor to this version)
```

**`data/thermal_splines/thermal_tables.bin` is not in git** (`.gitignore`).
Put the file we send you (196 KB) at that path, or generate it yourself:
`pip install numpy scipy cosmoTransitions`, then `python3 tools/export_thermal_splines.py`.

---

## 2. Build (once, for the compute-node architecture)

If login and compute nodes have different CPUs, build **on a compute node**
(on Nurion, the SKL binary did not run on KNL nodes).

```bash
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <cmake> <python3>   # use your center's module names
export CC=mpicc CXX=mpicxx CXXFLAGS="-std=c++14"
export HDF5_ROOT=/path/to/hdf5-parallel        # prefix containing include/hdf5.h
export MYFFTW3_PATH=/path/to/fftw-mpi          # prefix containing include/fftw3-mpi.h

rm -rf external/cosmolattice/build_mpi
python3 simulation/run_cosmolattice.py --install --build --mpi --dry_run \
  --Nx 32 --tMax 1 --no_snapshots
ls -l external/cosmolattice/build_mpi/thermal_inflation    # success if this exists
```

`--install` symlinks the model headers into the CosmoLattice submodule and applies
the required patches automatically (Langevin evolver registration, snapshot hook,
HDF5 string length 256 → 1024, ...). Do not edit the submodule by hand.
Our KISTI build job is in `simulation/pbs_build_tipt_skl.pbs` as an example.

---

## 3. Smoke test (a few minutes)

At 256³ and below, raw snapshots are converted to NPZ after the run, so `numpy` is required.

```bash
NX=64 NP=8 TMAX=20 bash simulation/run_tipt_gamma.sh
tail data/lattice/set8/64x64x64_*_CL/run.log    # OK if you see "Step N done. ... T=... GeV" lines
```

---

## 4. Production run

### Standard CLI (what the script actually executes)

```bash
python3 simulation/run_cosmolattice.py \
  --mpi --np "$NP" --Nx 1024 --tMax 1000 \
  --dx_phys 1e-3 --dt_phys 1e-4 --T0 1230 --gamma 4.1667e-4 --param_set set8 \
  --potential_type V_correct --nb 20 --nf 20 \
  --eta_follows_T --n_scalars 2 --with_gws \
  --expansion_mode staged --expansion_f_switch 0.01 --expansion_phi_esc 5e4 --T_rh 1000 \
  --langevin_off_after_nucleation --langevin_off_f_switch 0.01 --langevin_off_phi_esc 5e4 \
  --phi_threshold 50000 --steps 4000 --steps_dense 100 \
  --tOutputFreq 1 --tOutputInfreq 1 --tOutputRareFreq 20 --backup_steps 1000 \
  --snapshot_format hdf5
```

The one-line equivalent is `NX=1024 NP=<ranks> bash simulation/run_tipt_gamma.sh`.
The runner uses the `mpirun` found on `PATH`, and it must be started **from the repo root**
(the output path in `input.in` is relative to the repo).

### Slurm template

```bash
#!/bin/bash
#SBATCH -J tipt1024
#SBATCH -N 16
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -t 96:00:00
#SBATCH -p <partition>
#SBATCH -o tipt1024.%j.log

REPO=/path/to/PhaseTransition
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <python3>
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
NX=1024 NP=$SLURM_NTASKS bash simulation/run_tipt_gamma.sh
```

### PBS template (as used at KISTI)

```bash
#!/bin/bash
#PBS -N tipt1024
#PBS -q <queue>
#PBS -l select=16:ncpus=64:mpiprocs=64:ompthreads=1
#PBS -l walltime=96:00:00
#PBS -j oe

REPO=/path/to/PhaseTransition
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <python3>
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
NX=1024 NP=1024 bash simulation/run_tipt_gamma.sh
```

If your center only allows `srun` (not `mpirun`), generate `input.in` inside the job with
`DRY_RUN=1 NX=1024 NP=... bash simulation/run_tipt_gamma.sh`, then launch the binary directly:

```bash
srun external/cosmolattice/build_mpi/thermal_inflation input=data/lattice/set8/<run_dir>/input.in
```

---

## 5. Resource estimates (measured at KISTI)

| Configuration | Result |
|---------------|--------|
| 1024³, 64 ranks on 1 KNL node | killed: out of memory |
| 1024³, 128 ranks on 4 SKL nodes (192 GB/node, 768 GB total) | killed: cgroup memory limit |
| 1024³, **1024 ranks** (KNL, 64 ranks/node) | ran fine; ~23–27 s/step, but **reached only t=642 (6,420 steps) in the 48 h walltime** |

- **Memory:** 768 GB total was not enough. Request at least 1.5 TB in total.
- **Domain decomposition:** slabs, `(NP 1 1)`. Use a divisor of 1024 for `NP` (256, 512, 1024) and do not exceed 1024.
- **Walltime:** the full 10,000 steps take ~75 h on KNL. Newer Xeon/EPYC nodes should be faster, but request **≥ 96 h or a long queue**.
- **Restart:** CosmoLattice writes checkpoints (`thermal_inflation_backup.h5`, every 1000 steps),
  but the model's expansion stage, bath temperature and Langevin-off state are not stored in them.
  Restarting is **not validated**; plan to finish the run in a single job.
- **Disk:** one real 1024³ field is 8 GiB. 3D snapshots (2 field components + 3 energy densities) accumulate,
  so reserve **several TB** of scratch.

---

## 6. Output and monitoring

The output directory is
`data/lattice/set8/1024x1024x1024_T0_1230_complex_..._etaT_langoff_f0.01_..._stochasticrk_V_correct_CL/`.

| File | Content |
|------|---------|
| `run.log` | live progress (`Step N done. Current time: t  T=... GeV`) |
| `average_energies.txt`, `average_scalar_*.txt`, `average_scale_factor.txt` | volume averages (every 10 steps) |
| `spectra_scalar_*.txt`, `spectra_gws.txt`, `energy_gws.txt` | spectra / GWs (every 10 steps) |
| `field_snapshot.h5`, `*_snapshot*.h5` | 3D fields / energies (large) |
| `input.in`, `cl_run_params.json` | parameters actually used |

```bash
tail -f data/lattice/set8/<run_dir>/run.log
export TIPT_LIVE_LOG=/scratch/<user>/tipt_live.log   # optional: set before the run to mirror the log to this file
```

When sending results back to us, send **all `.txt` files plus `input.in` and `run.log`** first;
we will arrange the `.h5` files separately. GW spectra can be plotted with
`python3 postprocess/plot_cl_gw_spectrum.py <run_dir> --physical --T_rh 1000`.
The 1024³ string analysis (`tools/cl_hdf5_string_pipeline.py`) needs ≥ 128 GB RAM on a single node.

---

## 7. Variants (only if needed)

| Purpose | How |
|---------|-----|
| fermion-only potential | `TIPT_POT=fermion_only` (sets T0=1600 automatically) |
| different γ | `GAMMA=1e-5` → `data/lattice/set_gamma_1e-05/` (never overwrites set8) |
| quick 256³ version | `NX=256 NP=64` (snapshots switch to raw automatically) |
| print the command only | `DRY_RUN=1` |

**Caution:** `DRY_RUN=1`, or re-running the same configuration, overwrites `input.in` and
`cl_run_params.json` in an existing run directory with the same name. To keep earlier results,
use the CLI from section 4 directly with `--run_name <name>`, or move the old run directory first.
