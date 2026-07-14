#!/usr/bin/env python3
"""
CLI wrapper to build and run the CosmoLattice thermal-inflation model, mirroring
the argument style of simulation/latticeSimeRescale_numba.py.

It can (idempotently):
  --install : symlink the extension headers into the CosmoLattice submodule and
              register the custom `stochasticrk` evolver (two small upstream edits).
  --build   : run cmake -DMODEL=thermal_inflation && make cosmolattice.
  (default) : generate a run-specific .in file (from CLI overrides) and execute
              the compiled binary, writing outputs under data/lattice/{param_set}/.

Program-variable mapping (see CosmoLattice manual Sec. 4.1):
  omegaStar = mphi  ->  dx_tilde = mphi*dx_phys,  dt_tilde = mphi*dt_phys
  kIR = 2*pi / (N * dx_tilde)

Examples
--------
  # First time: tables + install + build + run a 64^3 smoke test
  python tools/export_thermal_splines.py
  python simulation/run_cosmolattice.py --install --build \
      --Nx 64 --T0 7350 --potential_type V_correct --gamma 4.1667e-4 \
      --dx_phys 1e-3 --dt_phys 1e-4 --tMax 2000 --param_set set8

  # Fermion-only (Set C)
  python simulation/run_cosmolattice.py --potential_type fermion_only --nb 0 --param_set set8
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(REPO, "external", "cosmolattice")
EXT = os.path.join(REPO, "cosmolattice_ext")
TABLE = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")

BUILD_DIR_NOMPI = "build"
BUILD_DIR_MPI = "build_mpi"
BINARY_NAME = "thermal_inflation"

# Prefer MPICH over OpenMPI on macOS: Homebrew OpenMPI 5's mpirun (PRRTE) segfaults
# on some Apple Silicon hosts. CosmoLattice MPI builds should use the matching FFTW.
FFTW_MPICH_PREFIX = os.path.join(REPO, "external", "fftw_mpich")
MPICH_HOMEBREW = "/opt/homebrew/opt/mpich"

MODEL_HEADERS = ["thermal_inflation.h", "thermal_tables.hpp", "thermal_force.h", "field_snapshot.hpp"]
EVOLVER_HEADER = "stochasticrk.h"
MEASUREMENT_HEADER = "field_snapshot.hpp"

MARK_OPEN = "// >>> thermal-inflation"
MARK_CLOSE = "// <<< thermal-inflation"


# ---------------------------------------------------------------------------
# Argument parsing (numba-compatible subset + CosmoLattice extras)
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run CosmoLattice thermal-inflation model")
    # lattice
    p.add_argument("--Nx", type=int, default=64, help="Lattice size per dimension (cubic)")
    p.add_argument("--Ny", type=int, default=None, help="(unused; CosmoLattice is cubic) kept for numba parity")
    p.add_argument("--Nz", type=int, default=None, help="(unused) kept for numba parity")
    p.add_argument("--dx_phys", type=float, default=1e-3, help="Physical lattice spacing (GeV^-1)")
    p.add_argument("--dt_phys", type=float, default=1e-4, help="Physical time step (GeV^-1)")
    # times
    p.add_argument("--tMax", type=float, default=2000.0, help="Max program time")
    p.add_argument("--tOutputFreq", type=float, default=10.0, help="Frequent-output interval (program time)")
    p.add_argument("--tOutputInfreq", type=float, default=100.0, help="Infrequent-output interval")
    p.add_argument("--steps", type=int, default=None,
                   help="Coarse snapshot interval in lattice iterations (numba --steps)")
    p.add_argument("--phi_threshold", type=float, default=None,
                   help="When max|phi| (GeV) exceeds this, switch to dense snapshots")
    p.add_argument("--steps_dense", type=int, default=None,
                   help="Dense snapshot interval after phi_threshold crossed (numba --steps_dense)")
    p.add_argument("--save_snapshots", action="store_true",
                   help="Enable 3D phi snapshots (default on when --steps is set)")
    p.add_argument("--no_snapshots", action="store_true", help="Disable 3D phi snapshots")
    p.add_argument("--export_only", action="store_true",
                   help="Only export raw snapshots in run dir to NPZ (no simulation)")
    p.add_argument("--keep_raw", action="store_true", help="Keep .raw files after NPZ export")
    p.add_argument("--run_dir", default=None, help="Explicit run directory for --export_only")
    # physics
    p.add_argument("--T0", type=float, default=7350.0, help="Initial temperature (GeV)")
    p.add_argument("--mphi", type=float, default=1000.0, help="Scalar mass / mu (GeV)")
    p.add_argument("--gamma", type=float, default=4.1667e-4, help="phi0 = gamma*M_Pl; sets lambda and delV")
    p.add_argument("--potential_type", choices=["V_correct", "fermion_only"], default="V_correct")
    p.add_argument("--nb", type=float, default=20.0, help="Boson multiplicity")
    p.add_argument("--nf", type=float, default=20.0, help="Fermion multiplicity")
    p.add_argument("--boson_coupling", type=float, default=1.09)
    p.add_argument("--fermion_coupling", type=float, default=1.09)
    p.add_argument("--gauge", type=float, default=1.05, help="Gauge coupling (boson and fermion)")
    p.add_argument("--include_cw", type=int, default=1, help="1=include Coleman-Weinberg force; 0=numba-parity")
    # Langevin / expansion
    p.add_argument("--eta_phys", type=float, default=None, help="Friction (GeV); default = T0")
    p.add_argument("--thermal_noise", type=int, default=1, help="1=FDT noise on, 0=deterministic")
    p.add_argument("--noise_seed", type=int, default=1)
    p.add_argument("--no_hubble", action="store_true", help="Fixed T=T0, no expansion")
    p.add_argument("--expansion_mode", choices=["legacy", "staged"], default="legacy",
                   help="legacy: H(T,delV)+T=T0/a; staged: ti→md→rd after PT")
    p.add_argument("--expansion_T_switch", type=float, default=0.0,
                   help="Enter matter era when T<=this (GeV); 0=use false-vac fraction")
    p.add_argument("--expansion_f_switch", type=float, default=1e-5,
                   help="Enter matter era when false-vac fraction <= this")
    p.add_argument("--expansion_phi_esc", type=float, default=1e4,
                   help="Escape |phi|/rho threshold (GeV) for false-vac fraction")
    p.add_argument("--T_rh", type=float, default=0.0,
                   help="Reheating T (GeV); 0=remain in matter era until tMax")
    p.add_argument("--evolver", default="stochasticrk",
                   help="CosmoLattice evolver name (default: stochasticrk)")
    p.add_argument("--stochastic_scheme", default="numba",
                   choices=["numba", "fdt", "fused"],
                   help="stochasticrk noise: numba (parity), fdt (sqrt2 equipartition), fused (legacy)")
    p.add_argument("--kCutOff", type=float, default=4.0, help="Initial-fluctuation cutoff")
    p.add_argument("--cosmolattice_ic", action="store_true",
                   help="Use CosmoLattice kCutOff spectral IC (default: numba phi=0.01 GeV, pi=0)")
    p.add_argument("--baseSeed", type=int, default=1)
    p.add_argument("--bubble_seed_phi", type=float, default=0.0,
                   help="Seed centre patch to this phi (GeV) after IC init (0=off)")
    p.add_argument("--bubble_seed_bg", type=float, default=0.0,
                   help="Background phi (GeV) outside patch when bubble_seed_phi is set")
    p.add_argument("--bubble_seed_radius", type=int, default=0,
                   help="Patch half-width in cells (0=centre site only, 2=5^3 cube)")
    p.add_argument("--uniform_phi", type=float, default=0.0,
                   help="Set all sites to uniform phi (GeV), pi=0 (roll test IC)")
    # GW / spectra
    p.add_argument("--with_gws", action="store_true",
                   help="Enable on-lattice GW evolution and gws spectrum output")
    p.add_argument("--PS_type", type=int, default=1, choices=[1, 2],
                   help="Power spectrum normalization type (CosmoLattice)")
    p.add_argument("--PS_version", type=int, default=1, choices=[1, 2, 3],
                   help="Power spectrum algorithm version")
    p.add_argument("--GWprojectorType", type=int, default=2, choices=[1, 2, 3],
                   help="GW TT projector type")
    p.add_argument("--deltaKBin", type=int, default=1, help="Spectral bin width")
    # complex-field / Z_N (global U(1) strings)
    p.add_argument("--n_scalars", type=int, default=1, choices=[1, 2],
                   help="Number of real scalar components (2 = complex phi1+i*phi2)")
    p.add_argument("--zn_order", type=int, default=0,
                   help="Z_N symmetry breaking order (0 = pure U(1))")
    p.add_argument("--zn_strength", type=float, default=0.0,
                   help="Z_N potential strength delta_V")
    p.add_argument("--zn_turn_on_T", type=float, default=0.0,
                   help="Activate Z_N below this T (GeV); 0 = always on if zn_order>0")
    # MPI parallelism
    p.add_argument("--mpi", action="store_true",
                   help="Use MPI build and launch via mpirun (requires --build --mpi once)")
    p.add_argument("--np", type=int, default=None, dest="mpi_np",
                   help="MPI ranks (default: logical CPU count when --mpi)")
    # orchestration
    p.add_argument("--param_set", default="set8", help="Output folder tag under data/lattice/")
    p.add_argument("--install", action="store_true", help="Install headers + register evolver in submodule")
    p.add_argument("--build", action="store_true", help="cmake + make the model")
    p.add_argument("--dry_run", action="store_true", help="Generate .in and print command; do not execute")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Install: symlink headers + register evolver (idempotent)
# ---------------------------------------------------------------------------
def _symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    print(f"  linked {os.path.relpath(dst, REPO)} -> {os.path.relpath(src, REPO)}")


def install():
    if not os.path.isdir(CL):
        sys.exit("ERROR: external/cosmolattice submodule not found. Run: git submodule update --init")
    models_dir = os.path.join(CL, "src", "models")
    evolvers_dir = os.path.join(CL, "src", "include", "CosmoInterface", "evolvers")
    measurements_dir = os.path.join(CL, "src", "include", "CosmoInterface", "measurements")
    print("Installing thermal-inflation extension into submodule:")
    for h in MODEL_HEADERS:
        src = os.path.join(EXT, "models" if h != "field_snapshot.hpp" else "measurements", h)
        _symlink(src, os.path.join(models_dir, h))
    _symlink(os.path.join(EXT, "measurements", MEASUREMENT_HEADER),
             os.path.join(measurements_dir, MEASUREMENT_HEADER))
    _register_evolver(evolvers_dir)
    _register_snapshot_measurer()
    _register_main_snapshot()
    _register_ghost_refresh_after_measure()
    _register_verbose_temperature()
    _register_numba_ic()
    _register_freeze_inactive_scalars()
    print("Install complete.")


def _register_numba_ic():
    """Apply numba-style ICs after CosmoLattice spectral init when ic_numba=1."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    if "applyNumbaInitialConditions" in text and "applyUniformPhi" not in text:
        text = text.replace(
            "        model.applyNumbaInitialConditions();  "
            + MARK_OPEN + " ic-numba " + MARK_CLOSE + "\n"
            "        model.applyBubbleSeed();  "
            + MARK_OPEN + " bubble-seed " + MARK_CLOSE,
            "        model.applyNumbaInitialConditions();  "
            + MARK_OPEN + " ic-numba " + MARK_CLOSE + "\n"
            "        model.applyUniformPhi();  "
            + MARK_OPEN + " uniform-phi " + MARK_CLOSE + "\n"
            "        model.applyBubbleSeed();  "
            + MARK_OPEN + " bubble-seed " + MARK_CLOSE,
        )
        with open(main_cpp, "w") as f:
            f.write(text)
        print(f"  patched {os.path.relpath(main_cpp, REPO)} [bubble seed hook]")
        return
    if "applyNumbaInitialConditions" in text:
        return
    old = (
        "        initializer.initialize(model, runParams);\n"
        "        // 2) We initialize the model.\n"
    )
    new = (
        "        initializer.initialize(model, runParams);\n"
        "        // 2) We initialize the model.\n"
        "        model.applyNumbaInitialConditions();  "
        + MARK_OPEN + " ic-numba " + MARK_CLOSE + "\n"
    )
    if old not in text:
        raise RuntimeError("cosmolattice.cpp IC anchor not found")
    text = text.replace(old, new)
    with open(main_cpp, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(main_cpp, REPO)} [numba IC hook]")


def _register_freeze_inactive_scalars():
    """Zero the second scalar component when n_scalars=1 (backward-compatible nucleation)."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    marker = f"{MARK_OPEN} freeze-inactive {MARK_CLOSE}"
    if "freezeInactiveScalars" in text:
        return
    anchor = (
        "        model.applyBubbleSeed();  "
        + MARK_OPEN + " bubble-seed " + MARK_CLOSE
    )
    insert = (
        anchor + "\n"
        f"        model.freezeInactiveScalars();  {marker}"
    )
    if anchor not in text:
        raise RuntimeError("cosmolattice.cpp freeze-inactive anchor not found")
    text = text.replace(anchor, insert)
    with open(main_cpp, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(main_cpp, REPO)} [freeze inactive scalar]")


def _register_main_snapshot():
    """Call field snapshots every lattice step (required for dense-mode switching)."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    marker = f"{MARK_OPEN} snapshot // <<< thermal-inflation"
    if "model.saveFieldSnapshotIfDue" in text:
        return
    old = (
        "        if(measurer.areWeMeasuring(i))\n"
        "        //We proceed to measure\n"
        "        {\n"
        "            evolver.sync(model, t - runParams.t0);\n"
        "            //Some evolvers like staggered leapfrog have fields and momenta which\n"
        "            //do not live at the same timesteps. Before measuring, we synchronize them.\n"
        "            measurer.measure(i, t, model);\n"
    )
    new = (
        "        if(measurer.areWeMeasuring(i))\n"
        "        //We proceed to measure\n"
        "        {\n"
        "            evolver.sync(model, t - runParams.t0);\n"
        "            //Some evolvers like staggered leapfrog have fields and momenta which\n"
        "            //do not live at the same timesteps. Before measuring, we synchronize them.\n"
        "        }\n\n"
        "        model.saveFieldSnapshotIfDue(i, t);  // >>> thermal-inflation snapshot // <<< thermal-inflation\n\n"
        "        if(measurer.areWeMeasuring(i))\n"
        "        {\n"
        "            measurer.measure(i, t, model);\n"
    )
    if old not in text:
        raise RuntimeError(f"cosmolattice.cpp snapshot anchor not found")
    text = text.replace(old, new)
    with open(main_cpp, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(main_cpp, REPO)} [per-step snapshot]")


def _register_ghost_refresh_after_measure():
    """Restore config space + ghost cells after in-place FFT spectra measurements."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    marker = f"{MARK_OPEN} ghost-refresh {MARK_CLOSE}"
    if marker in text:
        return
    anchor = "            // a measurement.\n        }\n\n        evolver.evolve(model, t - runParams.t0);"
    insert = (
        "            // a measurement.\n"
        f"            model.refreshFieldsAfterMeasurement();  {marker}\n"
        "        }\n\n        evolver.evolve(model, t - runParams.t0);"
    )
    if anchor not in text:
        raise RuntimeError("cosmolattice.cpp ghost-refresh anchor not found")
    text = text.replace(anchor, insert)
    with open(main_cpp, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(main_cpp, REPO)} [ghost refresh after measure]")


def _register_snapshot_measurer():
    """Legacy no-op: snapshots are hooked in cosmolattice.cpp main loop."""
    pass


def _register_verbose_temperature():
    """Add T to the periodic Step-done terminal message in measurer.h."""
    measurer = os.path.join(CL, "src", "include", "CosmoInterface", "measurements", "measurer.h")
    with open(measurer, "r") as f:
        text = f.read()
    if "model.currentT()" in text and "Step " in text:
        return
    old = 'sayMPI << "Step " << n << " done. Current time:" << t <<"\\n";'
    new = (
        'sayMPI << "Step " << n << " done. Current time: " << t\n'
        '                       << "  T=" << std::setprecision(6) << model.currentT() << " GeV\\n";'
    )
    if old not in text:
        raise RuntimeError(f"measurer.h verbose-output anchor not found")
    text = text.replace(old, new)
    with open(measurer, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(measurer, REPO)} [verbose T]")


def _patch_block(path, anchor, insert, tag):
    """Insert `insert` right after the line containing `anchor`, guarded by markers."""
    with open(path, "r") as f:
        text = f.read()
    marker = f"{MARK_OPEN} {tag} {MARK_CLOSE}"
    if marker in text:
        return False  # already patched
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"anchor not found in {path}: {anchor!r}")
    line_end = text.find("\n", idx) + 1
    block = f"{marker}\n{insert}\n"
    text = text[:line_end] + block + text[line_end:]
    with open(path, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(path, REPO)} [{tag}]")
    return True


def _register_evolver(evolvers_dir):
    etype = os.path.join(evolvers_dir, "evolvertype.h")
    evol = os.path.join(evolvers_dir, "evolver.h")
    # Enum value + parser (edited in place; guard on the actual enum token).
    with open(etype, "r") as f:
        t = f.read()
    if "RK3_4_A, STOCHASTICRK" not in t:
        t = t.replace(
            "RK3_4, RK3_4_A};",
            "RK3_4, RK3_4_A, STOCHASTICRK};",
        )
        t = t.replace(
            'else if(tmp.empty()){}',
            'else if(tmp=="stochasticrk"||tmp=="STOCHASTICRK") eType=STOCHASTICRK;\n'
            '        else if(tmp.empty()){}',
        )
        with open(etype, "w") as f:
            f.write(t)
        print(f"  patched {os.path.relpath(etype, REPO)} [enum+parser]")
    # 2) evolver.h: include, member, dispatch
    with open(evol, "r") as f:
        e = f.read()
    if "StochasticRK" not in e:
        e = e.replace(
            '#include "CosmoInterface/evolvers/velocityverlet.h"',
            '#include "CosmoInterface/evolvers/velocityverlet.h"\n'
            '#include "CosmoInterface/evolvers/stochasticrk.h"  ' + MARK_OPEN + ' include ' + MARK_CLOSE,
        )
        # constructor dispatch
        e = e.replace(
            "            if( type == LF){\n                lf = std::make_shared<LeapFrog<T>>(model, rPar);\n            }\n            else{",
            "            if( type == LF){\n                lf = std::make_shared<LeapFrog<T>>(model, rPar);\n            }\n"
            "            else if( type == STOCHASTICRK){ srk = std::make_shared<StochasticRK<T>>(model, rPar); }  " + MARK_OPEN + " ctor " + MARK_CLOSE + "\n            else{",
        )
        # evolve dispatch
        e = e.replace(
            "            if( type == LF){\n                lf->evolve(model, tMinust0);\n            }\n            else{",
            "            if( type == LF){\n                lf->evolve(model, tMinust0);\n            }\n"
            "            else if( type == STOCHASTICRK){ srk->evolve(model, tMinust0); }  " + MARK_OPEN + " evolve " + MARK_CLOSE + "\n            else{",
        )
        # sync dispatch
        e = e.replace(
            "            if(type == LF){\n                lf->sync(model, tMinust0);\n            }\n            else {",
            "            if(type == LF){\n                lf->sync(model, tMinust0);\n            }\n"
            "            else if( type == STOCHASTICRK){ srk->sync(model, tMinust0); }  " + MARK_OPEN + " sync " + MARK_CLOSE + "\n            else {",
        )
        # member
        e = e.replace(
            "        std::shared_ptr<VelocityVerlet<T> > vv;",
            "        std::shared_ptr<VelocityVerlet<T> > vv;\n"
            "        std::shared_ptr<StochasticRK<T> > srk;  " + MARK_OPEN + " member " + MARK_CLOSE,
        )
        with open(evol, "w") as f:
            f.write(e)
        print(f"  patched {os.path.relpath(evol, REPO)} [evolver dispatch]")


# ---------------------------------------------------------------------------
# Build / binary paths
# ---------------------------------------------------------------------------
def _logical_cpu_count():
    n = os.cpu_count()
    return n if n and n > 0 else 1


def build_dirname(mpi=False):
    return BUILD_DIR_MPI if mpi else BUILD_DIR_NOMPI


def binary_path(mpi=False):
    return os.path.join(CL, build_dirname(mpi), BINARY_NAME)


def _default_mpi_np(nx):
    """Largest MPI rank count <= CPU count that divides nx."""
    target = _logical_cpu_count()
    best = 1
    for np in range(1, min(target, nx) + 1):
        if nx % np == 0:
            best = np
    return best


def _mpirun_candidates():
    """Ordered mpirun paths: prefer Homebrew MPICH (works on this macOS), then PATH."""
    cands = []
    mpich_bin = os.path.join(MPICH_HOMEBREW, "bin", "mpirun")
    if os.path.isfile(mpich_bin) and os.access(mpich_bin, os.X_OK):
        cands.append(mpich_bin)
    which = shutil.which("mpirun")
    if which and which not in cands:
        cands.append(which)
    return cands


def _resolve_mpirun():
    """Return an mpirun that can spawn at least one rank, or None."""
    for mpirun in _mpirun_candidates():
        try:
            subprocess.run(
                [mpirun, "-np", "1", "true"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return mpirun
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    return None


def _check_mpirun():
    if _resolve_mpirun() is None:
        sys.exit(
            "ERROR: no working mpirun found.\n"
            "  On macOS, Homebrew OpenMPI 5 often segfaults in PRRTE/hwloc.\n"
            "  Prefer MPICH:\n"
            "    brew unlink open-mpi && brew install mpich && brew link mpich\n"
            "  Then rebuild CosmoLattice: python simulation/run_cosmolattice.py --build --mpi --dry_run\n"
            "  Or test single-rank without a launcher: --mpi --np 1"
        )


def _mpi_launch_cmd(binary, in_arg, mpi_np):
    """Build command line for an MPI CosmoLattice run."""
    if mpi_np == 1:
        # Singleton MPI rank: launch directly (no launcher needed).
        return [binary, in_arg]
    mpirun = _resolve_mpirun()
    if mpirun is None:
        sys.exit(
            "ERROR: mpirun failed a launch probe.\n"
            "  Homebrew OpenMPI 5 (PRRTE) is broken on some Macs — use MPICH instead:\n"
            "    brew unlink open-mpi && brew link mpich\n"
            "  Then rebuild: python simulation/run_cosmolattice.py --build --mpi --dry_run\n"
            "  For single-rank testing use: --mpi --np 1"
        )
    return [mpirun, "-np", str(mpi_np), binary, in_arg]


def _validate_mpi_np(nx, mpi_np):
    if mpi_np < 1:
        sys.exit(f"ERROR: --np must be >= 1 (got {mpi_np})")
    if nx % mpi_np != 0:
        sys.exit(
            f"ERROR: lattice N={nx} must be divisible by --np={mpi_np} for MPI decomposition"
        )
    # CosmoLattice uses 3D Cartesian split; prefer np that factorizes into a cube.
    side = round(mpi_np ** (1.0 / 3.0))
    if side ** 3 != mpi_np:
        print(
            f"WARNING: np={mpi_np} is not a perfect cube; CosmoLattice may still run "
            f"but prefer np in {{1,8,27,...}} for cubic decomposition."
        )


def build(mpi=False):
    build_dir = os.path.join(CL, build_dirname(mpi))
    # Wipe stale CMake cache when switching MPI implementations.
    if mpi and os.path.isdir(build_dir):
        cache = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.isfile(cache):
            with open(cache, encoding="utf-8", errors="ignore") as f:
                cache_txt = f.read()
            wants_mpich = os.path.isdir(FFTW_MPICH_PREFIX) and os.path.isdir(MPICH_HOMEBREW)
            linked_openmpi = "open-mpi" in cache_txt or "OpenMPI" in cache_txt
            if wants_mpich and linked_openmpi:
                print("Stale OpenMPI CMake cache detected; clearing build_mpi/ ...")
                shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    mpi_flag = "ON" if mpi else "OFF"
    print(f"Configuring + building CosmoLattice (MODEL=thermal_inflation, MPI={mpi_flag})...")

    cmake_cmd = [
        "cmake",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DMPI={mpi_flag}",
        "-DMODEL=thermal_inflation",
    ]
    env = os.environ.copy()
    if mpi:
        # Prefer MPICH + matching FFTW (Homebrew fftw is built against OpenMPI ABI).
        prefix_parts = []
        if os.path.isdir(FFTW_MPICH_PREFIX):
            cmake_cmd.append(f"-DMYFFTW3_PATH={FFTW_MPICH_PREFIX}")
            prefix_parts.append(FFTW_MPICH_PREFIX)
            print(f"Using MPICH FFTW at {FFTW_MPICH_PREFIX}")
        elif not os.path.isdir(FFTW_MPICH_PREFIX):
            print(
                "WARNING: external/fftw_mpich not found. Homebrew FFTW is OpenMPI-ABI;\n"
                "  if you use MPICH, rebuild FFTW into external/fftw_mpich first."
            )
        if os.path.isdir(MPICH_HOMEBREW):
            prefix_parts.append(MPICH_HOMEBREW)
            mpich_bin = os.path.join(MPICH_HOMEBREW, "bin")
            env["PATH"] = mpich_bin + os.pathsep + env.get("PATH", "")
            env.setdefault("CC", os.path.join(mpich_bin, "mpicc"))
            env.setdefault("CXX", os.path.join(mpich_bin, "mpicxx"))
        if prefix_parts:
            cmake_cmd.append(f"-DCMAKE_PREFIX_PATH={';'.join(prefix_parts)}")
    cmake_cmd.append("..")

    subprocess.check_call(cmake_cmd, cwd=build_dir, env=env)
    subprocess.check_call(["make", "cosmolattice", "-j"], cwd=build_dir, env=env)
    print(f"Build complete: {os.path.relpath(binary_path(mpi), REPO)}")


# ---------------------------------------------------------------------------
# Generate run .in and execute
# ---------------------------------------------------------------------------
def make_input(args, out_dir):
    N = args.Nx
    mu = args.mphi
    dx_tilde = mu * args.dx_phys
    dt_tilde = mu * args.dt_phys
    kIR = 2.0 * math.pi / (N * dx_tilde)
    eta = args.eta_phys if args.eta_phys is not None else args.T0
    expansion = "false" if args.no_hubble else "true"

    save_snaps = _snapshots_enabled(args)
    coarse_steps = _snapshot_steps(args)

    lines = [
        "#Output",
        f"outputfile = {out_dir}/",
        "",
        "#Evolution",
        f"expansion = {expansion}",
        f"evolver = {args.evolver}",
        "",
        "#Lattice",
        f"N = {N}",
        f"dt = {dt_tilde:.10g}",
        f"kIR = {kIR:.10g}",
        "",
        "#Times",
        f"tOutputFreq = {args.tOutputFreq:g}",
        f"tOutputInfreq = {args.tOutputInfreq:g}",
        f"tMax = {args.tMax:g}",
        "",
        "#Field snapshots (phi -> field_states/*.raw; export via tools/export_cl_snapshots.py)",
        f"save_snapshots = {1 if save_snaps else 0}",
        f"snapshot_steps = {coarse_steps}",
    ]
    if args.phi_threshold is not None:
        lines.append(f"phi_threshold = {args.phi_threshold:g}")
    if args.steps_dense is not None:
        lines.append(f"snapshot_steps_dense = {args.steps_dense}")
    lines += [
        "#IC",
        f"kCutOff = {args.kCutOff:g}",
        f"baseSeed = {args.baseSeed}",
        "initial_amplitudes = 0.0",
        "initial_momenta = 0.0",
        "",
        "#Spectra / GWs",
        f"PS_type = {args.PS_type}",
        f"PS_version = {args.PS_version}",
        f"withGWs = {'true' if args.with_gws else 'false'}",
        f"GWprojectorType = {args.GWprojectorType}",
        f"deltaKBin = {args.deltaKBin}",
        "",
        "#Thermal-inflation model",
        f"potential_type = {args.potential_type}",
        f"mphi = {args.mphi:g}",
        f"gamma = {args.gamma:g}",
        f"boson_coupling = {args.boson_coupling:g}",
        f"boson_gauge_coupling = {args.gauge:g}",
        f"fermion_coupling = {args.fermion_coupling:g}",
        f"fermion_gauge_coupling = {args.gauge:g}",
        "boson_mass_squared = 1.0e6",
        f"nb = {0 if args.potential_type == 'fermion_only' else args.nb:g}",
        f"nf = {args.nf:g}",
        "g_star_pot = 100.0",
        "g_star_hubble = 106.75",
        "",
        "#Temperature / Langevin",
        f"T0 = {args.T0:g}",
        f"eta_phys = {eta:g}",
        f"dx_phys = {args.dx_phys:g}",
        f"dt_phys = {args.dt_phys:g}",
        f"include_cw = {args.include_cw}",
        f"thermal_noise = {args.thermal_noise}",
        f"noise_seed = {args.noise_seed}",
        f"ic_numba = {0 if (args.cosmolattice_ic or args.bubble_seed_phi > 0 or args.uniform_phi > 0) else 1}",
        f"uniform_phi = {args.uniform_phi:g}",
        f"bubble_seed_phi = {args.bubble_seed_phi:g}",
        f"bubble_seed_bg = {args.bubble_seed_bg:g}",
        f"bubble_seed_radius = {args.bubble_seed_radius}",
        f"stochastic_scheme = {getattr(args, 'stochastic_scheme', 'numba')}",
        f"thermal_table = {TABLE}",
        f"n_scalars = {args.n_scalars}",
        f"zn_order = {args.zn_order}",
        f"zn_strength = {args.zn_strength:g}",
        f"zn_turn_on_T = {args.zn_turn_on_T:g}",
        "",
        "#Post-PT expansion staging (ti → md → rd)",
        f"expansion_mode = {args.expansion_mode}",
        f"expansion_T_switch = {args.expansion_T_switch:g}",
        f"expansion_f_switch = {args.expansion_f_switch:g}",
        f"expansion_phi_esc = {args.expansion_phi_esc:g}",
        f"T_rh = {args.T_rh:g}",
        "",
    ]
    return "\n".join(lines)


def _snapshots_enabled(args):
    if args.no_snapshots:
        return False
    if args.save_snapshots or args.steps is not None:
        return True
    return False


def _t_snapshot_freq(args, dt_tilde):
    """Legacy helper; snapshots now use snapshot_steps directly in the .in file."""
    if args.steps is not None and args.steps > 0:
        return args.steps * dt_tilde
    return args.tOutputInfreq


def _snapshot_steps(args):
    return args.steps if args.steps is not None else max(1, int(round(args.tOutputInfreq / (args.mphi * args.dt_phys))))


def write_run_params(args, out_dir):
    import json
    mu = args.mphi
    M_PL = 2.4e18
    phi0 = args.gamma * M_PL
    lam = mu * mu / (phi0 * phi0)
    steps = args.steps if args.steps is not None else 100_000
    params = {
        "Nx": args.Nx, "Ny": args.Nx, "Nz": args.Nx,
        "dx_phys": args.dx_phys, "dt_phys": args.dt_phys,
        "mphi": args.mphi, "lam": lam, "gamma": args.gamma,
        "vev": math.sqrt(mu * mu / lam),
        "T0": args.T0,
        "eta_phys": args.eta_phys if args.eta_phys is not None else args.T0,
        "nb": 0 if args.potential_type == "fermion_only" else args.nb,
        "nf": args.nf,
        "boson_coupling": args.boson_coupling,
        "fermion_coupling": args.fermion_coupling,
        "potential_type": args.potential_type,
        "no_hubble": args.no_hubble,
        "expansion_mode": args.expansion_mode,
        "expansion_T_switch": args.expansion_T_switch,
        "expansion_f_switch": args.expansion_f_switch,
        "expansion_phi_esc": args.expansion_phi_esc,
        "T_rh": args.T_rh,
        "integrator": f"{args.evolver}_CL",
        "steps": steps,
        "phi_threshold": args.phi_threshold,
        "steps_dense": args.steps_dense,
        "Nt": int(round(args.tMax / (mu * args.dt_phys))),
        "total_time": args.tMax,
        "tMax": args.tMax,
        "with_gws": args.with_gws,
        "n_scalars": args.n_scalars,
        "zn_order": args.zn_order,
        "zn_strength": args.zn_strength,
        "zn_turn_on_T": args.zn_turn_on_T,
        "mpi": bool(args.mpi),
        "mpi_np": args.mpi_np if args.mpi else 1,
    }
    path = os.path.join(out_dir, "cl_run_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def export_snapshots(run_dir, keep_raw=False):
    export_script = os.path.join(REPO, "tools", "export_cl_snapshots.py")
    cmd = [sys.executable, export_script, run_dir]
    if keep_raw:
        cmd.append("--keep-raw")
    subprocess.check_call(cmd)


def output_dirname(args):
    """Match latticeSimeRescale_numba.py save_path naming, with a _CL suffix."""
    N = args.Nx
    steps = args.steps if args.steps is not None else 100_000
    hubble_tag = "_nohubble" if args.no_hubble else "_hubble"
    staged_tag = "_staged" if args.expansion_mode == "staged" else ""
    eta = args.eta_phys if args.eta_phys is not None else args.T0
    eta_tag = f"_eta_{eta:g}"
    nb = 0 if args.potential_type == "fermion_only" else args.nb
    coupling_tag = (
        f"_gb_{args.boson_coupling:g}_gf_{args.fermion_coupling:g}"
        f"_nb_{nb:g}_nf_{args.nf:g}"
    )
    integrator_tag = f"_{args.evolver}"
    pot_type_tag = f"_{args.potential_type}" if args.potential_type != "V_p" else ""
    field_tag = "_complex" if args.n_scalars >= 2 else ""
    zn_tag = f"_ZN{args.zn_order}" if args.n_scalars >= 2 and args.zn_order > 0 else ""
    return (
        f"{N}x{N}x{N}_T0_{int(args.T0)}{field_tag}{zn_tag}"
        f"_dx_{args.dx_phys:g}_dtphys_{args.dt_phys:g}"
        f"_interval_{steps}_3D{hubble_tag}{staged_tag}{eta_tag}{coupling_tag}"
        f"{integrator_tag}{pot_type_tag}_CL"
    )


def main():
    args = parse_args()

    if args.export_only:
        run_dir = args.run_dir
        if run_dir is None:
            out_root = os.path.join(REPO, "data", "lattice", args.param_set)
            run_dir = os.path.join(out_root, output_dirname(args))
        if not os.path.isdir(run_dir):
            sys.exit(f"ERROR: run directory not found: {run_dir}")
        export_snapshots(run_dir, keep_raw=args.keep_raw)
        return

    if not os.path.exists(TABLE):
        print(f"NOTE: thermal table {os.path.relpath(TABLE, REPO)} not found.")
        print("      Run: python tools/export_thermal_splines.py")
        if not args.dry_run:
            sys.exit(1)

    if args.install:
        install()
    if args.build:
        build(mpi=args.mpi)

    mpi_np = args.mpi_np
    if args.mpi:
        if mpi_np is None:
            mpi_np = _default_mpi_np(args.Nx)
            print(f"MPI ranks: auto-selected np={mpi_np} (Nx={args.Nx}, cpus={_logical_cpu_count()})")
        _validate_mpi_np(args.Nx, mpi_np)
        args.mpi_np = mpi_np
        if mpi_np > 1:
            _check_mpirun()

    out_root = os.path.join(REPO, "data", "lattice", args.param_set)
    out_dir = os.path.join(out_root, output_dirname(args))
    os.makedirs(out_dir, exist_ok=True)

    in_text = make_input(args, out_dir)
    in_path = os.path.join(out_dir, "input.in")
    with open(in_path, "w") as f:
        f.write(in_text)
    write_run_params(args, out_dir)
    print(f"Wrote input file: {os.path.relpath(in_path, REPO)}")

    binary = binary_path(mpi=args.mpi)
    in_arg = f"input={in_path}"
    if args.mpi:
        cmd = _mpi_launch_cmd(binary, in_arg, args.mpi_np)
    else:
        cmd = [binary, in_arg]
    print("Run command:\n  " + " ".join(cmd))

    if args.dry_run:
        print("(dry run; not executing)")
        return
    if not os.path.exists(binary):
        hint = "--install --build --mpi" if args.mpi else "--install --build"
        print(f"ERROR: binary not found: {binary}\n  Run with {hint} first.")
        sys.exit(1)
    subprocess.check_call(cmd)

    if _snapshots_enabled(args):
        print("Exporting field snapshots to numba NPZ format...")
        export_snapshots(out_dir, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
