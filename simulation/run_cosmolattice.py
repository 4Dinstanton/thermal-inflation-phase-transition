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

  # Different γ → auto set name (setA / set_gamma_1e-5), never overwrites set8
  python simulation/run_cosmolattice.py --gamma 1e-5 --Nx 256 --with_gws ...
  # or: bash simulation/run_tipt_gamma.sh   (GAMMA=1e-5)
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(REPO, "external", "cosmolattice")
EXT = os.path.join(REPO, "cosmolattice_ext")
TABLE = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")

# γ → data/lattice/<set>/ mapping (avoids clobbering set8 on a γ scan)
sys.path.insert(0, os.path.join(REPO, "simulation"))
try:
    from gamma_sets import (  # noqa: E402
        SET8_GAMMA,
        format_gamma_tag,
        gammas_close,
        resolve_param_set,
        set_name_for_gamma,
        v0_of_gamma,
    )
except ImportError:  # pragma: no cover
    resolve_param_set = None
    SET8_GAMMA = 4.1667e-4
    gammas_close = lambda a, b, rel=1e-4: abs(a - b) / max(a, b) <= rel  # noqa: E731
    format_gamma_tag = lambda g: f"{g:.4g}".replace("+", "")  # noqa: E731
    set_name_for_gamma = lambda g: "set8" if gammas_close(g, SET8_GAMMA) else f"set_gamma_{format_gamma_tag(g)}"
    v0_of_gamma = None

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
    p.add_argument("--tOutputInfreq", type=float, default=100.0,
                   help="Infrequent-output interval (program time): spectra, GW spectra")
    p.add_argument("--tOutputRareFreq", type=float, default=None,
                   help="Rare-output interval (program time): 3D energy HDF5. "
                        "Default: CosmoLattice 1000*dt. Independent of --steps.")
    p.add_argument("--tBackupFreq", type=float, default=None,
                   help="CosmoLattice checkpoint interval (program time). "
                        "Writes thermal_inflation_backup.h5. Default off (-1).")
    p.add_argument("--backup_steps", type=int, default=None,
                   help="Checkpoint every this many lattice steps "
                        "(sets tBackupFreq = backup_steps * mphi * dt_phys).")
    p.add_argument("--steps", type=int, default=None,
                   help="Coarse snapshot interval in lattice iterations (numba --steps)")
    p.add_argument("--phi_threshold", type=float, default=None,
                   help="When max|phi| (GeV) exceeds this, switch to dense snapshots")
    p.add_argument("--steps_dense", type=int, default=None,
                   help="Dense snapshot interval after phi_threshold crossed (numba --steps_dense)")
    p.add_argument("--save_snapshots", action="store_true",
                   help="Enable 3D phi snapshots (default on when --steps is set)")
    p.add_argument("--no_snapshots", action="store_true", help="Disable 3D phi snapshots")
    p.add_argument("--snapshot_format", choices=["hdf5", "raw"], default="hdf5",
                   help="3D field dump: hdf5=local MPI slabs (safe at 1024^3); "
                        "raw=classical gather to field_states/*.raw (OK at 256^3)")
    p.add_argument("--export_only", action="store_true",
                   help="Only export raw snapshots in run dir to NPZ (no simulation)")
    p.add_argument("--keep_raw", action="store_true",
                   help="Keep leftover .raw files after NPZ export (legacy gather dumps)")
    p.add_argument("--no_export", action="store_true",
                   help="Skip .raw->NPZ after the run")
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
    p.add_argument("--eta_follows_T", action="store_true",
                   help="Scale friction with the bath: eta_phys(t) = eta_phys * T(t)/T0 "
                        "(default off: eta frozen at T0). With default eta_phys=T0 this is eta~T.")
    p.add_argument("--thermal_noise", type=int, default=1, help="1=FDT noise on, 0=deterministic")
    p.add_argument("--langevin_off_after_nucleation", action="store_true",
                   help="Zero Langevin η and FDT noise (Hubble 3H kept) when "
                        "false-vac fraction <= --langevin_off_f_switch")
    p.add_argument("--langevin_off_f_switch", type=float, default=0.99,
                   help="Langevin-off when false-vac fraction <= this "
                        "(0.99 = early bubbles; ~0.1 = bulk converted)")
    p.add_argument("--langevin_off_phi_esc", type=float, default=None,
                   help="Escape |phi| (GeV) for Langevin-off fraction; "
                        "default = --expansion_phi_esc")
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
                   help="Post–flaton-decay reheating T (GeV); bath cools in MD "
                        "until T<=T_rh then RD with that T. 0=stay in MD")
    p.add_argument("--evolver", default="stochasticrk",
                   help="CosmoLattice evolver name (default: stochasticrk)")
    p.add_argument("--stochastic_scheme", default="numba",
                   choices=["numba", "fused_rk2", "rk2_fused", "fdt", "nonfused_rk2", "fused"],
                   help="stochasticrk scheme: "
                        "numba/fused_rk2 = 4-pass RK2, independent 0.5*sigma kicks (inline); "
                        "rk2_fused = 4-pass RK2, same z both half-steps (Numba rk2_fused, full FDT); "
                        "fdt = independent-z fused RK2, half-kicks *sqrt(2); "
                        "nonfused_rk2 = 2-pass RK2, one full-dt FDT kick; "
                        "fused = legacy symplectic kernel")
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
    p.add_argument(
        "--param_set",
        default="auto",
        help="Output folder under data/lattice/<param_set>/. "
             "'auto' (default) picks set8 / setA / set_gamma_<γ> from --gamma. "
             "Passing set8 with a non-set8 γ is remapped unless --force_param_set.",
    )
    p.add_argument(
        "--force_param_set",
        action="store_true",
        help="Do not remap --param_set when it conflicts with --gamma "
             "(can overwrite an existing set; use with care).",
    )
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


def _restore_stock_u1_initializer():
    """Atakan pqera writes csFluctScaleRe/Im into u1initializer.h; thermal_inflation
    does not have those members. Restore stock if the pqera patch is present."""
    dst = os.path.join(
        CL, "src", "include", "CosmoInterface", "initializers", "u1initializer.h"
    )
    if not os.path.isfile(dst):
        return
    with open(dst, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if "csFluctScaleRe" not in text:
        return
    candidates = [
        dst + ".stock_before_atakan-pqera",
        os.path.join(EXT, "stock", "u1initializer.h"),
    ]
    src = next((p for p in candidates if os.path.isfile(p)), None)
    if src is None:
        print(
            "  ERROR: u1initializer.h still has Atakan pqera csFluctScaleRe/Im,\n"
            "    and no stock copy was found. Upload cosmolattice_ext/stock/"
            "u1initializer.h"
        )
        sys.exit(1)
    shutil.copy2(src, dst)
    print("  restored stock u1initializer.h (removed Atakan pqera patch)")


def install():
    if not os.path.isdir(CL):
        sys.exit("ERROR: external/cosmolattice submodule not found. Run: git submodule update --init")
    models_dir = os.path.join(CL, "src", "models")
    evolvers_dir = os.path.join(CL, "src", "include", "CosmoInterface", "evolvers")
    measurements_dir = os.path.join(CL, "src", "include", "CosmoInterface", "measurements")
    print("Installing thermal-inflation extension into submodule:")
    _restore_stock_u1_initializer()
    for h in MODEL_HEADERS:
        src = os.path.join(EXT, "models" if h != "field_snapshot.hpp" else "measurements", h)
        _symlink(src, os.path.join(models_dir, h))
    _symlink(os.path.join(EXT, "measurements", MEASUREMENT_HEADER),
             os.path.join(measurements_dir, MEASUREMENT_HEADER))
    _symlink(os.path.join(EXT, "evolvers", EVOLVER_HEADER),
             os.path.join(evolvers_dir, EVOLVER_HEADER))
    _register_evolver(evolvers_dir)
    _register_snapshot_measurer()
    _register_main_snapshot()
    _register_ghost_refresh_after_measure()
    _register_verbose_temperature()
    _register_hdf5_param_string_limit()
    _register_numba_ic()
    _register_freeze_inactive_scalars()
    print("Install complete.")


def _patch_after_hook(text, needle, insert_line, label):
    """Insert insert_line immediately after the first line containing needle."""
    if insert_line.strip() in text:
        return text, False
    idx = text.find(needle)
    if idx < 0:
        return text, False
    eol = text.find("\n", idx)
    if eol < 0:
        return text, False
    text = text[:eol + 1] + insert_line + text[eol + 1:]
    return text, True


def _register_numba_ic():
    """Apply numba-style ICs after CosmoLattice spectral init when ic_numba=1."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    orig = text

    numba_line = (
        "        model.applyNumbaInitialConditions();  "
        + MARK_OPEN + " ic-numba " + MARK_CLOSE + "\n"
    )
    uniform_line = (
        "        model.applyUniformPhi();  "
        + MARK_OPEN + " uniform-phi " + MARK_CLOSE + "\n"
    )
    bubble_line = (
        "        model.applyBubbleSeed();  "
        + MARK_OPEN + " bubble-seed " + MARK_CLOSE + "\n"
    )

    if "applyNumbaInitialConditions" not in text:
        old = (
            "        initializer.initialize(model, runParams);\n"
            "        // 2) We initialize the model.\n"
        )
        if old not in text:
            raise RuntimeError("cosmolattice.cpp IC anchor not found")
        text = text.replace(old, old + numba_line, 1)
        print(f"  patched {os.path.relpath(main_cpp, REPO)} [numba IC hook]")

    text, added = _patch_after_hook(
        text, "model.applyNumbaInitialConditions();", uniform_line, "uniform-phi")
    if added:
        print(f"  patched {os.path.relpath(main_cpp, REPO)} [uniform phi hook]")

    after = "model.applyUniformPhi();" if "applyUniformPhi" in text else "model.applyNumbaInitialConditions();"
    text, added = _patch_after_hook(text, after, bubble_line, "bubble-seed")
    if added:
        print(f"  patched {os.path.relpath(main_cpp, REPO)} [bubble seed hook]")

    if text != orig:
        with open(main_cpp, "w") as f:
            f.write(text)


def _register_freeze_inactive_scalars():
    """Zero the second scalar component when n_scalars=1 (backward-compatible nucleation)."""
    main_cpp = os.path.join(CL, "src", "cosmolattice.cpp")
    with open(main_cpp, "r") as f:
        text = f.read()
    if "freezeInactiveScalars" in text:
        return
    freeze_line = (
        "        model.freezeInactiveScalars();  "
        + MARK_OPEN + " freeze-inactive " + MARK_CLOSE + "\n"
    )
    for after in (
        "model.applyBubbleSeed();",
        "model.applyUniformPhi();",
        "model.applyNumbaInitialConditions();",
    ):
        text, added = _patch_after_hook(text, after, freeze_line, "freeze-inactive")
        if added:
            with open(main_cpp, "w") as f:
                f.write(text)
            print(f"  patched {os.path.relpath(main_cpp, REPO)} [freeze inactive scalar]")
            return
    raise RuntimeError("cosmolattice.cpp freeze-inactive anchor not found")


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
    """Add T to the Step-done message, then std::cout.flush() (not << std::flush).

    sayMPI is TempLat::StreamCacher, which does not accept iostream manipulators.
    Flush cout after the sayMPI statement so PBS/tee sees Step lines promptly.
    """
    measurer = os.path.join(CL, "src", "include", "CosmoInterface", "measurements", "measurer.h")
    with open(measurer, "r") as f:
        text = f.read()
    patched = (
        'sayMPI << "Step " << n << " done. Current time: " << t\n'
        '                       << "  T=" << std::setprecision(6) << model.currentT() << " GeV";\n'
        '                // sayMPI is StreamCacher (not ostream); flush cout after it dumps.\n'
        '                std::cout.flush();'
    )
    if "model.currentT()" in text and "std::cout.flush()" in text and "Step " in text:
        return
    stock = 'sayMPI << "Step " << n << " done. Current time:" << t <<"\\n";'
    # Broken patch from an earlier attempt (StreamCacher rejects << std::flush).
    broken_flush = (
        'sayMPI << "Step " << n << " done. Current time: " << t\n'
        '                       << "  T=" << std::setprecision(6) << model.currentT() << " GeV\\n"\n'
        '                       << std::flush;'
    )
    old_t = (
        'sayMPI << "Step " << n << " done. Current time: " << t\n'
        '                       << "  T=" << std::setprecision(6) << model.currentT() << " GeV\\n";'
    )
    old_t_no_nl = (
        'sayMPI << "Step " << n << " done. Current time: " << t\n'
        '                       << "  T=" << std::setprecision(6) << model.currentT() << " GeV";'
    )
    if stock in text:
        text = text.replace(stock, patched)
    elif broken_flush in text:
        text = text.replace(broken_flush, patched)
    elif old_t in text:
        text = text.replace(old_t, patched)
    elif old_t_no_nl in text and "std::cout.flush()" not in text:
        text = text.replace(old_t_no_nl, patched)
    else:
        raise RuntimeError("measurer.h verbose-output anchor not found")
    with open(measurer, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(measurer, REPO)} [verbose T + cout.flush]")


def _register_hdf5_param_string_limit():
    """CosmoLattice backup stores each parser key=value as a 256-char HDF5 string.

    Our outputfile path is longer than that, so the first tBackupFreq dump aborts
    with StringIsTooLong. Bump the fixed width.
    """
    path = os.path.join(
        CL, "src", "include", "TempLat", "lattice", "IO", "HDF5", "helpers", "hdf5type.h"
    )
    with open(path, "r") as f:
        text = f.read()
    old = "static constexpr int FixedSizeStringLength = 256;"
    new = "static constexpr int FixedSizeStringLength = 1024;"
    if new in text:
        return
    if old not in text:
        raise RuntimeError("hdf5type.h FixedSizeStringLength anchor not found")
    with open(path, "w") as f:
        f.write(text.replace(old, new))
    print(f"  patched {os.path.relpath(path, REPO)} [HDF5 param strings 256->1024]")


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


def _on_pbs():
    return bool(os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"))


def _mpirun_candidates():
    """Ordered mpirun paths. On PBS use PATH (module mpirun); on macOS prefer MPICH."""
    cands = []
    which = shutil.which("mpirun")
    if _on_pbs():
        if which:
            cands.append(which)
        return cands
    mpich_bin = os.path.join(MPICH_HOMEBREW, "bin", "mpirun")
    if os.path.isfile(mpich_bin) and os.access(mpich_bin, os.X_OK):
        cands.append(mpich_bin)
    if which and which not in cands:
        cands.append(which)
    return cands


def _resolve_mpirun():
    """Return an mpirun that can spawn at least one rank, or None."""
    cands = _mpirun_candidates()
    if _on_pbs() and cands:
        return cands[0]
    for mpirun in cands:
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
        return _stdbuf_prefix() + [binary, in_arg]
    mpirun = _resolve_mpirun()
    if mpirun is None:
        sys.exit(
            "ERROR: mpirun failed a launch probe.\n"
            "  Homebrew OpenMPI 5 (PRRTE) is broken on some Macs — use MPICH instead:\n"
            "    brew unlink open-mpi && brew link mpich\n"
            "  Then rebuild: python simulation/run_cosmolattice.py --build --mpi --dry_run\n"
            "  For single-rank testing use: --mpi --np 1"
        )
    return [mpirun, "-np", str(mpi_np)] + _stdbuf_prefix() + [binary, in_arg]


def _stdbuf_prefix():
    """Line-buffer CosmoLattice stdout (PBS/mpirun is not a TTY)."""
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        return [stdbuf, "-oL", "-eL"]
    return []


def _run_logged(cmd, log_paths):
    """Run cmd, tee stdout/stderr to log files, line-buffered."""
    logs = []
    try:
        for p in log_paths:
            if not p:
                continue
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            logs.append(open(p, "a", buffering=1))
        header = (
            "\n===== " + time.strftime("%Y-%m-%d %H:%M:%S")
            + "  " + " ".join(cmd) + " =====\n"
        )
        sys.stdout.write(header)
        sys.stdout.flush()
        for f in logs:
            f.write(header)
            f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            cwd=REPO,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                text = chunk.decode("latin-1", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            for f in logs:
                f.write(text)
                f.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        for f in logs:
            f.close()


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


def _hdf5_prefix(mpi=False):
    env = os.environ.get("HDF5_ROOT") or os.environ.get("MYHDF5_PATH")
    if env:
        inc = os.path.join(env, "include", "hdf5.h")
        if os.path.isfile(inc):
            return env
    if mpi:
        p = os.path.join(REPO, "external", "hdf5_mpich")
        if os.path.isfile(os.path.join(p, "include", "hdf5.h")):
            return p
    return ""


def _binary_built_with_hdf5(mpi=False):
    """True only if the configured CosmoLattice build has -DHDF5=ON.

    Having hdf5 libs on disk (_hdf5_prefix) is not enough: a -DHDF5=OFF binary
    still throws PureMPISaverNotImplemented on energy_snapshot / tBackupFreq.
    """
    cache = os.path.join(CL, build_dirname(mpi), "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if "HDF5:BOOL=ON" in txt or "HDF5:UNINITIALIZED=ON" in txt:
        return True
    if "HDF5:BOOL=OFF" in txt or "HDF5:UNINITIALIZED=OFF" in txt:
        return False
    return "HDF5" in txt and "BOOL=ON" in txt


def build(mpi=False, require_hdf5=None):
    hdf5 = _hdf5_prefix(mpi)
    if require_hdf5 is None:
        require_hdf5 = mpi  # default: MPI production expects HDF5 field dumps
    if require_hdf5 and not hdf5:
        sys.exit(
            "ERROR: MPI HDF5 field dumps need parallel HDF5.\n"
            "  Set HDF5_ROOT / MYHDF5_PATH, or use --snapshot_format raw "
            "(classical .raw; no HDF5 required for fields).\n"
            "  Or put hdf5 under external/hdf5_mpich."
        )
    build_dir = os.path.join(CL, build_dirname(mpi))
    # Wipe a Mac / other-host CMake cache (Nurion: CMAKE_CACHEFILE_DIR still
    # points at /Users/... if build_mpi/ was rsynced from the laptop).
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8", errors="ignore") as f:
            cache_txt = f.read()
        here = os.path.realpath(CL)
        foreign = (
            "/Users/" in cache_txt
            or "/home/" in cache_txt
            or here not in cache_txt
        )
        wants_mpich = os.path.isdir(FFTW_MPICH_PREFIX) and os.path.isdir(MPICH_HOMEBREW)
        linked_openmpi = "open-mpi" in cache_txt or "OpenMPI" in cache_txt
        hdf5_off = "HDF5:BOOL=OFF" in cache_txt or "HDF5:UNINITIALIZED=OFF" in cache_txt
        if foreign or (wants_mpich and linked_openmpi) or (hdf5 and hdf5_off):
            print("Stale CMake cache (other host or MPI mix); clearing "
                  + os.path.relpath(build_dir, REPO) + " ...")
            shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    mpi_flag = "ON" if mpi else "OFF"
    print(f"Configuring + building CosmoLattice (MODEL=thermal_inflation, MPI={mpi_flag})...")

    cmake_cmd = [
        "cmake",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DMPI={mpi_flag}",
        "-DMODEL=thermal_inflation",
        "-DCMAKE_CXX_STANDARD=14",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
    ]
    env = os.environ.copy()
    if env.get("CC"):
        cmake_cmd.append("-DCMAKE_C_COMPILER=" + env["CC"])
    if env.get("CXX"):
        cmake_cmd.append("-DCMAKE_CXX_COMPILER=" + env["CXX"])
    prefix_parts = []
    if hdf5:
        cmake_cmd.append("-DHDF5=ON")
        cmake_cmd.append("-DMYHDF5_PATH=" + hdf5)
        prefix_parts.append(hdf5)
        print("Using HDF5 at " + hdf5)
    else:
        cmake_cmd.append("-DHDF5=OFF")
        print("WARNING: no HDF5_ROOT / hdf5_mpich; 3D field dumps need HDF5.")
    if mpi:
        env_fftw = (
            env.get("MYFFTW3_PATH") or env.get("FFTW_DIR") or env.get("FFTW_ROOT") or ""
        )
        if env_fftw and os.path.isdir(env_fftw):
            cmake_cmd.append(f"-DMYFFTW3_PATH={env_fftw}")
            prefix_parts.append(env_fftw)
            print(f"Using FFTW from env at {env_fftw}")
        elif sys.platform == "darwin" and os.path.isdir(FFTW_MPICH_PREFIX):
            cmake_cmd.append(f"-DMYFFTW3_PATH={FFTW_MPICH_PREFIX}")
            prefix_parts.append(FFTW_MPICH_PREFIX)
            print(f"Using MPICH FFTW at {FFTW_MPICH_PREFIX}")
        elif sys.platform == "darwin":
            print(
                "WARNING: external/fftw_mpich not found. Homebrew FFTW is OpenMPI-ABI;\n"
                "  if you use MPICH, rebuild FFTW into external/fftw_mpich first."
            )
        if sys.platform == "darwin" and os.path.isdir(MPICH_HOMEBREW):
            prefix_parts.append(MPICH_HOMEBREW)
            mpich_bin = os.path.join(MPICH_HOMEBREW, "bin")
            env["PATH"] = mpich_bin + os.pathsep + env.get("PATH", "")
            env.setdefault("CC", os.path.join(mpich_bin, "mpicc"))
            env.setdefault("CXX", os.path.join(mpich_bin, "mpicxx"))
        if prefix_parts:
            cmake_cmd.append("-DCMAKE_PREFIX_PATH=" + os.pathsep.join(prefix_parts))
    cmake_cmd.append("..")

    subprocess.check_call(cmake_cmd, cwd=build_dir, env=env)
    subprocess.check_call(["make", "cosmolattice", "-j"], cwd=build_dir, env=env)
    print(f"Build complete: {os.path.relpath(binary_path(mpi), REPO)}")


# ---------------------------------------------------------------------------
# Generate run .in and execute
# ---------------------------------------------------------------------------
def _cl_rel(path):
    """Path relative to the repo (CosmoLattice cwd). Backup HDF5 stores key=value
    as a 256-char string; absolute /scratch/.../long-dirname overflows."""
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(".."):
        return os.path.abspath(path)
    return rel.replace("\\", "/")


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
    if args.backup_steps is not None:
        t_backup = args.backup_steps * dt_tilde
    elif args.tBackupFreq is not None:
        t_backup = args.tBackupFreq
    else:
        t_backup = -1.0
    # CosmoLattice checkpoints / energy_snapshot use HDF5 FileIO. A binary built
    # with -DHDF5=OFF aborts (PureMPISaverNotImplemented) even if libs exist.
    snap_fmt = getattr(args, "snapshot_format", "hdf5")
    hdf5_ok = _binary_built_with_hdf5(bool(getattr(args, "mpi", False)))
    if t_backup > 0 and not hdf5_ok:
        print(
            "WARNING: disabling tBackupFreq — thermal_inflation was built with "
            "HDF5=OFF (checkpoints need -DHDF5=ON). Classical --snapshot_format "
            "raw does not need backups."
        )
        t_backup = -1.0
    if snap_fmt == "raw" and t_backup > 0:
        # Keep classical runs free of CosmoLattice HDF5 I/O entirely.
        print("WARNING: disabling tBackupFreq with --snapshot_format raw "
              "(CosmoLattice backups are HDF5-only).")
        t_backup = -1.0

    out_rel = _cl_rel(out_dir)
    if not out_rel.endswith("/"):
        out_rel += "/"

    lines = [
        "#Output",
        f"outputfile = {out_rel}",
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
        f"tOutputRareFreq = {args.tOutputRareFreq if args.tOutputRareFreq is not None else 1000.0 * dt_tilde:g}",
        f"tOutputVerb = {args.tOutputFreq:g}",
        f"tMax = {args.tMax:g}",
        f"tBackupFreq = {t_backup:g}",
        "",
        "#Field snapshots",
        f"save_snapshots = {1 if save_snaps else 0}",
        f"snapshot_steps = {coarse_steps}",
        f"snapshot_format = {getattr(args, 'snapshot_format', 'hdf5')}",
    ]
    # CosmoLattice EnergySnapshotsMeasurer needs a -DHDF5=ON *binary*.
    use_hdf5_snaps = (
        save_snaps
        and snap_fmt == "hdf5"
        and hdf5_ok
    )
    if use_hdf5_snaps:
        lines.append("energy_snapshot = E_S_K E_S_G E_V")
    elif save_snaps and snap_fmt == "hdf5" and not hdf5_ok:
        print(
            "WARNING: snapshot_format=hdf5 but binary has HDF5=OFF; "
            "field dumps will no-op. Use --snapshot_format raw or rebuild "
            "with HDF5."
        )
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
        f"eta_follows_T = {1 if getattr(args, 'eta_follows_T', False) else 0}",
        f"dx_phys = {args.dx_phys:g}",
        f"dt_phys = {args.dt_phys:g}",
        f"include_cw = {args.include_cw}",
        f"thermal_noise = {args.thermal_noise}",
        f"langevin_off_after_nucleation = {1 if args.langevin_off_after_nucleation else 0}",
        f"langevin_off_f_switch = {args.langevin_off_f_switch:g}",
        f"langevin_off_phi_esc = "
        f"{(args.langevin_off_phi_esc if args.langevin_off_phi_esc is not None else args.expansion_phi_esc):g}",
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
        "eta_follows_T": bool(getattr(args, "eta_follows_T", False)),
        "thermal_noise": args.thermal_noise,
        "langevin_off_after_nucleation": bool(args.langevin_off_after_nucleation),
        "langevin_off_f_switch": args.langevin_off_f_switch,
        "langevin_off_phi_esc": (
            args.langevin_off_phi_esc
            if args.langevin_off_phi_esc is not None
            else args.expansion_phi_esc
        ),
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
        "stochastic_scheme": getattr(args, "stochastic_scheme", "numba"),
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
    """Match latticeSimeRescale_numba.py save_path naming, with a _CL suffix.

    CosmoLattice backup stores outputfile=... as a short HDF5 string; we pass a
    *relative* outputfile (see _cl_rel), so this name can stay descriptive.
    """
    N = args.Nx
    steps = args.steps if args.steps is not None else 100_000
    hubble_tag = "_nohubble" if args.no_hubble else "_hubble"
    if args.expansion_mode == "staged":
        staged_parts = ["_staged"]
        if args.expansion_T_switch > 0:
            staged_parts.append(f"_Tsw_{args.expansion_T_switch:g}")
        else:
            staged_parts.append(f"_fsw_{args.expansion_f_switch:g}")
            staged_parts.append(f"_phiesc_{args.expansion_phi_esc:g}")
        if args.T_rh > 0:
            staged_parts.append(f"_Trh_{args.T_rh:g}")
        staged_tag = "".join(staged_parts)
    else:
        staged_tag = ""
    eta = args.eta_phys if args.eta_phys is not None else args.T0
    eta_tag = f"_eta_{eta:g}"
    if getattr(args, "eta_follows_T", False):
        eta_tag += "_etaT"
    if args.langevin_off_after_nucleation:
        eta_tag += f"_langoff_f{args.langevin_off_f_switch:g}"
    nb = 0 if args.potential_type == "fermion_only" else args.nb
    coupling_tag = (
        f"_gb_{args.boson_coupling:g}_gf_{args.fermion_coupling:g}"
        f"_nb_{nb:g}_nf_{args.nf:g}"
    )
    integrator_tag = f"_{args.evolver}"
    scheme = getattr(args, "stochastic_scheme", "numba")
    if scheme != "numba":
        integrator_tag += f"_{scheme}"
    pot_type_tag = f"_{args.potential_type}" if args.potential_type != "V_p" else ""
    field_tag = "_complex" if args.n_scalars >= 2 else ""
    zn_tag = f"_ZN{args.zn_order}" if args.n_scalars >= 2 and args.zn_order > 0 else ""
    # Tag γ in the run dirname when it is not the historical set8 value, so
    # different γ never collide even if forced into the same param_set folder.
    gamma = float(getattr(args, "gamma", SET8_GAMMA))
    if gammas_close(gamma, SET8_GAMMA):
        gamma_tag = ""
    else:
        gamma_tag = f"_g_{format_gamma_tag(gamma)}"
    return (
        f"{N}x{N}x{N}_T0_{int(args.T0)}{field_tag}{zn_tag}{gamma_tag}"
        f"_dx_{args.dx_phys:g}_dtphys_{args.dt_phys:g}"
        f"_interval_{steps}_3D{hubble_tag}{staged_tag}{eta_tag}{coupling_tag}"
        f"{integrator_tag}{pot_type_tag}_CL"
    )


def _apply_param_set(args):
    """Resolve data/lattice/<set>/ from --gamma / --param_set / --force_param_set."""
    requested = args.param_set
    if getattr(args, "force_param_set", False):
        if requested in (None, "", "auto"):
            args.param_set = set_name_for_gamma(args.gamma)
        else:
            args.param_set = requested
    elif resolve_param_set is not None:
        args.param_set = resolve_param_set(
            args.gamma, requested, auto=(requested in (None, "", "auto"))
        )
    else:
        args.param_set = set_name_for_gamma(args.gamma) if requested in (None, "", "auto") else requested

    if v0_of_gamma is not None:
        print(
            f"param_set={args.param_set}  gamma={args.gamma:g}  "
            f"V0={v0_of_gamma(args.gamma):.4e} GeV^4  "
            f"→ data/lattice/{args.param_set}/"
        )
    else:
        print(f"param_set={args.param_set}  gamma={args.gamma:g}")
    return args.param_set


def main():
    args = parse_args()
    _apply_param_set(args)

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
        need_hdf5 = getattr(args, "snapshot_format", "hdf5") == "hdf5"
        build(mpi=args.mpi, require_hdf5=need_hdf5 if args.mpi else False)

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
    in_rel = _cl_rel(in_path)
    in_arg = f"input={in_rel}"
    if args.mpi:
        cmd = _mpi_launch_cmd(binary, in_arg, args.mpi_np)
    else:
        cmd = _stdbuf_prefix() + [binary, in_arg]
    print("Run command:\n  " + " ".join(cmd))

    run_log = os.path.join(out_dir, "run.log")
    live_log = os.environ.get("TIPT_LIVE_LOG", "").strip()
    pointer = os.path.join(REPO, "kisti_log", "LATEST_RUN.txt")
    os.makedirs(os.path.dirname(pointer), exist_ok=True)
    with open(pointer, "w") as f:
        f.write("run_dir=" + out_dir + "\n")
        f.write("run_log=" + run_log + "\n")
        if live_log:
            f.write("live_log=" + live_log + "\n")
        f.write("watch:\n")
        f.write("  tail -f " + (live_log or run_log) + "\n")
        f.write("  tail -f " + os.path.join(out_dir, "average_energies.txt") + "\n")
    print("Live log: " + (live_log or run_log))
    print("Run dir:  " + out_dir)
    print("Pointer:  " + pointer)

    if args.dry_run:
        print("(dry run; not executing)")
        return
    if not os.path.exists(binary):
        hint = "--install --build --mpi" if args.mpi else "--install --build"
        print(f"ERROR: binary not found: {binary}\n  Run with {hint} first.")
        sys.exit(1)
    _run_logged(cmd, [run_log])

    raw_dir = os.path.join(out_dir, "field_states")
    has_raw = os.path.isdir(raw_dir) and any(
        n.endswith(".raw") for n in os.listdir(raw_dir)
    )
    if has_raw and not getattr(args, "no_export", False):
        print("Exporting leftover .raw snapshots to NPZ...")
        export_snapshots(out_dir, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
