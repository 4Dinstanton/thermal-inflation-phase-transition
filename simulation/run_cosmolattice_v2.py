#!/usr/bin/env python3
"""Driver for the CosmoLattice **v2.0** thermal-inflation runs.

This is the v2 counterpart of ``simulation/run_cosmolattice.py``. The v1 script
and the v1 submodule (``external/cosmolattice``, pinned to v1.1.2-30-ga5654d6)
are left completely untouched, so existing runs remain reproducible; a snapshot
of the v1 tree also lives under ``backups/``.

It reuses the v1 argument parser and output-directory naming so run directories
stay comparable, and overrides everything that differs in v2:

* code tree           ``external/cosmolattice_v2`` (tag ``v2.0.0``)
* extension headers   ``cosmolattice_ext_v2/``
* model / binary      ``thermal_inflation_v2``
* evolver name        ``stochastic`` (v1 called it ``stochasticrk``)
* compiler            needs alias-template CTAD (C++20); Apple clang 15 cannot
                      build TempLat, so we default to Homebrew g++-16
* snapshots           custom ``field_states/*.raw`` writer (same as v1: ``--steps``,
                      ``--phi_threshold``, ``--steps_dense``) + optional upstream HDF5
* build system        TempLat + Kokkos are fetched by CMake; MPI/HDF5 are
                      cmake options rather than a hand-rolled FFTW build

Typical use::

    python simulation/run_cosmolattice_v2.py --install --build \\
        --Nx 64 --T0 1230 --tMax 400 --param_set set8 \\
        --n_scalars 2 --with_gws --expansion_mode staged
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_cosmolattice as v1  # noqa: E402  (v1 module: arg parsing + naming)

REPO = v1.REPO
CL2 = os.path.join(REPO, "external", "cosmolattice_v2")
EXT2 = os.path.join(REPO, "cosmolattice_ext_v2")
TABLE = v1.TABLE

MODEL_NAME = "thermal_inflation_v2"
EVOLVER_NAME = "stochastic"
BUILD_DIR = "build_ti"
BUILD_DIR_MPI = "build_ti_mpi"

MARK_OPEN = "// >>> thermal-inflation"
MARK_CLOSE = "// <<< thermal-inflation"

# TempLat v1.0.0 uses CTAD on an alias template (C++20, P1814R0). Apple clang 15
# rejects it; GCC >= 12 and clang >= 19 accept it.
CXX_CANDIDATES = [
    "/opt/homebrew/opt/gcc/bin/g++-16",
    "/opt/homebrew/opt/gcc@16/bin/g++-16",
    "/opt/homebrew/opt/llvm/bin/clang++",
    "g++-16",
    "g++-15",
    "g++-14",
]


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
def find_cxx(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("CL2_CXX")
    if env:
        return env
    for c in CXX_CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# ---------------------------------------------------------------------------
# Install: link extension headers, apply the three upstream registrations
# ---------------------------------------------------------------------------
def _symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            return
        os.remove(dst)
    os.symlink(src, dst)
    print(f"  linked {os.path.relpath(dst, REPO)}")


def _patch_block(path, anchor, insert, tag):
    """Insert `insert` right after `anchor` unless the tag is already present."""
    with open(path) as f:
        text = f.read()
    if f"{MARK_OPEN} {tag}" in text or insert.strip() in text:
        return False
    if anchor not in text:
        sys.exit(f"ERROR: anchor for '{tag}' not found in {path}")
    text = text.replace(anchor, anchor + insert, 1)
    with open(path, "w") as f:
        f.write(text)
    print(f"  patched {os.path.relpath(path, REPO)} [{tag}]")
    return True


def install():
    if not os.path.isdir(CL2):
        sys.exit(
            f"ERROR: {os.path.relpath(CL2, REPO)} missing.\n"
            "  git clone --branch v2.0.0 https://github.com/cosmolattice/cosmolattice.git "
            "external/cosmolattice_v2"
        )
    print("Installing thermal-inflation extension into CosmoLattice v2.0 ...")

    models = os.path.join(CL2, "models")
    evolvers = os.path.join(CL2, "include", "CosmoInterface", "evolvers")
    _symlink(os.path.join(EXT2, "models", "thermal_inflation.h"),
             os.path.join(models, f"{MODEL_NAME}.h"))
    for h in ("thermal_force.h", "thermal_tables.hpp"):
        _symlink(os.path.join(EXT2, "models", h), os.path.join(models, h))
    # Snapshot writer lives under measurements/ but is #include'd from the model.
    _symlink(os.path.join(EXT2, "measurements", "field_snapshot.hpp"),
             os.path.join(models, "field_snapshot.hpp"))
    _symlink(os.path.join(EXT2, "evolvers", "stochasticrk.h"),
             os.path.join(evolvers, "stochasticrk.h"))

    _register_evolver(evolvers)
    _register_ic_hooks(os.path.join(CL2, "source", "cosmolattice.cpp"))
    _register_snapshot_hook(os.path.join(CL2, "source", "cosmolattice.cpp"))
    _relax_defects_guards(os.path.join(CL2, "include", "CosmoInterface", "initializers"))
    print("Install complete.")


def _register_evolver(evolvers_dir):
    etype = os.path.join(evolvers_dir, "evolvertype.h")
    evol = os.path.join(evolvers_dir, "evolver.h")

    _patch_block(
        etype,
        "PV2, PV4, PV6, PV8, PV10, PV6_2",
        f",\n                     STOCHASTIC /* {MARK_OPEN} evolver-enum {MARK_CLOSE} */",
        "evolver-enum",
    )
    _patch_block(
        etype,
        '      eType = PV6_2; // alternative scheme for PV6 (see documentation)\n',
        '    else if (tmp == "stochastic" || tmp == "STOCHASTIC" || tmp == "stochasticrk")\n'
        f'      eType = STOCHASTIC; {MARK_OPEN} evolver-parse {MARK_CLOSE}\n',
        "evolver-parse",
    )
    _patch_block(
        etype,
        '    else if (eType == PV6_2)\n      return "PV6_2";\n',
        "    else if (eType == STOCHASTIC)\n"
        f'      return "STOCHASTIC"; {MARK_OPEN} evolver-string {MARK_CLOSE}\n',
        "evolver-string",
    )

    _patch_block(
        evol,
        '#include "CosmoInterface/evolvers/rk2nstorage.h"\n',
        '#include "CosmoInterface/evolvers/stochasticrk.h" '
        f"{MARK_OPEN} include {MARK_CLOSE}\n",
        "include",
    )
    _patch_block(
        evol,
        "  template <typename M> struct CheckAxionU1<M, decltype((void)M::IsAxionU1Coupled, void())> {\n"
        "    static constexpr bool value = M::IsAxionU1Coupled;\n  };\n",
        f"\n  {MARK_OPEN} concept\n"
        "  template <typename M>\n"
        "  concept IsLangevinModel = requires(M &m) {\n"
        "    m.stochasticScheme;\n"
        "    m.maybeDisableLangevin();\n"
        "  };\n"
        f"  {MARK_CLOSE}\n",
        "concept",
    )
    _patch_block(
        evol,
        "      if (rk2n != nullptr) {\n        rk2n->setDelta(extraFlds);\n      }\n",
        f"\n      {MARK_OPEN} construct\n"
        "      if constexpr (IsLangevinModel<Model>) {\n"
        "        if (type == STOCHASTIC) srk = std::make_shared<StochasticLangevin<Model>>(model, rPar);\n"
        "      }\n"
        "      if (srk != nullptr) return;\n"
        f"      {MARK_CLOSE}\n",
        "construct",
    )
    _patch_block(
        evol,
        "    inline void evolve(Model &model, T tMinust0) const\n    {\n",
        f"      {MARK_OPEN} evolve-dispatch\n"
        "      if constexpr (IsLangevinModel<Model>) {\n"
        "        if (srk != nullptr) {\n"
        "          srk->evolve(model, tMinust0);\n"
        "          return;\n        }\n      }\n"
        f"      {MARK_CLOSE}\n",
        "evolve-dispatch",
    )
    _patch_block(
        evol,
        "    inline void sync(Model &model, T tMinust0) const\n    {\n",
        f"      {MARK_OPEN} sync-dispatch\n"
        "      if constexpr (IsLangevinModel<Model>) {\n"
        "        if (srk != nullptr) {\n"
        "          srk->sync(model, tMinust0);\n"
        "          return;\n        }\n      }\n"
        f"      {MARK_CLOSE}\n",
        "sync-dispatch",
    )
    _patch_block(
        evol,
        "    std::shared_ptr<RK2NStorage<Model>> rk2n;\n",
        f"    std::shared_ptr<StochasticLangevin<Model>> srk; {MARK_OPEN} member {MARK_CLOSE}\n",
        "member",
    )


def _relax_defects_guards(init_dir):
    """Let the thermal-inflation model set ``DefectsModel = true``.

    Upstream reserves the defects module for fixed-background runs seeded with a
    scaling network, because (extra)fattening is only tested there. We want it
    solely for the ``winfindLengthStrings`` observable: our strings condense out
    of the thermal bath, the model prescribes H(T) itself, and fattening stays
    off. Both guards are therefore skipped for models exposing
    ``maybeDisableLangevin()``, i.e. only ours.
    """
    scalefactor = os.path.join(init_dir, "scalefactorinitializer.h")
    with open(scalefactor) as f:
        text = f.read()
    old = (
        "      if (!rPar.fixedBackground && Model::DefectsModel)\n"
        '        throw(RunParametersInconsistent("Running a defects model with self-consistent expansion is not tested, and "\n'
        '                                        "features such as (extra)fattening may not work correctly. If you really want "\n'
        '                                        "to run this option, comment out this exception in scalefactorinitializar.h"));\n'
    )
    if f"{MARK_OPEN} defects-expansion" not in text:
        if old not in text:
            sys.exit(f"ERROR: anchor for 'defects-expansion' not found in {scalefactor}")
        new = (
            f"      {MARK_OPEN} defects-expansion\n"
            "      if constexpr (!requires(Model & m) { m.maybeDisableLangevin(); }) {\n"
            "        if (!rPar.fixedBackground && Model::DefectsModel)\n"
            '          throw(RunParametersInconsistent("Running a defects model with self-consistent expansion is not "\n'
            '                                          "tested, and features such as (extra)fattening may not work "\n'
            '                                          "correctly."));\n'
            "      }\n"
            f"      {MARK_CLOSE}\n"
        )
        with open(scalefactor, "w") as f:
            f.write(text.replace(old, new, 1))
        print(f"  patched {os.path.relpath(scalefactor, REPO)} [defects-expansion]")

    singlet = os.path.join(init_dir, "scalarsingletinitializer.h")
    with open(singlet) as f:
        text = f.read()
    old = (
        "      if (Model::DefectsModel && (flagSIC != InitialConditionsType::S::DefectsNetwork &&\n"
        "                                  flagSIC != InitialConditionsType::S::DefectsWhiteNoise))\n"
    )
    if f"{MARK_OPEN} defects-ic" not in text:
        if old not in text:
            sys.exit(f"ERROR: anchor for 'defects-ic' not found in {singlet}")
        new = (
            f"      {MARK_OPEN} defects-ic\n"
            "      constexpr bool skipDefectsICCheck = requires(Model &m) { m.maybeDisableLangevin(); };\n"
            f"      {MARK_CLOSE}\n"
            "      if (!skipDefectsICCheck && Model::DefectsModel &&\n"
            "          (flagSIC != InitialConditionsType::S::DefectsNetwork &&\n"
            "           flagSIC != InitialConditionsType::S::DefectsWhiteNoise))\n"
        )
        with open(singlet, "w") as f:
            f.write(text.replace(old, new, 1))
        print(f"  patched {os.path.relpath(singlet, REPO)} [defects-ic]")


def _register_ic_hooks(main_cpp):
    _patch_block(
        main_cpp,
        "    initializer.initialize(model, runParams, measurer.getFilesManager(), extraFlds);\n",
        f"\n    {MARK_OPEN} initial-conditions\n"
        "    if constexpr (requires { model.applyNumbaInitialConditions(); }) {\n"
        "      model.applyNumbaInitialConditions();\n"
        "      model.applyUniformPhi();\n"
        "      model.applyBubbleSeed();\n"
        "      model.freezeInactiveScalars();\n"
        "    }\n"
        f"    {MARK_CLOSE}\n",
        "initial-conditions",
    )


def _register_snapshot_hook(main_cpp):
    """Call saveFieldSnapshotIfDue every step so dense switching can fire."""
    with open(main_cpp) as f:
        text = f.read()
    if "saveFieldSnapshotIfDue" in text:
        return
    old = (
        "    if (measurer.areWeMeasuring(i))\n"
        "    // We proceed to measure\n"
        "    {\n"
        "      evolver.sync(model, t - runParams.t0);\n"
        "      // Some evolvers like staggered leapfrog have fields and momenta which\n"
        "      // do not live at the same timesteps. Before measuring, we synchronize them.\n"
        "      measurer.measure(i, t, model);\n"
        "      // Note that measurer.measure advances automatically conjugate momenta by half step in case\n"
        "      // the evolver (e.g. leapfrog) required them to have been synchronised previously for\n"
        "      // a measurement.\n"
        "    }\n"
    )
    new = (
        "    if (measurer.areWeMeasuring(i))\n"
        "    // We proceed to measure\n"
        "    {\n"
        "      evolver.sync(model, t - runParams.t0);\n"
        "      // Some evolvers like staggered leapfrog have fields and momenta which\n"
        "      // do not live at the same timesteps. Before measuring, we synchronize them.\n"
        "    }\n"
        "\n"
        f"    // {MARK_OPEN} snapshot\n"
        "    if constexpr (requires { model.saveFieldSnapshotIfDue(0, 0.0); }) {\n"
        "      model.saveFieldSnapshotIfDue(i, t);\n"
        "    }\n"
        f"    // {MARK_CLOSE}\n"
        "\n"
        "    if (measurer.areWeMeasuring(i))\n"
        "    {\n"
        "      measurer.measure(i, t, model);\n"
        "      // Note that measurer.measure advances automatically conjugate momenta by half step in case\n"
        "      // the evolver (e.g. leapfrog) required them to have been synchronised previously for\n"
        "      // a measurement.\n"
        "    }\n"
    )
    if old not in text:
        sys.exit(f"ERROR: snapshot-hook anchor not found in {main_cpp}")
    with open(main_cpp, "w") as f:
        f.write(text.replace(old, new, 1))
    print(f"  patched {os.path.relpath(main_cpp, REPO)} [snapshot]")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_dirname(mpi=False):
    return BUILD_DIR_MPI if mpi else BUILD_DIR


def binary_path(mpi=False):
    return os.path.join(CL2, build_dirname(mpi), MODEL_NAME)


def build(mpi=False, hdf5=False, cxx=None, jobs=None):
    compiler = find_cxx(cxx)
    if compiler is None:
        sys.exit(
            "ERROR: no suitable C++20 compiler found.\n"
            "  TempLat needs alias-template CTAD; Apple clang 15 cannot compile it.\n"
            "  brew install gcc   (then re-run), or set CL2_CXX=/path/to/g++"
        )
    build_dir = os.path.join(CL2, build_dirname(mpi))
    os.makedirs(build_dir, exist_ok=True)

    # Homebrew FFTW's libfftw3_mpi is OpenMPI-ABI. Launching with MPICH then
    # dies with: dyld: symbol not found '_ompi_mpi_char'. Same fix as v1:
    # point FFTW at external/fftw_mpich (built against MPICH).
    fftw_mpich = v1.FFTW_MPICH_PREFIX
    mpich = v1.MPICH_HOMEBREW
    env = os.environ.copy()
    cmake = [
        "cmake",
        f"-DMODEL={MODEL_NAME}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DMPI={'ON' if mpi else 'OFF'}",
        f"-DHDF5={'ON' if hdf5 else 'OFF'}",
        f"-DAUTOBUILD_HDF5={'ON' if hdf5 else 'OFF'}",
    ]

    if mpi:
        cache = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.isfile(cache):
            with open(cache) as f:
                cache_txt = f.read()
            # Stale cache pointing at Homebrew (OpenMPI) FFTW → wipe and reconfigure.
            if "/opt/homebrew/lib/libfftw3_mpi" in cache_txt or "open-mpi" in cache_txt:
                print("Stale OpenMPI/Homebrew-FFTW CMake cache detected; clearing "
                      f"{os.path.relpath(build_dir, REPO)}/ ...")
                shutil.rmtree(build_dir)
                os.makedirs(build_dir, exist_ok=True)

        if not os.path.isdir(fftw_mpich):
            sys.exit(
                f"ERROR: {os.path.relpath(fftw_mpich, REPO)} missing.\n"
                "  Homebrew FFTW is OpenMPI-ABI and cannot be mixed with MPICH.\n"
                "  Rebuild FFTW against MPICH into external/fftw_mpich (same as v1)."
            )
        if not os.path.isdir(mpich):
            sys.exit(
                f"ERROR: MPICH not found at {mpich}.\n"
                "  brew install mpich"
            )

        prefix = f"{fftw_mpich};{mpich}"
        # Keep g++-16 as the real compiler (mpicxx wraps Apple clang, which
        # cannot build TempLat). FindMPI still uses the MPICH wrappers.
        cmake += [
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DFFTW_DIR={fftw_mpich}",
            f"-DMPI_CXX_COMPILER={os.path.join(mpich, 'bin', 'mpicxx')}",
            f"-DMPI_C_COMPILER={os.path.join(mpich, 'bin', 'mpicc')}",
            f"-DMPIEXEC_EXECUTABLE={os.path.join(mpich, 'bin', 'mpiexec')}",
        ]
        env["PATH"] = os.path.join(mpich, "bin") + os.pathsep + env.get("PATH", "")
        env["FFTW_DIR"] = fftw_mpich
        env["MPICH_CXX"] = compiler  # if anything invokes mpicxx, wrap g++-16
        print(f"Using MPICH FFTW at {fftw_mpich}")
    else:
        cmake.append(f"-DCMAKE_CXX_COMPILER={compiler}")

    # TempLat needs system HDF5, or AUTOBUILD_HDF5=ON to FetchContent it.
    # Homebrew's hdf5-mpi is OpenMPI-only; this project uses MPICH, so prefer
    # the in-tree autobuild when snapshots are requested.
    cmake.append("..")
    print("Configuring:\n  " + " ".join(cmake))
    subprocess.check_call(cmake, cwd=build_dir, env=env)
    make = ["make", "cosmolattice", f"-j{jobs}" if jobs else "-j"]
    print("Building:\n  " + " ".join(make))
    subprocess.check_call(make, cwd=build_dir, env=env)
    print(f"Binary: {os.path.relpath(binary_path(mpi), REPO)}")
    if mpi:
        # Sanity: the MPI FFTW must be the MPICH build, not Homebrew's OpenMPI one.
        try:
            libs = subprocess.check_output(
                ["otool", "-L", binary_path(mpi)], text=True
            )
        except (OSError, subprocess.CalledProcessError):
            libs = ""
        if "libfftw3_mpi" in libs and "fftw_mpich" not in libs:
            sys.exit(
                "ERROR: MPI binary still links Homebrew/OpenMPI libfftw3_mpi.\n"
                f"  otool -L {binary_path(mpi)}\n"
                "  Wipe build_ti_mpi and rebuild with --build --mpi."
            )
        if "open-mpi" in libs:
            sys.exit(
                "ERROR: MPI binary links OpenMPI.\n"
                f"  otool -L {binary_path(mpi)}\n"
                "  Wipe build_ti_mpi and rebuild with --build --mpi."
            )


# ---------------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------------
def make_input(args, out_dir):
    import math

    N = args.Nx
    mu = args.mphi
    dx_tilde = mu * args.dx_phys
    dt_tilde = mu * args.dt_phys
    kIR = 2.0 * math.pi / (N * dx_tilde)
    eta = args.eta_phys if args.eta_phys is not None else args.T0
    expansion = "false" if args.no_hubble else "true"
    snaps = v1._snapshots_enabled(args)
    langevin_off_phi_esc = (
        args.langevin_off_phi_esc
        if args.langevin_off_phi_esc is not None
        else args.expansion_phi_esc
    )
    # winfindLengthStrings is a winding number of Complexify(fldS0, fldS1), so it
    # is only meaningful when both components are live.
    measure_defects = args.measure_defects and args.n_scalars >= 2
    if args.measure_defects and not measure_defects:
        print("NOTE: --measure_defects ignored; string windings need --n_scalars 2.")

    lines = [
        "#Output",
        f"outputfile = {out_dir}/",
        "print_headers = true",
        "overwriteFiles = true",
        "",
        "#Evolution",
        f"expansion = {expansion}",
        f"evolver = {EVOLVER_NAME}",
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
        "#Cosmic defects (in-simulation string length; needs n_scalars = 2)",
        f"measureDefectsStructure = {'true' if measure_defects else 'false'}",
        f"measureDefectsEnergies = {'true' if measure_defects else 'false'}",
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
        f"include_cw = {args.include_cw}",
        f"thermal_noise = {args.thermal_noise}",
        f"langevin_off_after_nucleation = {1 if args.langevin_off_after_nucleation else 0}",
        f"langevin_off_f_switch = {args.langevin_off_f_switch:g}",
        f"langevin_off_phi_esc = {langevin_off_phi_esc:g}",
        f"ic_numba = {0 if (args.cosmolattice_ic or args.bubble_seed_phi > 0 or args.uniform_phi > 0) else 1}",
        f"uniform_phi = {args.uniform_phi:g}",
        f"bubble_seed_phi = {args.bubble_seed_phi:g}",
        f"bubble_seed_bg = {args.bubble_seed_bg:g}",
        f"bubble_seed_radius = {args.bubble_seed_radius}",
        f"stochastic_scheme = {getattr(args, 'stochastic_scheme_v2', 'ou')}",
        f"thermal_table = {TABLE}",
        f"n_scalars = {args.n_scalars}",
        f"zn_order = {args.zn_order}",
        f"zn_strength = {args.zn_strength:g}",
        f"zn_turn_on_T = {args.zn_turn_on_T:g}",
        "",
        "#Post-PT expansion staging (ti -> md -> rd)",
        f"expansion_mode = {args.expansion_mode}",
        f"expansion_T_switch = {args.expansion_T_switch:g}",
        f"expansion_f_switch = {args.expansion_f_switch:g}",
        f"expansion_phi_esc = {args.expansion_phi_esc:g}",
        f"T_rh = {args.T_rh:g}",
        "",
    ]
    if snaps:
        coarse = v1._snapshot_steps(args)
        phi_thr = args.phi_threshold if args.phi_threshold is not None else -1.0
        lines += [
            "#Field snapshots (raw via field_snapshot.hpp; export with tools/export_cl_snapshots.py)",
            "save_snapshots = 1",
            f"snapshot_steps = {coarse}",
            f"phi_threshold = {phi_thr:g}",
        ]
        if args.steps_dense is not None:
            lines.append(f"snapshot_steps_dense = {args.steps_dense}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    """v1 parser plus the v2-only knobs (consumed first, then handed over)."""
    v2p = argparse.ArgumentParser(add_help=False)
    v2p.add_argument("--cxx", default=None, help="C++ compiler for the v2 build (default: g++-16)")
    v2p.add_argument("--hdf5", action="store_true",
                     help="build with HDF5 (optional; raw --steps snapshots do not need it)")
    v2p.add_argument("--jobs", type=int, default=None, help="make -j value")
    v2p.add_argument("--stochastic_scheme_v2", default="ou",
                     choices=["ou", "verlet", "numba"],
                     help="Langevin scheme: ou = exact FDT temperature; "
                          "verlet = explicit friction (v1 fdt amplitude); "
                          "numba = verlet + half-FDT noise (v1 default amplitude / T_c1 parity)")
    v2p.add_argument("--measure_defects", action="store_true",
                     help="write the in-simulation comoving string length "
                          "(average_defects.txt, column winfindLengthStrings); "
                          "requires --n_scalars 2")
    known, rest = v2p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = v1.parse_args()
    for k, v in vars(known).items():
        setattr(args, k, v)
    args.evolver = EVOLVER_NAME
    # So cl_run_params.json / any v1 helpers see the v2 scheme name.
    args.stochastic_scheme = getattr(args, "stochastic_scheme_v2", "ou")
    return args


def main():
    args = parse_args()

    if not os.path.exists(TABLE):
        print(f"NOTE: thermal table {os.path.relpath(TABLE, REPO)} not found.")
        print("      Run: python tools/export_thermal_splines.py")
        if not args.dry_run:
            sys.exit(1)

    if args.install:
        install()
    if args.build:
        build(mpi=args.mpi, hdf5=args.hdf5, cxx=args.cxx, jobs=args.jobs)

    out_root = os.path.join(REPO, "data", "lattice", args.param_set)
    scheme = getattr(args, "stochastic_scheme_v2", "ou")
    out_dir = os.path.join(out_root, v1.output_dirname(args) + f"_v2_{scheme}")
    os.makedirs(out_dir, exist_ok=True)

    in_path = os.path.join(out_dir, "input.in")
    with open(in_path, "w") as f:
        f.write(make_input(args, out_dir))
    v1.write_run_params(args, out_dir)
    print(f"Wrote input file: {os.path.relpath(in_path, REPO)}")

    binary = binary_path(mpi=args.mpi)
    cmd = [binary, f"input={in_path}"]
    if args.mpi and args.mpi_np and args.mpi_np > 1:
        cmd = v1._mpi_launch_cmd(binary, f"input={in_path}", args.mpi_np)
    print("Run command:\n  " + " ".join(cmd))

    if args.dry_run:
        print("(dry run; not executing)")
        return
    if not os.path.exists(binary):
        sys.exit(f"ERROR: binary not found: {binary}\n  Run with --install --build first.")
    subprocess.check_call(cmd)

    if v1._snapshots_enabled(args):
        print("Exporting field snapshots to numba NPZ format...")
        v1.export_snapshots(out_dir, keep_raw=getattr(args, "keep_raw", False))


if __name__ == "__main__":
    main()
