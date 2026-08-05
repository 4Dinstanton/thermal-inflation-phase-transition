#!/usr/bin/env python3
"""Check that the Langevin evolver samples the temperature it is asked for.

Runs a small non-expanding box at fixed T with the thermal potential, and
compares the measured <pi^2> against the equipartition value in CosmoLattice
program variables,

    <pi^2>_eq = T / (dx_com^3 fStar^2 omegaStar^2)          (a = 1)

which follows from <phidot^2> = T / (a^3 dx_com^3) and pi = a^3 phidot /
(fStar omegaStar). See cosmolattice_ext_v2/README.md section 1.

Expected outcome:

  * v2 `ou`      -> ratio 1.00 at every dt, until the conservative Verlet core
                    itself goes unstable (dt >= 0.3 for these parameters).
  * v2 `verlet`  -> ratio 1 / (1 - eta*dt/4), the explicit-friction bias.
  * v1 `fdt`     -> same bias as v2 `verlet` (cross-check of two independent
                    implementations).
  * v1 `numba`   -> half of that: the numba reference injects half the FDT
                    variance by construction.

Usage:
    python cosmolattice_ext_v2/tests/fdt_check.py              # v2 only
    python cosmolattice_ext_v2/tests/fdt_check.py --with-v1    # also v1
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2_BIN = os.path.join(REPO, "external", "cosmolattice_v2", "build_ti", "thermal_inflation_v2")
V1_BIN = os.path.join(REPO, "external", "cosmolattice", "build", "thermal_inflation")
TABLE = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")

# The box: hot enough that phi stays pinned near the origin, so the measured
# <pi^2> is the thermal plateau rather than a rolling field.
MPHI = 1000.0
GAMMA = 4.1667e-4
T0 = 7350.0
ETA = 7350.0
DX_PHYS = 1.0e-3
F_STAR = GAMMA * 2.4e18  # = mphi / sqrt(lambda)
PI2_EQ = T0 / (DX_PHYS**3 * F_STAR**2 * MPHI**2)

INPUT = """\
outputfile = {out}/
print_headers = true
overwriteFiles = true

expansion = false
evolver = {evolver}

N = 32
dt = {dt}
kIR = 0.19634954

tOutputFreq = 5
tOutputInfreq = 100
tMax = 10

kCutOff = 4
baseSeed = 7
initial_amplitudes = 0.0
initial_momenta = 0.0
ic_numba = 1

PS_type = 1
PS_version = 1
withGWs = false

potential_type = V_correct
mphi = {mphi}
gamma = {gamma}
boson_coupling = 1.09
boson_gauge_coupling = 1.09
fermion_coupling = 1.09
fermion_gauge_coupling = 1.09
boson_mass_squared = 1.0e6
nb = 20
nf = 20
g_star_pot = 100.0
g_star_hubble = 106.75

T0 = {T0}
eta_phys = {eta}
dx_phys = {dx}
include_cw = 1
thermal_noise = 1
stochastic_scheme = {scheme}
n_scalars = 1
thermal_table = {table}

expansion_mode = legacy
"""


def run(binary, evolver, scheme, dt, workdir):
    out = os.path.join(workdir, f"out_{evolver}_{scheme}_{dt}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    in_path = os.path.join(out, "input.in")
    with open(in_path, "w") as f:
        f.write(INPUT.format(out=out, evolver=evolver, scheme=scheme, dt=dt,
                             mphi=MPHI, gamma=GAMMA, T0=T0, eta=ETA,
                             dx=DX_PHYS, table=TABLE))
    res = subprocess.run([binary, f"input={in_path}"], cwd=workdir,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return None
    with open(os.path.join(out, "average_scalar_0.txt")) as f:
        last = f.readlines()[-1].split()
    return float(last[4])  # <pi^2>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-v1", action="store_true",
                    help="also run the v1 binary for comparison")
    ap.add_argument("--workdir", default="/tmp/ti_fdt_check")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    if not os.path.exists(V2_BIN):
        sys.exit(f"ERROR: v2 binary not found: {V2_BIN}\n"
                 "  python simulation/run_cosmolattice_v2.py --install --build")

    print(f"<pi^2>_eq = T / (dx^3 fStar^2 mphi^2) = {PI2_EQ:.4g}")
    print(f"{'binary':<6} {'scheme':<8} {'dt':<8} {'eta*dt':<8} "
          f"{'measured/eq':<13} {'expected':<10}")

    jobs = [("v2", V2_BIN, "stochastic", s, d)
            for s in ("ou", "verlet", "numba")
            for d in (0.05, 0.1)]
    if args.with_v1:
        if not os.path.exists(V1_BIN):
            sys.exit(f"ERROR: v1 binary not found: {V1_BIN}")
        jobs += [("v1", V1_BIN, "stochasticrk", s, 0.1) for s in ("numba", "fdt")]

    for tag, binary, evolver, scheme, dt in jobs:
        pi2 = run(binary, evolver, scheme, dt, args.workdir)
        if pi2 is None:
            print(f"{tag:<6} {scheme:<8} {dt:<8} {'':<8} FAILED")
            continue
        eta_dt = ETA / MPHI * dt
        bias = 1.0 / (1.0 - eta_dt / 4.0)
        expected = 1.0 if scheme == "ou" else (0.5 * bias if scheme == "numba" else bias)
        print(f"{tag:<6} {scheme:<8} {dt:<8} {eta_dt:<8.3f} "
              f"{pi2 / PI2_EQ:<13.4g} {expected:<10.4g}")


if __name__ == "__main__":
    main()
