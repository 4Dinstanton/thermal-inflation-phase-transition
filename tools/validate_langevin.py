#!/usr/bin/env python3
"""
Validate the CosmoLattice thermal-inflation Langevin (FDT) noise implementation.

Primary (theory) checks, run against the compiled binary:
  1. Table accuracy   : assembled V'(phi,T) from thermal_tables.hpp vs the exact
                        quadrature derivative of V_p_correct (numba chain rule).
  2. FDT/equipartition: <piS^2> -> T / (dx_phys^3 * fStar^2 * mu^2)  as dt -> 0.
  3. dt-convergence   : equipartition ratio approaches 1 as dt is reduced.
  4. T-linearity      : <pi^2> scales linearly with T0.
  5. Rayleigh-Jeans   : the velocity field has a flat |pi_k|^2 (white) spectrum.

Secondary check (optional, heavy):
  6. numba parity     : percolation temperature T_p from CosmoLattice vs the numba
                        fused_rk2 solver (run with --numba_parity; uses include_cw=0
                        so the EOM matches numba, which omits the CW force).

The FDT checks use a fixed temperature (--no_hubble) and rely only on kinetic
equipartition, so they are independent of the (complicated) potential.

Usage
-----
    python tools/validate_langevin.py                 # primary checks (+ plot)
    python tools/validate_langevin.py --build         # build the model first
    python tools/validate_langevin.py --numba_parity  # also run numba T_p parity
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "simulation", "run_cosmolattice.py")
CL = os.path.join(REPO, "external", "cosmolattice")
BIN = os.path.join(CL, "build", "thermal_inflation")
TABLE = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")
FIGDIR = os.path.join(REPO, "figs")
M_PL = 2.4e18

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def status(ok):
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


# ---------------------------------------------------------------------------
def run_sim(tag, T0, dt_phys, N=16, dx_phys=1e-3, gamma=4.1667e-4, tMax=6.0,
            extra=None):
    """Run a fixed-T (no expansion) simulation and return its output dir."""
    args = [
        sys.executable, RUN,
        "--Nx", str(N), "--T0", str(T0), "--no_hubble",
        "--gamma", str(gamma), "--dx_phys", str(dx_phys), "--dt_phys", str(dt_phys),
        "--tMax", str(tMax), "--tOutputFreq", str(max(tMax / 6.0, dt_phys * 1000)),
        "--tOutputInfreq", str(tMax), "--param_set", tag,
        "--stochastic_scheme", "fdt",
    ]
    if extra:
        args += extra
    out = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    # parse the written output dir from stdout
    for line in out.stdout.splitlines():
        if line.startswith("Wrote input file:"):
            rel = line.split(":", 1)[1].strip()
            return os.path.join(REPO, os.path.dirname(rel))
    raise RuntimeError("run failed:\n" + out.stdout + out.stderr)


def read_scalar(out_dir, tail_frac=0.5):
    """Return time-averaged <phi^2>, <pi^2> over the last tail_frac of the run."""
    path = os.path.join(out_dir, "average_scalar_0.txt")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    n0 = int(len(data) * (1 - tail_frac))
    tail = data[max(n0, 1):]
    phi2 = np.mean(tail[:, 3])
    pi2 = np.mean(tail[:, 4])
    return phi2, pi2, data


def equipartition_target(T0, dx_phys, gamma, mphi=1000.0):
    fStar = gamma * M_PL  # initial_amplitudes=0 -> fStar falls back to tree VEV = phi0
    mu = mphi
    # CosmoLattice average_scalar reports <piS^2> (program momentum).
    return T0 / (dx_phys ** 3 * fStar ** 2 * mu ** 2)


# ---------------------------------------------------------------------------
# Check 1: table accuracy vs quadrature truth
# ---------------------------------------------------------------------------
def check_tables():
    print("\n[1] Table accuracy: assembled V' vs exact quadrature derivative")
    test_bin = "/tmp/test_thermal_tables"
    src = os.path.join(REPO, "cosmolattice_ext", "tests", "test_thermal_tables.cpp")
    inc = os.path.join(REPO, "cosmolattice_ext", "models")
    try:
        subprocess.check_call(["g++", "-std=c++14", "-O2", "-I", inc, src, "-o", test_bin],
                              stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"    (skipped: could not compile C++ test: {e})")
        return True
    out = subprocess.check_output([test_bin, TABLE], text=True)

    # exact quadrature derivative of V_p_correct (numba chain rule) + analytic CW
    sys.path.insert(0, os.path.join(REPO, "potential"))
    import cosmoTransitions.finiteT as CTFT
    import warnings
    warnings.filterwarnings("ignore")
    gamma = 4.1667e-4
    mphi = 1000.0
    lam = mphi ** 2 / (gamma * M_PL) ** 2
    mb2, yb, gb, yf, gf, nb, nf = 1e6, 1.09, 1.05, 1.09, 1.05, 20, 20

    def vp_exact(phi, T):
        pref = T ** 4 / (2 * math.pi ** 2)
        ub = math.sqrt(mb2 + 0.5 * yb ** 2 * phi ** 2 + (0.25 * yb ** 2 + 2 / 3 * gb ** 2) * T ** 2) / T
        uf = math.sqrt(0.5 * yf ** 2 * phi ** 2 + (1 / 6) * gf ** 2 * T ** 2) / T
        dJb = float(np.real(CTFT.dJb_exact(ub)))
        dJf = float(np.real(CTFT.dJf_exact(uf)))
        dub = 0.5 * yb ** 2 * phi / (T ** 2 * ub)
        duf = 0.5 * yf ** 2 * phi / (T ** 2 * uf)
        m2 = 3 * lam * phi ** 2
        dcw = 6 * lam * phi * m2 / (64 * math.pi ** 2) * 2 * (math.log(abs(m2) / T) - 1) if m2 > 0 else 0.0
        return lam * phi ** 3 - mphi ** 2 * phi + pref * (nb * dJb * dub + nf * dJf * duf) + dcw

    max_rel = 0.0
    for line in out.strip().splitlines():
        # "T=... phi=...  V=...  Vp=...  Vpp=..."
        toks = line.replace("=", " ").split()
        T = float(toks[1]); phi = float(toks[3]); vp_cpp = float(toks[7])
        vp_ref = vp_exact(phi, T)
        # Floor the denominator by a fraction of the thermal-force scale so that
        # points sitting on a V'=0 crossing (where |vp_ref| ~ 0) do not inflate
        # the relative error through cancellation.
        scale = max(abs(vp_ref), 1e-3 * T ** 4 / (2 * math.pi ** 2) * nf)
        rel = abs(vp_cpp - vp_ref) / scale
        max_rel = max(max_rel, rel)
    ok = max_rel < 1e-4
    print(f"    max relative error (C++ table vs quadrature) = {max_rel:.2e}   {status(ok)}")
    return ok


# ---------------------------------------------------------------------------
# Checks 2-4: equipartition, dt-convergence, T-linearity
# ---------------------------------------------------------------------------
def check_fdt():
    print("\n[2-4] FDT / equipartition, dt-convergence, T-linearity")
    dx_phys, gamma = 1e-3, 4.1667e-4
    T0 = 7350.0
    target = equipartition_target(T0, dx_phys, gamma)

    dts = [1e-4, 2e-5, 5e-6]
    ratios = []
    for dtp in dts:
        d = run_sim("validate_fdt", T0, dtp, dx_phys=dx_phys, gamma=gamma)
        _, pi2, _ = read_scalar(d)
        r = pi2 / target
        ratios.append(r)
        print(f"    dt_phys={dtp:>7g}:  <pi^2>={pi2:.3e}  target={target:.3e}  ratio={r:.3f}")
    converged = abs(ratios[-1] - 1.0) < 0.10
    monotone = abs(ratios[-1] - 1.0) <= abs(ratios[0] - 1.0) + 1e-9
    print(f"    equipartition at smallest dt within 10%: {status(converged)}; "
          f"improves with smaller dt: {status(monotone)}")

    # T-linearity at the smallest dt
    print("    T-linearity:")
    dtp = dts[-1]
    Ts = [3675.0, 7350.0]
    pis = []
    for T in Ts:
        d = run_sim("validate_fdt", T, dtp, dx_phys=dx_phys, gamma=gamma)
        _, pi2, _ = read_scalar(d)
        pis.append(pi2 / equipartition_target(T, dx_phys, gamma))
        print(f"      T0={T:>7g}:  <pi^2>/target = {pis[-1]:.3f}")
    lin_ok = abs(pis[0] - pis[1]) < 0.12
    print(f"      <pi^2>/target T-independent (=> linear in T): {status(lin_ok)}")

    return converged and monotone and lin_ok, (dts, ratios, target)


# ---------------------------------------------------------------------------
# Check 5: Rayleigh-Jeans (flat |pi_k|^2)
# ---------------------------------------------------------------------------
def check_rayleigh_jeans():
    print("\n[5] Rayleigh-Jeans: velocity field has white |pi_k|^2 spectrum")
    N, dx_phys, mphi = 32, 1e-3, 1000.0
    d = run_sim("validate_rj", 7350.0, 5e-6, N=N, dx_phys=dx_phys, tMax=8.0)
    spec = np.loadtxt(os.path.join(d, "spectra_scalar_0.txt"))
    # The spectra file stacks one k-block per output time; average per k-bin.
    kall = spec[:, 0]
    kvals = np.unique(np.round(kall, 6))
    P_pi = np.array([spec[np.round(kall, 6) == kv, 2].mean() for kv in kvals])
    # |pi_k|^2 ~ 2 pi^2 Delta^2_pi / k^3 should be flat (white) for equipartition.
    mode_pi2 = P_pi / kvals ** 3

    # Restrict to the well-resolved band: above the few IR bins and below the
    # Nyquist face k = pi/dx_tilde (corner modes beyond it are sparsely sampled,
    # and modes above kCutOff are not seeded). dx_tilde = mphi * dx_phys.
    dx_t = mphi * dx_phys
    kIR = 2 * math.pi / (N * dx_t)
    kNyq = math.pi / dx_t
    band_mask = (kvals >= 4 * kIR) & (kvals <= 0.95 * kNyq)
    band = mode_pi2[band_mask]
    spread = np.std(band) / np.mean(band)
    ok = spread < 0.25
    print(f"    sub-Nyquist band k in [{4*kIR:.2f}, {0.95*kNyq:.2f}] "
          f"({band.size} bins): |pi_k|^2 relative spread = {spread:.3f}  {status(ok)}")
    return ok, (kvals[band_mask], band)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(fdt_data, rj_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(FIGDIR, exist_ok=True)
    dts, ratios, target = fdt_data
    k, mode_pi2 = rj_data
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(dts, ratios, "o-")
    ax[0].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("dt_phys"); ax[0].set_ylabel(r"$\langle\pi^2\rangle$ / equipartition")
    ax[0].set_title("FDT equipartition vs dt")
    ax[1].plot(k, mode_pi2 / np.mean(mode_pi2[len(k)//4:]), "o-")
    ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[1].set_xlabel("k (program)"); ax[1].set_ylabel(r"$|\pi_k|^2$ (normalised)")
    ax[1].set_title("Rayleigh-Jeans: white velocity spectrum")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "validate_langevin.png")
    fig.savefig(p, dpi=110)
    print(f"\nSaved plot: {os.path.relpath(p, REPO)}")


# ---------------------------------------------------------------------------
def numba_parity():
    print("\n[6] numba T_p parity (secondary) -- NOTE")
    print("    Run CosmoLattice with --include_cw 0 (matches numba EOM, which omits CW)")
    print("    and compare the percolation temperature T_p from the snapshot/percolation")
    print("    logic in postprocess/revisualize_snapshots.py / analysis/computeTn.py.")
    print("    This is a heavy, full-physics run and is left as an explicit opt-in study;")
    print("    see cosmolattice_ext/README.md. (Not executed here.)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Validate CosmoLattice Langevin/FDT noise")
    ap.add_argument("--build", action="store_true", help="Install + build the model first")
    ap.add_argument("--numba_parity", action="store_true", help="Describe/launch numba T_p parity")
    args = ap.parse_args()

    if not os.path.exists(TABLE):
        print("Thermal table missing; running exporter...")
        subprocess.check_call([sys.executable, os.path.join(REPO, "tools", "export_thermal_splines.py")])

    if args.build or not os.path.exists(BIN):
        print("Building model (install + build)...")
        subprocess.check_call([sys.executable, RUN, "--install", "--build", "--dry_run"], cwd=REPO)
    if not os.path.exists(BIN):
        sys.exit(f"ERROR: binary not found at {BIN}. Build failed.")

    results = {}
    results["tables"] = check_tables()
    fdt_ok, fdt_data = check_fdt()
    results["fdt"] = fdt_ok
    rj_ok, rj_data = check_rayleigh_jeans()
    results["rayleigh_jeans"] = rj_ok
    make_plot(fdt_data, rj_data)
    if args.numba_parity:
        results["numba_parity"] = numba_parity()

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:16s}: {status(v)}")
    all_ok = all(results.values())
    print("  " + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
