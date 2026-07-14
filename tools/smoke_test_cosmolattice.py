#!/usr/bin/env python3
"""
End-to-end smoke test of the CosmoLattice thermal-inflation + Langevin pipeline.

Runs an N^3 simulation with inflation-style expansion (T = T0/a) and FDT thermal
noise, starting near the spinodal temperature so the field transitions from the
symmetric (phi ~ 0) phase to the broken phase during the run -- i.e. thermal
fluctuations nucleate and the true-vacuum region grows. Produces:

  - figs/cosmolattice_smoke.png : T(t), a(t), <phi>(t), rms(phi)(t)
  - figs/cosmolattice_smoke_spectrum.png : field power spectrum evolution
  - figs/cosmolattice_smoke_slice.png : revisualized midplane slice (if snapshots enabled)
  - field_states/state_step_*.npz : numba-compatible snapshots for revisualize_snapshots.py

Usage
-----
    python tools/smoke_test_cosmolattice.py                 # 64^3 default
    python tools/smoke_test_cosmolattice.py --Nx 32 --tMax 300
"""
import argparse
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "simulation", "run_cosmolattice.py")
BIN = os.path.join(REPO, "external", "cosmolattice", "build", "thermal_inflation")
TABLE = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")
FIGDIR = os.path.join(REPO, "figs")
M_PL = 2.4e18


def run(args):
    steps = max(1, int(args.tMax / (args.mphi * args.dt_phys) / 20))
    cmd = [
        sys.executable, RUN,
        "--Nx", str(args.Nx), "--T0", str(args.T0), "--gamma", str(args.gamma),
        "--dx_phys", str(args.dx_phys), "--dt_phys", str(args.dt_phys),
        "--tMax", str(args.tMax),
        "--tOutputFreq", str(args.tMax / 100.0),
        "--tOutputInfreq", str(args.tMax / 5.0),
        "--steps", str(steps),
        "--potential_type", args.potential_type,
        "--param_set", args.param_set,
    ]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout); print(out.stderr); sys.exit("run failed")
    for line in out.stdout.splitlines():
        if line.startswith("Wrote input file:"):
            return os.path.join(REPO, os.path.dirname(line.split(":", 1)[1].strip()))
    sys.exit("could not locate output dir")


def main():
    ap = argparse.ArgumentParser(description="CosmoLattice thermal-inflation smoke test")
    ap.add_argument("--Nx", type=int, default=64)
    ap.add_argument("--T0", type=float, default=1150.0,
                    help="Start near the Set-B spinodal (~1150 GeV) so the transition occurs")
    ap.add_argument("--gamma", type=float, default=4.1667e-4)
    ap.add_argument("--dx_phys", type=float, default=1e-3)
    ap.add_argument("--dt_phys", type=float, default=1e-4)
    ap.add_argument("--mphi", type=float, default=1000.0)
    ap.add_argument("--tMax", type=float, default=400.0)
    ap.add_argument("--potential_type", choices=["V_correct", "fermion_only"], default="V_correct")
    ap.add_argument("--param_set", default="set8_smoke")
    ap.add_argument("--skip_revisualize", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TABLE):
        subprocess.check_call([sys.executable, os.path.join(REPO, "tools", "export_thermal_splines.py")])
    if not os.path.exists(BIN):
        subprocess.check_call([sys.executable, RUN, "--install", "--build", "--dry_run"], cwd=REPO)

    print(f"Running {args.Nx}^3 thermal-inflation smoke test (T0={args.T0}, tMax={args.tMax})...")
    d = run(args)
    print(f"Output: {os.path.relpath(d, REPO)}")

    scal = np.loadtxt(os.path.join(d, "average_scalar_0.txt"))
    sf = np.loadtxt(os.path.join(d, "average_scale_factor.txt"))
    if scal.ndim == 1:
        scal = scal[None, :]
    if sf.ndim == 1:
        sf = sf[None, :]

    fStar = args.gamma * M_PL                      # = phi0 (tree VEV); program field unit
    t = scal[:, 0]
    mean_phi = scal[:, 1] * fStar                  # physical <phi> (GeV)
    rms_phi = scal[:, 5] * fStar                   # physical rms(phi)
    a = np.interp(t, sf[:, 0], sf[:, 1])
    T = args.T0 / a

    print(f"  <phi>:  {mean_phi[0]:.3e} -> {mean_phi[-1]:.3e} GeV   (phi0={fStar:.3e})")
    print(f"  rms(phi): {rms_phi[0]:.3e} -> {rms_phi[-1]:.3e} GeV")
    print(f"  T:      {T[0]:.1f} -> {T[-1]:.1f} GeV     a: {a[0]:.4f} -> {a[-1]:.4f}")
    transitioned = abs(mean_phi[-1]) > 0.1 * fStar or rms_phi[-1] > 0.1 * fStar
    print(f"  transition toward broken phase observed: {'YES' if transitioned else 'not yet (try larger tMax or lower T0)'}")

    import glob
    npz_files = sorted(glob.glob(os.path.join(d, "field_states", "state_step_*.npz")))
    print(f"  field snapshots (NPZ): {len(npz_files)}")
    if npz_files:
        sample = np.load(npz_files[-1])
        phi_max = float(np.max(np.abs(sample["phi"])))
        print(f"    last snapshot step={int(sample['step'])}  |phi|_max={phi_max:.3e} GeV")

    if npz_files and not args.skip_revisualize:
        rev = os.path.join(REPO, "postprocess", "revisualize_snapshots.py")
        out_png = os.path.join(FIGDIR, "cosmolattice_smoke_slice.png")
        os.makedirs(FIGDIR, exist_ok=True)
        subprocess.run(
            [sys.executable, rev, d, "--mode", "normalized",
             "--step_min", str(int(np.load(npz_files[0])["step"])),
             "--step_max", str(int(np.load(npz_files[-1])["step"]))],
            cwd=REPO, check=False,
        )
        rev_dir = os.path.join(d, "revisualized_normalized")
        pngs = sorted(glob.glob(os.path.join(rev_dir, "*.png")))
        if pngs:
            import shutil
            shutil.copy(pngs[-1], out_png)
            print(f"Saved {os.path.relpath(out_png, REPO)} (from revisualize_snapshots)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plots")
        return
    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(t, T); ax[0, 0].set_xlabel("program time"); ax[0, 0].set_ylabel("T (GeV)")
    ax[0, 0].axhline(1150, ls="--", c="r", lw=0.8, label="spinodal ~1150")
    ax[0, 0].set_title("Temperature  T = T0/a"); ax[0, 0].legend()
    ax[0, 1].plot(t, a); ax[0, 1].set_xlabel("program time"); ax[0, 1].set_ylabel("a")
    ax[0, 1].set_title("Scale factor (inflation H)")
    ax[1, 0].plot(t, mean_phi / fStar); ax[1, 0].set_xlabel("program time")
    ax[1, 0].set_ylabel(r"$\langle\phi\rangle/\phi_0$"); ax[1, 0].set_title("Mean field")
    ax[1, 1].plot(t, rms_phi / fStar); ax[1, 1].set_xlabel("program time")
    ax[1, 1].set_ylabel(r"rms$(\phi)/\phi_0$")
    ax[1, 1].set_title("Field fluctuations (bubble growth)")
    fig.suptitle(f"CosmoLattice thermal-inflation smoke test ({args.Nx}^3, {args.potential_type})")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "cosmolattice_smoke.png")
    fig.savefig(p, dpi=110)
    print(f"Saved {os.path.relpath(p, REPO)}")

    # Field power spectrum evolution (true-vacuum domain scale growth).
    try:
        spec = np.loadtxt(os.path.join(d, "spectra_scalar_0.txt"))
        kall = spec[:, 0]
        kv = np.unique(np.round(kall, 6))
        nblk = len(kall) // len(kv)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for b in range(nblk):
            blk = spec[b * len(kv):(b + 1) * len(kv)]
            ax2.loglog(blk[:, 0], blk[:, 1] + 1e-300, lw=1,
                       label=f"block {b}" if b in (0, nblk - 1) else None)
        ax2.set_xlabel("k (program)"); ax2.set_ylabel(r"$\Delta^2_\phi(k)$")
        ax2.set_title("Scalar power spectrum evolution"); ax2.legend()
        fig2.tight_layout()
        p2 = os.path.join(FIGDIR, "cosmolattice_smoke_spectrum.png")
        fig2.savefig(p2, dpi=110)
        print(f"Saved {os.path.relpath(p2, REPO)}")
    except Exception as e:
        print(f"(spectrum plot skipped: {e})")


if __name__ == "__main__":
    main()
