#!/usr/bin/env python3
"""Find operational T_c1 from lattice snapshots (false-vac fraction <= 1e-5).

For complex fields uses rho = sqrt(phi1^2+phi2^2); for real fields uses |phi|.
"""
import argparse
import csv
import glob
import os

import numpy as np


def _amp(data, escape_phi):
    if "rho" in data:
        return np.asarray(data["rho"], dtype=np.float64)
    if "phi1" in data and "phi2" in data:
        p1 = np.asarray(data["phi1"], dtype=np.float64)
        p2 = np.asarray(data["phi2"], dtype=np.float64)
        return np.sqrt(p1 * p1 + p2 * p2)
    if "phi" in data:
        return np.abs(np.asarray(data["phi"], dtype=np.float64))
    raise KeyError("snapshot lacks phi/rho/phi1+phi2")


def main():
    parser = argparse.ArgumentParser(
        description="Find Tc1 (where false vacuum fraction <= 10^-5)"
    )
    parser.add_argument("state_dir", help="Run directory containing field_states/")
    parser.add_argument(
        "--escape_phi",
        type=float,
        default=10000.0,
        help="Escape threshold on |phi| or rho (GeV)",
    )
    parser.add_argument(
        "--frac_thresh",
        type=float,
        default=1e-5,
        help="False-vac fraction threshold for Tc1 (default 1e-5)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path for (step,T,frac_false)",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.state_dir, "field_states", "state_step_*.npz"))
    )
    if not files:
        print(f"No state_step_*.npz found in {args.state_dir}/field_states")
        return

    data = []
    for f in files:
        d = np.load(f)
        amp = _amp(d, args.escape_phi)
        T = float(d["temperature"])
        step = int(d["step"])
        frac_false = float(np.mean(amp <= args.escape_phi))
        data.append((T, step, frac_false))

    data.sort(key=lambda x: x[0], reverse=True)

    print(f"escape_phi = {args.escape_phi:.2e}  frac_thresh = {args.frac_thresh:g}")
    print(f"{'Step':>10} | {'Temperature (GeV)':>20} | {'False Vacuum Frac':>20}")
    print("-" * 57)

    found_tc1 = False
    tc1 = None
    for T, step, frac in data:
        marker = ""
        if frac <= args.frac_thresh and not found_tc1:
            marker = " <--- Tc1 (fraction <= {:.0e})".format(args.frac_thresh)
            found_tc1 = True
            tc1 = (T, step, frac)
        print(f"{step:10d} | {T:20.4f} | {frac:20.6e}{marker}")

    if tc1 is None:
        print("\nTc1 not reached in available snapshots "
              f"(min false-vac frac = {min(d[2] for d in data):.3e}).")
    else:
        print(f"\nTc1 ≈ {tc1[0]:.4f} GeV  (step {tc1[1]}, frac={tc1[2]:.3e})")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "temperature_GeV", "false_vac_frac"])
            for T, step, frac in sorted(data, key=lambda x: x[1]):
                w.writerow([step, f"{T:.8g}", f"{frac:.8e}"])
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
