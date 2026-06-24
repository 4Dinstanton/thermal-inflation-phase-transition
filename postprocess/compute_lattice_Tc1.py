#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Find Tc1 (where false vacuum fraction <= 10^-5)")
    parser.add_argument("state_dir", help="Directory containing state_step_*.npz")
    parser.add_argument("--escape_phi", type=float, default=10000.0, help="Escape threshold (default 10000)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.state_dir, "field_states", "state_step_*.npz")))
    if not files:
        print(f"No state_step_*.npz found in {args.state_dir}")
        return

    data = []
    for f in files:
        d = np.load(f)
        phi = d["phi"].astype(np.float64)
        T = float(d["temperature"])
        step = int(d["step"])
        frac_false = np.mean(np.abs(phi) <= args.escape_phi)
        data.append((T, step, frac_false))
    
    # Sort by temperature descending (usually it cools down)
    data.sort(key=lambda x: x[0], reverse=True)
    
    print(f"escape_phi = {args.escape_phi:.2e}")
    print(f"{'Step':>10} | {'Temperature (GeV)':>20} | {'False Vacuum Frac':>20}")
    print("-" * 57)
    
    found_tc1 = False
    for T, step, frac in data:
        marker = ""
        if frac <= 1e-5 and not found_tc1:
            marker = " <--- Tc1 (fraction <= 1e-5)"
            found_tc1 = True
            
        print(f"{step:10d} | {T:20.4f} | {frac:20.6e}{marker}")

if __name__ == "__main__":
    main()
