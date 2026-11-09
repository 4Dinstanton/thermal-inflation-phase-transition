#!/usr/bin/env python3
"""Compare the C++ two_step evaluator (test_two_step.cpp) with potential/twoStepPotential.py.

    g++ -std=c++14 -O2 -I cosmolattice_ext/models cosmolattice_ext/tests/test_two_step.cpp -o /tmp/test_two_step
    python cosmolattice_ext/tests/check_two_step_cpp.py [--bin /tmp/test_two_step]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from potential.twoStepPotential import TwoStepPotential  # noqa: E402
from simulation.run_cosmolattice import two_step_coefficients  # noqa: E402


def exact_derivatives(pot, phis, T, g, nb, h=1e-5):
    """Tree + exact cosmoTransitions dJb (the Python J table is ~1% off at u < 0.1)."""
    import cosmoTransitions.finiteT as CTFT

    dJ = lambda u: float(np.real(CTFT.dJb_exact(u)))  # noqa: E731
    pref = T**4 / (2 * np.pi**2) * nb
    d1, d2 = [], []
    for p in phis:
        u = g * abs(p) / T
        if u < 2 * h:
            d2J = (-3 * dJ(u) + 4 * dJ(u + h) - dJ(u + 2 * h)) / (2 * h)
        else:
            d2J = (dJ(u + h) - dJ(u - h)) / (2 * h)
        d1.append(np.sign(p) * (float(pot.dV_tree(abs(p))) + pref * g / T * dJ(u)))
        d2.append(float(pot.d2V_tree(abs(p))) + pref * (g / T) ** 2 * d2J)
    return np.array(d1), np.array(d2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/tmp/test_two_step")
    ap.add_argument("--table", default=os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin"))
    args = ap.parse_args()
    V0, a, v, frac, g, nb = 1e28, 3e6, 3e7, 2e-4, 30.0, 10.0
    pot = TwoStepPotential.from_vacua(V0=V0, a=a, v=v, delV1_frac=frac, bosons=((nb, g),))
    k = two_step_coefficients(pot)
    worst = 0.0
    for T in (4e5, 7e5, 1.2e6):
        phis = np.concatenate([[-2 * a, -1e5], np.geomspace(1e2, 1.2 * v, 25), [a, v]])
        cmd = [args.bin, args.table, repr(V0), repr(v), *(repr(x) for x in k), repr(g), repr(nb),
               repr(T), *(repr(float(p)) for p in phis)]
        out = np.loadtxt(subprocess.check_output(cmd, text=True).splitlines())
        ref_V = pot.V_rel(np.abs(phis), T)
        ref_dV, ref_d2V = exact_derivatives(pot, phis, T, g, nb)
        for name, c, r, scale in (("V-V(0)", out[:, 1], ref_V, max(T**4, 1e-6 * V0)),
                                  ("V'", out[:, 2], ref_dV, max(T**3, 1e-6 * V0 / v)),
                                  ("V''", out[:, 3], ref_d2V, max(T**2, 1e-6 * V0 / v**2))):
            err = np.max(np.abs(c - r) / np.maximum(np.abs(r), scale))
            worst = max(worst, err)
            print(f"T={T:.3g} {name:7s} max rel err = {err:.2e}")
    print("C++ vs Python worst relative error:", f"{worst:.2e}")
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
