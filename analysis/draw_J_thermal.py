#!/usr/bin/env python3
"""Plot J_B(u), J_F(u), and J_B(u) + J_F(u) for u in [0, 1].

Uses the definitions in semiAnalytical.py (paper Eqs. 2 & 4) with u = m_eff / T:
  J_B(u) = J_B(u^2) = +(1/2pi^2) int_0^inf dx x^2 ln(1 - exp(-sqrt(x^2+u^2)))
  J_F(u) = J_F(u^2) = -(1/2pi^2) int_0^inf dx x^2 ln(1 + exp(-sqrt(x^2+u^2)))
"""

from __future__ import annotations

import os
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

MPL = 2.2
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 11


def JF(z: float) -> float:
    def integrand(x):
        E = sqrt(x * x + z)
        if E > 500:
            return 0.0
        return x * x * np.log(1.0 + np.exp(-E))

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)
    return -val / (2.0 * pi**2)


def JB(z: float) -> float:
    def integrand(x):
        E = sqrt(x * x + z)
        if E > 500:
            return 0.0
        arg = 1.0 - np.exp(-E)
        if arg <= 0:
            return 0.0
        return x * x * np.log(arg)

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)
    return val / (2.0 * pi**2)


def main() -> None:
    u = np.linspace(0.0, 1.0, 80)
    jb = np.array([JB(uu**2) for uu in u])
    jf = np.array([JF(uu**2) for uu in u])
    jsum = jb + jf

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(u, jb, color="royalblue", lw=MPL, label=r"$J_B(u)$")
    ax.plot(u, jf, color="crimson", lw=MPL, label=r"$J_F(u)$")
    ax.plot(u, jsum, color="black", lw=MPL, ls="--", label=r"$J_B(u) + J_F(u)$")
    ax.set_xlabel(r"$u = m_{\mathrm{eff}}\,/\,T$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"$J(u)$", fontsize=FONTSIZE_LABEL)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE_LEGEND, loc="best")
    ax.set_title(r"Thermal integrals $J_B$, $J_F$ (Eqs. 2 \& 4)", fontsize=FONTSIZE_LABEL)

    os.makedirs("figs/action", exist_ok=True)
    out = "figs/action/J_B_F_vs_u.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
