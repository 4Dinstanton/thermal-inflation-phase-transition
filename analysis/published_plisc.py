"""Published PLISC curves from Schmitz (2020).

Data: https://doi.org/10.5281/zenodo.3689582 (power-law-integrated_sensitivities.tar.gz)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

# Dotted linestyle for turbulence ('....') on log-log GW spectrum plots
LS_TURBULENCE = (0, (1.0, 2.8))

# filename → display name, color, anchor frequency [Hz]
DETECTORS = {
    "LISA": ("plis_LISA.dat", "purple", 3e-3),
    "DECIGO": ("plis_DECIGO.dat", "red", 0.2),
    "BBO": ("plis_BBO.dat", "cyan", 0.05),
    "ET": ("plis_ET.dat", "brown", 5.0),
    "aLIGO": ("plis_HL.dat", "green", 50.0),
}

# Label placement: anchor at PLISC dip or at f_target; offsets are multiplicative.
DETECTOR_LABEL_STYLE: dict[str, dict] = {
    "LISA": {
        "anchor": "f_target",
        "f_mult": 1.0,
        "y_mult": 3.5,
        "ha": "center",
        "va": "bottom",
    },
    "DECIGO": {
        "anchor": "dip",
        "f_mult": 40,
        "y_mult": 150,
        "ha": "left",
        "va": "bottom",
    },
    "BBO": {
        "anchor": "dip",
        "f_mult": 0.01,
        "y_mult": 50,
        "ha": "right",
        "va": "bottom",
    },
    "ET": {
        "anchor": "dip",
        "f_mult": 1.75,
        "y_mult": 2.5,
        "ha": "center",
        "va": "bottom",
    },
    "aLIGO": {
        "anchor": "f_target",
        "f_mult": 0.7,
        "y_mult": 3.0,
        "ha": "center",
        "va": "bottom",
    },
}


def place_detector_label(
    ax,
    name: str,
    fv: np.ndarray,
    sv: np.ndarray,
    f_target: float,
) -> None:
    """Draw a detector name near its published PLISC curve."""
    _, color, _ = DETECTORS[name]
    style = DETECTOR_LABEL_STYLE[name]
    if style["anchor"] == "f_target":
        idx = int(np.argmin(np.abs(fv - f_target)))
    else:
        idx = int(np.argmin(sv))
    y_val = float(sv[idx])
    if y_val < 1e-20 or y_val > 1e-5:
        idx = int(np.argmin(sv))
        y_val = float(sv[idx])
    ax.text(
        fv[idx] * style["f_mult"],
        y_val * style["y_mult"],
        name,
        fontsize=11,
        fontweight="bold",
        color=color,
        alpha=0.9,
        ha=style["ha"],
        va=style["va"],
        zorder=6,
    )


def load_plisc(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load log10(f), log10(h²Ω) columns from a Schmitz ``plis_*.dat`` file."""
    path = DATA_DIR / filename
    arr = np.loadtxt(path, comments="#")
    return 10.0 ** arr[:, 0], 10.0 ** arr[:, 1]


def load_all() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: load_plisc(fname) for name, (fname, _, _) in DETECTORS.items()}
