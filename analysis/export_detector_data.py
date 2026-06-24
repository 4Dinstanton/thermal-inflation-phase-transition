import numpy as np
import pandas as pd
import os

from gwSpectrum import (
    sensitivity_LISA,
    sensitivity_DECIGO,
    sensitivity_BBO,
    sensitivity_ET,
    sensitivity_aLIGO
)

out_dir = "data/gw_detectors"
os.makedirs(out_dir, exist_ok=True)

detectors = {
    "LISA": (sensitivity_LISA, 1e-5, 0.5),
    "DECIGO": (sensitivity_DECIGO, 1e-3, 100.0),
    "BBO": (sensitivity_BBO, 1e-3, 100.0),
    "ET": (sensitivity_ET, 1.0, 10000.0),
    "aLIGO": (sensitivity_aLIGO, 5.0, 5000.0)
}

for name, (func, fmin, fmax) in detectors.items():
    print(f"Exporting {name}...")
    f_vals = np.logspace(np.log10(fmin), np.log10(fmax), 500)
    omega_vals = func(f_vals)
    
    df = pd.DataFrame({
        "Frequency_Hz": f_vals,
        "Omega_GW_h2": omega_vals
    })
    
    out_path = os.path.join(out_dir, f"{name}_PLISC.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

print("All detector PLISC data exported successfully.")
