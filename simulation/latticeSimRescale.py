import numpy as np
import matplotlib.pyplot as plt
import math
import os

import cosmoTransitions.finiteT as CTFT
from scipy.interpolate import InterpolatedUnivariateSpline
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))
import Potential as p

# =====================================================
# Thermal J derivatives (precomputed spline)
# =====================================================
y2_grid = np.linspace(0, 100, 1000)

dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid])
dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid])

dJb_spline = InterpolatedUnivariateSpline(y2_grid, dJb_grid)
dJf_spline = InterpolatedUnivariateSpline(y2_grid, dJf_grid)

# =====================================================
# Physical parameters
# =====================================================
Nx, Ny = 128, 128

dx_phys = 1e-3
dt_phys = 1e-2 * dx_phys**2
Nt = 1_000_000

lam = 1e-16
mphi = 1_000
eta_phys = 0.3

T0 = 3000
cooling_rate = 1.0

# =====================================================
# Rescaling
# =====================================================
mu = mphi

dx = mu * dx_phys
dt = mu * dt_phys
eta = eta_phys / mu
cooling_rate = cooling_rate / mu

# =====================================================
# Fields
# =====================================================
phi = 0.01 * np.random.randn(Nx, Ny)
pi = np.zeros_like(phi)

# =====================================================
# Potential setup (unchanged)
# =====================================================
bosonMassSquared = 1_000_000
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

param = {
    "lambda": lam,
    "mphi": mphi,
    "epsilon": 0.0,
    "lambdaSix": 0.0,
    "bosonMassSquared": bosonMassSquared,
    "bosonCoupling": bosonCoupling,
    "bosonGaugeCoupling": bosonGaugeCoupling,
    "fermionCoupling": fermionCoupling,
    "fermionGaugeCoupling": fermionGaugeCoupling,
}

VT = p.finiteTemperaturePotential(param)

# =====================================================
# Helper functions
# =====================================================
def laplacian(phi):
    return (
        np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
      + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)
      - 4 * phi
    ) / dx**2

def dVdphi(phi):
    return lam * phi**3 - mphi**2 * phi

def Vprime(phi, T):
    VT.update_T(T)
    db = dJb_spline(VT.bosonic_input(phi)) * VT._dxdphi_boson(phi)
    df = dJf_spline(VT.fermionic_input(phi)) * VT._dxdphi_fermion(phi)
    return dVdphi(phi) + T**4 / (2 * math.pi**2) * (2 * db - df)

def temperature(t):
    return max(T0 - cooling_rate * t, 0.0)

# =====================================================
# Output
# =====================================================
param_set = "set6"
steps= 50_000
save_path = f'figs/latticeSim_rescaled/{param_set}/T0_{T0}_dt_{str(dt)}_a_{cooling_rate}_interval_{steps}'
os.makedirs(save_path, exist_ok=True)

# =====================================================
# Time evolution (RK2 Langevin)
# =====================================================
for n in range(Nt):
    t = n * dt
    T = temperature(t)

    # --- k1 ---
    k1_phi = pi
    k1_pi = laplacian(phi) - eta * pi - Vprime(phi, T) / mu**2

    # --- midpoint ---
    phi_mid = phi + 0.5 * dt * k1_phi
    pi_mid = pi + 0.5 * dt * k1_pi
    T_mid = temperature(t + 0.5 * dt)

    # --- k2 ---
    k2_phi = pi_mid
    k2_pi = laplacian(phi_mid) - eta * pi_mid - Vprime(phi_mid, T_mid) / mu**2

    # --- thermal noise ---
    noise = np.sqrt(2 * eta * T * dt / dx**2) * np.random.randn(Nx, Ny)

    # --- update ---
    phi += dt * k2_phi
    pi += dt * k2_pi + noise

    # --- visualization ---
    if n % steps == 0:
        print(phi)
        plt.clf()
        plt.imshow(phi, origin="lower", cmap="coolwarm", vmin=-1000, vmax=1000)
        plt.colorbar(label=r"$\phi$")
        plt.title(f"t={t/mu:.2e}, T={T:.1f}")
        plt.savefig(f'{save_path}/t_{str(t/mu)}.png')

print("Simulation finished.")
