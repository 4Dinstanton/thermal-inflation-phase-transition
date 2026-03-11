import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))
import Potential as p
import math

import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

y2_grid = np.linspace(0, 100, 1000)

dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid])
dJb_spline = InterpolatedUnivariateSpline(y2_grid, dJb_grid)

dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid])
dJf_spline = InterpolatedUnivariateSpline(y2_grid, dJf_grid)



# -----------------------
# Parameters
# -----------------------
Nx, Ny = 128, 128
dx = 1e-3
dt = 1e-2 * dx **2
Nt = 1_000_000

eta = 0.3

lam = 1e-16
v = 1e+6
mphi = 1000
epsil = 0
lambdaSix = 0
T0 = 7330
a = 1

# -----------------------
# Lattice fields
# -----------------------
phi = 0.01 * np.random.randn(Nx, Ny)
pi = np.zeros_like(phi)
print(phi.shape)

bosonMassSquared = 1000000
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05
param_set = "set6"

param = {param_set: {"lambda": lam, "mphi": mphi, "epsilon": epsil, "lambdaSix" : lambdaSix,
                  "bosonMassSquared": bosonMassSquared, "bosonCoupling": bosonCoupling, "bosonGaugeCoupling": bosonGaugeCoupling,
                  "fermionCoupling": fermionCoupling, "fermionGaugeCoupling": fermionGaugeCoupling}}




VT = p.finiteTemperaturePotential(param[param_set])
VT.update_T(T0)
print(VT.bosonic_input(np.array([1_000_000_000]).reshape(-1,1)))
print(VT.fermionic_input(np.array([1_000_000_000]).reshape(-1,1)))

H = 1e-5

# Potential parameters



# -----------------------
# Helper functions
# -----------------------
def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
      + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
      - 4 * phi
    ) / dx**2

def bosonic_input(x, T):
    return np.sqrt((bosonMassSquared + 0.5 * bosonCoupling**2 * x**2 + (0.25
                                                                                    * bosonCoupling**2 + 2 / 3 * bosonGaugeCoupling**2) * T**2))

def fermionic_input(x, T):
    return np.sqrt(0.5 * fermionCoupling**2 * x**2 + 1 / 6 * fermionGaugeCoupling**2 * T**2)


def dVdphi(phi):
    return lam * phi**3 - mphi**2 * phi

def Vprime(phi, T):
    # simple thermal mass model
    VT.update_T(T)
    thermalCorrectionBosonPotentialDerivative = dJb_spline(VT.bosonic_input(phi)) * VT._dxdphi_boson(phi)
    thermalCorrectionFermionPotentialDerivative = dJf_spline(VT.fermionic_input(phi)) * VT._dxdphi_fermion(phi)
    return dVdphi(phi) + T**4 / \
            (2 * math.pi**2) * (2 * thermalCorrectionBosonPotentialDerivative
                                - thermalCorrectionFermionPotentialDerivative)
    return V_p

def temperature(t):
    #return T0 * np.exp(-H*t)
    return T0 - a*t

#print((1_00000_000_000, 8000))
#print(ho)


# -----------------------
# Time evolution
# -----------------------
history = []
import os
steps = 10_000
save_path = f'figs/latticeSim/{param_set}/T0_{T0}_dt_{str(dt)}_a_{a}_interval_{steps}'
os.makedirs(save_path, exist_ok=True)
for n in range(Nt):
    t = n * dt
    T = temperature(t)
    VT.update_T(T)

    # --- k1 ---
    k1_phi = pi
    k1_pi = laplacian(phi) - eta*pi - Vprime(phi, T)

    # --- midpoint ---
    phi_mid = phi + 0.5 * dt * k1_phi
    pi_mid = pi + 0.5 * dt * k1_pi
    T_mid = temperature(t + 0.5*dt)

    # --- k2 ---
    k2_phi = pi_mid
    k2_pi = laplacian(phi_mid) - eta*pi_mid - Vprime(phi_mid, T_mid)


    # --- noise ---
    noise = np.sqrt(2 * eta * T * dt / dx**2) * np.random.randn(Nx, Ny)

    # --- update ---
    phi += dt * k2_phi
    pi += dt * k2_pi + noise


    if n % steps == 0:
        history.append(phi.copy())

        plt.clf()
        plt.imshow(phi, cmap='coolwarm', origin='lower', vmin=-1000, vmax=1000)
        plt.colorbar(label=r'$\phi$')
        plt.title(f"t = {t:.2f}, T : {T}")
        #plt.title(f"temp : {T}")
        plt.savefig(f'{save_path}/t_{t}.png')
        #plt.waitforbuttonpress()

    history.append(phi.copy())
    #print("phi", phi)



# -----------------------
# Visualization
# -----------------------
plt.imshow(history, aspect='auto', cmap='coolwarm')
plt.colorbar(label=r'$\phi(x,t)$')
plt.xlabel('x')
plt.ylabel('time slice')
plt.title('RK2 Langevin evolution with cooling')
plt.show()
