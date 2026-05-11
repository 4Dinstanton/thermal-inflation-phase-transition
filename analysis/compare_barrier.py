import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os

os.makedirs('figs', exist_ok=True)

def Jb_exact(y2):
    y = np.sqrt(np.abs(y2))
    def integrand(x):
        return x**2 * np.log(1 - np.exp(-np.sqrt(x**2 + y2)))
    if y2 >= 0:
        return integrate.quad(integrand, 0, np.inf)[0]
    return 0

def Jf_exact(y2):
    y = np.sqrt(np.abs(y2))
    def integrand(x):
        return x**2 * np.log(1 + np.exp(-np.sqrt(x**2 + y2)))
    if y2 >= 0:
        return integrate.quad(integrand, 0, np.inf)[0]
    return 0

# --- Physics Parameters (from drawPotential.py defaults) ---
mphi = 100.0  # Just an example from default config
param_dict = {
    "mphi": 100.0,
    "lam": 0.05,
    "lambdaSix": 1.0 / (5.0 * 10**8) ** 2,
    "bosonCoupling": 1.0,
    "fermionCoupling": 1.0,
    "bosonGaugeCoupling": 0.0,
    "fermionGaugeCoupling": 0.0,
    "bosonMassSquared": 0.0,
}

g_b = param_dict["bosonCoupling"]
g_f = param_dict["fermionCoupling"]
m0_b_2 = param_dict["bosonMassSquared"]
# Fermions don't usually have a bare mass in this model
m0_f_2 = 0.0

n_b = 1
n_f = 1

def V_tree(phi):
    return -0.5 * param_dict["mphi"]**2 * phi**2 + param_dict["lambdaSix"] * phi**6

def meff2_b(phi):
    return 0.5 * g_b**2 * phi**2 + m0_b_2

def meff2_f(phi):
    return 0.5 * g_f**2 * phi**2 + m0_f_2

def V_eff_boson(phi, T):
    if T == 0: return V_tree(phi)
    # Using exact integrals, but optimized for a small grid
    return V_tree(phi) + n_b * (T**4 / (2*np.pi**2)) * np.array([Jb_exact(meff2_b(p)/T**2) for p in np.atleast_1d(phi)])

def V_eff_fermion(phi, T):
    if T == 0: return V_tree(phi)
    # Note: Using your project convention: + Jf (in the model Jf includes a positive coefficient, but mathematically it's 2*Jb + Jf with Jf returning a positive term. Let's trace it)
    # Actually, in flatonPotential.py:
    # correctedPotential = treeLevelPotential + T^4/(2pi^2) * (2*Jb_exact + Jf_exact)
    # Let's match your exact convention! Jf_exact is calculated normally.
    return V_tree(phi) + n_f * (T**4 / (2*np.pi**2)) * np.array([Jf_exact(meff2_f(p)/T**2) for p in np.atleast_1d(phi)])

def Jb_high_T(y2):
    y = np.sqrt(np.abs(y2))
    a_b = np.exp(5.4076)
    term0 = - (np.pi**4 / 45)
    term2 = (np.pi**2 / 12) * y2
    term3 = - (np.pi / 6) * y**3
    term4 = - (y2**2 / 32) * np.log(np.abs(y2) / a_b + 1e-10)
    return np.where(y2 >= 0, term0 + term2 + term3 + term4, 0)

def Jf_high_T(y2):
    y = np.sqrt(np.abs(y2))
    a_f = np.exp(2.6351)
    term0 = (7 * np.pi**4 / 360)
    term2 = - (np.pi**2 / 24) * y2
    term4 = - (y2**2 / 32) * np.log(np.abs(y2) / a_f + 1e-10)
    return np.where(y2 >= 0, term0 + term2 + term4, 0)

def V_eff_boson_fast(phi, T):
    if T == 0: return V_tree(phi)
    V_th = n_b * (T**4 / (2*np.pi**2)) * Jb_high_T(meff2_b(phi)/T**2)
    return V_tree(phi) + V_th

def V_eff_fermion_fast(phi, T):
    if T == 0: return V_tree(phi)
    V_th = n_f * (T**4 / (2*np.pi**2)) * Jf_high_T(meff2_f(phi)/T**2)
    return V_tree(phi) + V_th

# Fast scan using coarse grid to find T_bar
def find_barrier_temperature(V_eff_func, T_start=50000.0, T_end=10.0, steps=2000):
    T_vals = np.linspace(T_start, T_end, steps)
    # The true vacuum is at phi ~ 5e8, but the barrier forms VERY close to the origin, around phi ~ T.
    # We must scan near the origin!
    phi_test = np.linspace(0.1, 50000.0, 1000)

    for T in T_vals:
        V = V_eff_func(phi_test, T)
        dV = np.diff(V)

        starts_up = dV[0] > 0
        if starts_up:
            if np.any(dV < 0):
                return T
    return None

print("Finding T_bar analytically...")
T_bar_boson = find_barrier_temperature(V_eff_boson_fast)
T_bar_fermion = find_barrier_temperature(V_eff_fermion_fast)

print(f"Boson T_bar: {T_bar_boson:.3f}" if T_bar_boson else "Boson T_bar: None")
print(f"Fermion T_bar: {T_bar_fermion:.3f}" if T_bar_fermion else "Fermion T_bar: Never forms a barrier!")

# Finer plot grid
phi_vals = np.linspace(0, 40000.0, 500)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

TB = T_bar_boson if T_bar_boson else 25000.0

T_plot = [TB + 2000, TB + 500, TB, TB - 500, TB - 2000]
colors = plt.cm.plasma(np.linspace(0, 0.9, len(T_plot)))

for i, T in enumerate(T_plot):
    V_B = V_eff_boson(phi_vals, T)
    V_B -= V_B[0]

    lw = 3 if abs(T - TB) < 1e-4 else 2
    ls = '-' if abs(T - TB) < 1e-4 else '--'
    label_B = f'$T = {T:.2f}$ ($T_{{bar}}$)' if abs(T - TB) < 1e-4 else f'$T = {T:.2f}$'

    ax1.plot(phi_vals, V_B, color=colors[i], linewidth=lw, linestyle=ls, label=label_B)

    V_F = V_eff_fermion(phi_vals, T)
    V_F -= V_F[0]
    ax2.plot(phi_vals, V_F, color=colors[i], linewidth=2, linestyle='--', label=f'$T = {T:.2f}$')

ax1.set_title(f'Bosonic Potential (n_b=1): Barrier begins forming at $T_{{bar}} \\approx {TB:.2f}$')
ax1.set_ylabel('$V_{eff}(\phi) - V_{eff}(0)$')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.legend()
ax1.grid(True)
ax1.set_ylim(-1e15, 1e15)

ax2.set_title('Fermionic Potential (n_f=1): NO barrier ever forms (Second Order)')
ax2.set_xlabel('$\phi$')
ax2.set_ylabel('$V_{eff}(\phi) - V_{eff}(0)$')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.legend()
ax2.grid(True)
ax2.set_ylim(-1e15, 1e15)

plt.tight_layout()
plt.savefig('figs/barrier_onset_temperature.png', dpi=300)
plt.close()

print("Saved barrier onset plot to figs/barrier_onset_temperature.png")
