import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os

os.makedirs('figs', exist_ok=True)

def Jb_exact(y2):
    y = np.sqrt(y2)
    def integrand(x):
        return x**2 * np.log(1 - np.exp(-np.sqrt(x**2 + y2)))
    return integrate.quad(integrand, 0, np.inf)[0]

def Jb_high_T(y2):
    a_b = np.exp(5.4076)
    y = np.sqrt(y2)
    term0 = - (np.pi**4 / 45)
    term2 = (np.pi**2 / 12) * y2
    term3 = - (np.pi / 6) * y**3
    term4 = - (y2**2 / 32) * np.log(y2 / a_b + 1e-10)
    return term0 + term2 + term3 + term4

def Jb_high_T_no_cubic(y2):
    a_b = np.exp(5.4076)
    y = np.sqrt(y2)
    term0 = - (np.pi**4 / 45)
    term2 = (np.pi**2 / 12) * y2
    term4 = - (y2**2 / 32) * np.log(y2 / a_b + 1e-10)
    return term0 + term2 + term4

def Jf_exact(y2):
    y = np.sqrt(y2)
    def integrand(x):
        return x**2 * np.log(1 + np.exp(-np.sqrt(x**2 + y2)))
    return integrate.quad(integrand, 0, np.inf)[0]

def Jf_high_T(y2):
    a_f = np.exp(2.6351)
    term0 = (7 * np.pi**4 / 360)
    term2 = - (np.pi**2 / 24) * y2
    term4 = - (y2**2 / 32) * np.log(y2 / a_f + 1e-10)
    return term0 + term2 + term4

# --- 1. Basic Jb and Jf tests vs y ---
y2_vals = np.logspace(-6, 0, 100)
Jb_ex = np.array([Jb_exact(y2) for y2 in y2_vals])
Jb_ht = np.array([Jb_high_T(y2) for y2 in y2_vals])
Jb_ht_nc = np.array([Jb_high_T_no_cubic(y2) for y2 in y2_vals])

Jf_ex = np.array([Jf_exact(y2) for y2 in y2_vals])
Jf_ht = np.array([Jf_high_T(y2) for y2 in y2_vals])

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(np.sqrt(y2_vals), Jb_ex, 'k-', label='Exact $J_B(y)$', linewidth=2)
plt.plot(np.sqrt(y2_vals), Jb_ht, 'b--', label='Expansion (with $y^3$)')
plt.plot(np.sqrt(y2_vals), Jb_ht_nc, 'r:', label='Expansion (no $y^3$)')
plt.title('Bosonic Thermal Integral and its High-T Expansion')
plt.xlabel('$y = m_{eff}/T$')
plt.ylabel('$J_B(y)$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(y2_vals), Jf_ex, 'k-', label='Exact $J_F(y)$', linewidth=2)
plt.plot(np.sqrt(y2_vals), Jf_ht, 'g--', label='Expansion (no $y^3$ needed)')
plt.title('Fermionic Thermal Integral and its High-T Expansion')
plt.xlabel('$y = m_{eff}/T$')
plt.ylabel('$J_F(y)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figs/thermal_expansion_check.png', dpi=300)
plt.close()

# --- 2. Error Plot ---
plt.figure(figsize=(10, 5))
plt.loglog(np.sqrt(y2_vals), np.abs(Jb_ex - Jb_ht), 'b-', label='Error: Exact - Exp (with $y^3$)')
plt.loglog(np.sqrt(y2_vals), np.abs(Jb_ex - Jb_ht_nc), 'r-', label='Error: Exact - Exp (no $y^3$)')
plt.loglog(np.sqrt(y2_vals), 1e-1 * np.sqrt(y2_vals)**3, 'k:', label='$\propto y^3$')
plt.loglog(np.sqrt(y2_vals), 1e-2 * np.sqrt(y2_vals)**4, 'g:', label='$\propto y^4$')
plt.title('Bosonic Integral Expansion Error vs $y$')
plt.xlabel('$y = m_{eff}/T$')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.savefig('figs/thermal_expansion_error.png', dpi=300)
plt.close()


# --- 3. Effect of cubic term on the potential shape for negative and positive phi ---
phi_vals = np.linspace(-3, 3, 200)
g = 1.0
T_high = 10.0
# m_eff^2 = g^2 phi^2 / 2 + m0^2 (small m0 to avoid zero-mass singularity)
m0_2 = 1e-4
def meff2(phi): return 0.5 * g**2 * phi**2 + m0_2

# Thermal potential contribution: Delta V = T^4 / (2 pi^2) * Jb(meff/T)
V_ex = np.array([T_high**4 / (2*np.pi**2) * Jb_exact(meff2(p)/T_high**2) for p in phi_vals])
V_ht = np.array([T_high**4 / (2*np.pi**2) * Jb_high_T(meff2(p)/T_high**2) for p in phi_vals])
V_ht_nc = np.array([T_high**4 / (2*np.pi**2) * Jb_high_T_no_cubic(meff2(p)/T_high**2) for p in phi_vals])

# Subtract V(0) for better comparison of the shape (forces V(0) = 0)
zero_idx = np.argmin(np.abs(phi_vals))
V_ex -= V_ex[zero_idx]
V_ht -= V_ht[zero_idx]
V_ht_nc -= V_ht_nc[zero_idx]

plt.figure(figsize=(8, 6))
plt.plot(phi_vals, V_ex, 'k-', linewidth=3, label='Exact $V_{th}$')
plt.plot(phi_vals, V_ht, 'b--', linewidth=2, label='High-T Exp (with $-|\phi|^3$)')
plt.plot(phi_vals, V_ht_nc, 'r:', linewidth=2, label='High-T Exp (no $-|\phi|^3$)')
plt.title(f'Bosonic Thermal Potential Shape at High $T$ ($T={T_high}$)')
plt.xlabel('$\phi$')
plt.ylabel('$\Delta V_{th}(\phi) - \Delta V_{th}(0)$')
plt.legend()
plt.grid(True)
plt.savefig('figs/thermal_potential_shape_phi.png', dpi=300)
plt.close()


# --- 4. Evolution of potential shape from Low-T to High-T ---
T_vals = [0.5, 1.0, 2.0, 5.0, 10.0]
plt.figure(figsize=(8, 6))

for T in T_vals:
    # Compute the exact thermal correction
    V = np.array([T**4 / (2*np.pi**2) * Jb_exact(meff2(p)/T**2) for p in phi_vals])
    V -= V[zero_idx]

    # We normalize by the maximum value (at phi=3 or -3) so we can compare the purely geometric shapes.
    # At low T, the potential is very flat near 0 and then rises slowly or exponentially.
    # At high T, the cubic term causes it to bow downwards significantly relative to a pure parabola.
    norm = V[-1] if V[-1] != 0 else 1.0
    plt.plot(phi_vals, V / norm, label=f'$T = {T}$')

# Add a pure parabola for reference
parabola = phi_vals**2
parabola -= parabola[zero_idx]
parabola /= parabola[-1]
plt.plot(phi_vals, parabola, 'k:', linewidth=2, label='Pure Parabola ($\phi^2$)')

plt.title('Shape Evolution of Thermal Correction (Normalized)')
plt.xlabel('$\phi$')
plt.ylabel('Normalized [ $\Delta V_{th}(\phi) - \Delta V_{th}(0)$ ]')
plt.legend()
plt.grid(True)
plt.savefig('figs/thermal_potential_evolution_phi.png', dpi=300)
plt.close()

print("All figures successfully saved to figs/ directory:")
print(" - figs/thermal_expansion_check.png")
print(" - figs/thermal_expansion_error.png")
print(" - figs/thermal_potential_shape_phi.png")
print(" - figs/thermal_potential_evolution_phi.png")