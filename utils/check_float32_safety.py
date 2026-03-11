#!/usr/bin/env python
"""
Check if float32 precision is safe for the lattice simulation.
Analyzes actual ranges of phi, V, V', and checks against float32 limits.
"""
import numpy as np
import math
import sys

print("=" * 70)
print("FLOAT32 SAFETY ANALYSIS")
print("=" * 70)

# Your parameters
lam = 1e-16
mphi = 1000.0
T0 = 7330.0
bosonMassSquared = 1_000_000.0
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

print("\nParameters:")
print(f"  λ = {lam:.2e}")
print(f"  m_φ = {mphi:.2e}")
print(f"  T0 = {T0:.2e}")
print(f"  bosonMassSquared = {bosonMassSquared:.2e}")

# Float32 limits
float32_max = np.finfo(np.float32).max
float32_min = np.finfo(np.float32).min
float32_eps = np.finfo(np.float32).eps
float64_eps = np.finfo(np.float64).eps

print(f"\nFloat32 limits:")
print(f"  Max value: ±{float32_max:.2e}")
print(f"  Min normal: {float32_min:.2e}")
print(f"  Machine epsilon: {float32_eps:.2e} (~7 decimal digits)")
print(f"Float64 epsilon: {float64_eps:.2e} (~16 decimal digits)")

# Analyze field range
print("\n" + "-" * 70)
print("FIELD RANGE ANALYSIS")
print("-" * 70)

# From your plots: phi ranges roughly -1000 to +1000 in your units
phi_max_expected = 1e3  # Conservative estimate from your plots
phi_test = np.array([-phi_max_expected, 0.0, phi_max_expected])

print(f"\nExpected φ range: ±{phi_max_expected:.2e}")
print(f"  Safe for float32? {abs(phi_max_expected) < float32_max}")

# Tree-level potential and derivative
print("\n" + "-" * 70)
print("TREE-LEVEL POTENTIAL")
print("-" * 70)

V_tree = lam / 4 * phi_test**4 - mphi**2 / 2 * phi_test**2
dV_tree = lam * phi_test**3 - mphi**2 * phi_test

print(f"\nAt φ = ±{phi_max_expected:.2e}:")
print(f"  V_tree: {abs(V_tree).max():.6e}")
print(f"  dV/dφ (tree): {abs(dV_tree).max():.6e}")
print(
    f"  Safe for float32? {abs(V_tree).max() < float32_max and abs(dV_tree).max() < float32_max}"
)

# Thermal corrections
print("\n" + "-" * 70)
print("THERMAL CORRECTIONS")
print("-" * 70)

T = T0
T2 = T * T
T4 = T2 * T2
pref = T4 / (2.0 * math.pi * math.pi)

print(f"\nAt T = {T:.2e}:")
print(f"  T^4/(2π²) = {pref:.6e}")

# Compute thermal mass squared (worst case at large phi)
gb2 = bosonCoupling * bosonCoupling
gg2 = bosonGaugeCoupling * bosonGaugeCoupling
gf2 = fermionCoupling * fermionCoupling
gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2

xb_sq = bosonMassSquared + 0.5 * gb2 * phi_max_expected**2 + coef_b_T * T2
xf_sq = 0.5 * gf2 * phi_max_expected**2 + (1.0 / 6.0) * gfg2 * T2

print(f"\nThermal mass arguments:")
print(f"  x_b² (boson): {xb_sq:.6e}")
print(f"  x_f² (fermion): {xf_sq:.6e}")

xb = math.sqrt(xb_sq) / T
xf = math.sqrt(xf_sq) / T

print(f"  x_b = {xb:.6e}")
print(f"  x_f = {xf:.6e}")

# dJ/dx is roughly O(1) for x ~ 1-10, falls off for large x
# Worst case: dJ ~ O(1)
dJb_est = 1.0  # Conservative estimate
dJf_est = 1.0

# Derivative contributions
dxb_dphi = 0.5 * gb2 * phi_max_expected / (T2 * max(xb, 1e-20))
dxf_dphi = 0.5 * gf2 * phi_max_expected / (T2 * max(xf, 1e-20))

thermal_correction = pref * (2.0 * dJb_est * dxb_dphi - dJf_est * dxf_dphi)

print(f"\nThermal contribution to V':")
print(f"  dx_b/dφ: {dxb_dphi:.6e}")
print(f"  dx_f/dφ: {dxf_dphi:.6e}")
print(f"  Thermal correction (estimate): {abs(thermal_correction):.6e}")

# Total derivative
V_prime_total = abs(dV_tree).max() + abs(thermal_correction)

print(f"\n" + "-" * 70)
print("TOTAL V' ESTIMATE")
print("-" * 70)
print(f"  |dV/dφ| (tree): {abs(dV_tree).max():.6e}")
print(f"  |dV/dφ| (thermal): {abs(thermal_correction):.6e}")
print(f"  |dV/dφ| (total): {V_prime_total:.6e}")

# Check safety
print(f"\n" + "=" * 70)
print("FLOAT32 SAFETY CHECK")
print("=" * 70)

safe_value = V_prime_total < 0.01 * float32_max  # Use 1% of max for safety margin
safe_precision = (
    V_prime_total * float32_eps < abs(dV_tree).max() * 1e-3
)  # Relative error < 0.1%

print(f"\n✓ Values within range? {V_prime_total < float32_max}")
print(f"  (Using 1% safety margin: {safe_value})")
print(f"\n✓ Precision adequate? (Checking if relative error < 0.1%)")

# Compute relative precision loss
rel_error_f32 = V_prime_total * float32_eps / (abs(dV_tree).max() + 1e-30)
rel_error_f64 = V_prime_total * float64_eps / (abs(dV_tree).max() + 1e-30)

print(f"  Float32 relative error: {rel_error_f32:.2e}")
print(f"  Float64 relative error: {rel_error_f64:.2e}")
print(f"  Precision loss: {rel_error_f32/rel_error_f64:.1f}× worse with float32")

# Additional fields that go through calculations
print(f"\n" + "-" * 70)
print("OTHER FIELDS")
print("-" * 70)

# Laplacian: O(phi/dx²)
dx = 1.0  # Your typical dx in rescaled units
lap_est = phi_max_expected / dx**2
print(f"  Laplacian ∇²φ: ~{lap_est:.6e}")

# Noise: sqrt(2*eta*T*dt/dx²)
eta = 0.3 / mphi
dt = 1e-2 * dx**2
noise_scale = math.sqrt(2.0 * eta * T * dt / dx**2)
print(f"  Noise scale: ~{noise_scale:.6e}")

# Pi (conjugate momentum): O(phi/dt) ~ phi*100
pi_est = phi_max_expected / dt
print(f"  π field: ~{pi_est:.6e}")

all_safe = (
    V_prime_total < float32_max and lap_est < float32_max and pi_est < float32_max
)

print(f"\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if safe_value and rel_error_f32 < 1e-3:
    print("\n✅ FLOAT32 IS SAFE")
    print("   - All values well within float32 range")
    print("   - Relative precision loss is acceptable")
    print("   - Expected speedup: ~1.5-2× (memory bandwidth bound)")
    print("\nTo enable float32:")
    print("   1. Change field arrays:")
    print("      phi = 0.01 * np.random.randn(Nx, Ny).astype(np.float32)")
    print("      pi = np.zeros((Nx, Ny), dtype=np.float32)")
    print("   2. Keep parameters as float64 (no change needed)")
elif all_safe and not safe_precision:
    print("\n⚠️  FLOAT32 USABLE BUT WITH CAUTION")
    print("   - Values are within range")
    print("   - But precision may affect small corrections")
    print("   - Test with comparison against float64 first")
else:
    print("\n❌ FLOAT32 NOT RECOMMENDED")
    print("   - Values may overflow or lose too much precision")
    print("   - Stick with float64")

print("=" * 70)
