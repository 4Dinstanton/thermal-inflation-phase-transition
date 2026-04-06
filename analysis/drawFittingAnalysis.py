"""
Plot r_c, phi_esc, S3/T from tunneling CSV and test analytic fitting ansatze:
    S3/T    ~ a * (ln(T-b))^3 + c
    r_c     ~ a * ln(T-b)/(T-b) + c
    phi_esc ~ a * T ln(T-b)   + c
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── data ──────────────────────────────────────────────────────────────────
import pandas as pd

dataset = "set7"

csv_path = f"data/tunneling/{dataset}/T-S_param_{dataset}_lambdaSix_0E+00_fermion_only_Long.csv"
df = pd.read_csv(csv_path).iloc[1:]

# Slice here – e.g. filter by temperature range:
# df = df[(df["T"] >= 7000) & (df["T"] <= 8500)]
# Or keep every other row:
# df = df.iloc[::2]

T = df["T"].values
S3_T = df["S3/T"].values
r_c = df["r_c"].values
phi_esc = df["phi_esc"].values


# ── fitting models ────────────────────────────────────────────────────────
def model_lnT3(T, a, b, c):
    return a * np.log(T - b) ** 3 + c


def model_lnT_over_T(T, a, b, c):
    return a * np.log(T - b) / (T - b) + c


def model_TlnT(T, a, b, c):
    return a * T * np.log(T - b) + c


def model_sqrtT(T, a, b, c):
    return a * np.sqrt(T - b) + c


def model_phi_esc_combined(T, a, b, c):
    lnT = np.log(T)
    return a * T * lnT + b * T**3 / lnT**2 + c


def model_S3T_combined(T, a, b):
    lnT = np.log(T)
    return a * (lnT + T**2 / lnT**2) ** 3 + b


def model_linear(T, a, b):
    return a * T + b


def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


# ── fit each quantity ─────────────────────────────────────────────────────
quantities = [
    ("S3/T", S3_T, model_lnT3, r"$a\,(\ln(T-b))^3 + c$"),
    ("r_c", r_c, model_lnT_over_T, r"$a\,\ln(T-b) / (T-b) + c$"),
    ("phi_esc", phi_esc, model_TlnT, r"$a\,T\ln(T-b) + c$"),
]

fits = {}
max_params = 4
header_names = ["a", "b", "c", "d"]
hdr = (
    f"{'Quantity':>10s}  "
    + "  ".join(f"{n:>14s}" for n in header_names)
    + f"  {'R^2':>10s}  Ansatz"
)
print("=" * len(hdr))
print(hdr)
print("-" * len(hdr))

for name, y, model, label in quantities:
    popt, _ = curve_fit(model, T, y, maxfev=100000)
    y_fit = model(T, *popt)
    R2 = r_squared(y, y_fit)
    fits[name] = (popt, y_fit, R2, label)
    params_str = "  ".join(f"{p:>+14.6e}" for p in popt)
    pad = max_params - len(popt)
    if pad > 0:
        params_str += "  " + "  ".join(" " * 14 for _ in range(pad))
    print(f"{name:>10s}  {params_str}  {R2:>10.6f}  {label}")

print("=" * len(hdr))

popt_sqrt, _ = curve_fit(model_sqrtT, T, S3_T)
y_fit_sqrt = model_sqrtT(T, *popt_sqrt)
R2_sqrt = r_squared(S3_T, y_fit_sqrt)
fits["S3/T_sqrt"] = (popt_sqrt, y_fit_sqrt, R2_sqrt, r"$a\,\sqrt{T} + b$")
print(
    f"\n  S3/T  alt:  a={popt_sqrt[0]:>+14.6e}  b={popt_sqrt[1]:>+14.6e}"
    f"  R^2={R2_sqrt:.6f}   a*sqrt(T)+b"
)

print("\nLinear baseline  a*T + b  for comparison:")
print("-" * 55)
for name, y, _, _ in quantities:
    popt_lin, _ = curve_fit(model_linear, T, y)
    R2_lin = r_squared(y, model_linear(T, *popt_lin))
    print(f"  {name:>10s}  R^2 = {R2_lin:.6f}")
print("-" * 55)

# ── main plot ─────────────────────────────────────────────────────────────
T_fine = np.linspace(T.min(), T.max(), 300)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, y, model, _) in zip(axes, quantities):
    popt, y_fit, R2, label = fits[name]
    ax.plot(T, y, "ko", markersize=4, label="data")
    ax.plot(
        T_fine,
        model(T_fine, *popt),
        "r-",
        lw=2,
        label=f"fit: {label}\n$R^2 = {R2:.5f}$",
    )
    ax.set_xlabel("T  (GeV)")
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Tunneling quantities  –  fermion_only  ({dataset})", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"figs/finiteTemp/fitting_analysis_fermion_only_{dataset}.png", dpi=200)
plt.close(fig)

# ── residual plot ─────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, y, model, _) in zip(axes2, quantities):
    popt, y_fit, R2, label = fits[name]
    residual = (y - y_fit) / np.abs(y) * 100
    ax.plot(T, residual, "s-", markersize=4, color="steelblue")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("T  (GeV)")
    ax.set_ylabel("Residual  (%)")
    ax.set_title(f"{name} residual  (R²={R2:.5f})")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"figs/finiteTemp/fitting_residuals_fermion_only_{dataset}.png", dpi=200)
plt.close(fig2)

# ── sqrt(T) dedicated figure ──────────────────────────────────────────────
p_sq, y_fit_sq, R2_sq, lbl_sq = fits["S3/T_sqrt"]

fig3, (ax_fit, ax_res) = plt.subplots(1, 2, figsize=(12, 5))

ax_fit.plot(T, S3_T, "ko", markersize=5, label="data")
ax_fit.plot(
    T_fine,
    model_sqrtT(T_fine, *p_sq),
    "b-",
    lw=2,
    label=(
        rf"$a\sqrt{{T - b}} + c$"
        f"\na = {p_sq[0]:.4e}"
        f"\nb = {p_sq[1]:.4e}"
        f"\nc = {p_sq[2]:.4e}"
        f"\n$R^2 = {R2_sq:.6f}$"
    ),
)
ax_fit.set_xlabel("T  (GeV)")
ax_fit.set_ylabel("S3 / T")
ax_fit.set_title(r"$S_3/T$ fitted to $a\sqrt{T-b}+c$")
ax_fit.legend(fontsize=9)
ax_fit.grid(True, alpha=0.3)

residual_sq = (S3_T - y_fit_sq) / np.abs(S3_T) * 100
ax_res.plot(T, residual_sq, "s-", markersize=4, color="steelblue")
ax_res.axhline(0, color="k", ls="--", lw=0.8)
ax_res.set_xlabel("T  (GeV)")
ax_res.set_ylabel("Residual  (%)")
ax_res.set_title(rf"$S_3/T$ sqrt fit residual  ($R^2={R2_sq:.5f}$)")
ax_res.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"figs/finiteTemp/fitting_sqrtT_fermion_only_{dataset}.png", dpi=200)
plt.close(fig3)

# ── combined-ansatz figure ────────────────────────────────────────────────
# S3/T  ~ a * (lnT + T^2/(lnT)^2)^3 + b
popt_s3c, _ = curve_fit(model_S3T_combined, T, S3_T)
yfit_s3c = model_S3T_combined(T, *popt_s3c)
R2_s3c = r_squared(S3_T, yfit_s3c)

# phi_esc ~ a*T*lnT + b*T^3/(lnT)^2 + c
popt_phic, _ = curve_fit(model_phi_esc_combined, T, phi_esc)
yfit_phic = model_phi_esc_combined(T, *popt_phic)
R2_phic = r_squared(phi_esc, yfit_phic)

print("\n" + "=" * 70)
print("Combined ansatze")
print("-" * 70)
print(f"  S3/T   ~ a*(lnT + T^2/(lnT)^2)^3 + b")
print(f"           a={popt_s3c[0]:+.6e}  b={popt_s3c[1]:+.6e}  R^2={R2_s3c:.6f}")
print(f"  phi_esc ~ a*T*lnT + b*T^3/(lnT)^2 + c")
print(
    f"           a={popt_phic[0]:+.6e}  b={popt_phic[1]:+.6e}"
    f"  c={popt_phic[2]:+.6e}  R^2={R2_phic:.6f}"
)
print("=" * 70)

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# ── S3/T fit ──
ax = axes4[0, 0]
ax.plot(T, S3_T, "ko", markersize=5, label="data")
ax.plot(
    T_fine,
    model_S3T_combined(T_fine, *popt_s3c),
    "r-",
    lw=2,
    label=(
        r"$a\left(\ln T + \frac{T^2}{(\ln T)^2}\right)^{\!3} + b$"
        f"\na = {popt_s3c[0]:.4e}"
        f"\nb = {popt_s3c[1]:.4e}"
        f"\n$R^2 = {R2_s3c:.6f}$"
    ),
)
ax.set_xlabel("T  (GeV)")
ax.set_ylabel("S3 / T")
ax.set_title(r"$S_3/T$ — combined ansatz")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── S3/T residual ──
ax = axes4[0, 1]
res_s3c = (S3_T - yfit_s3c) / np.abs(S3_T) * 100
ax.plot(T, res_s3c, "s-", markersize=4, color="steelblue")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("T  (GeV)")
ax.set_ylabel("Residual  (%)")
ax.set_title(f"S3/T residual  (R²={R2_s3c:.5f})")
ax.grid(True, alpha=0.3)

# ── phi_esc fit ──
ax = axes4[1, 0]
ax.plot(T, phi_esc, "ko", markersize=5, label="data")
ax.plot(
    T_fine,
    model_phi_esc_combined(T_fine, *popt_phic),
    "r-",
    lw=2,
    label=(
        r"$a\,T\ln T + b\,\frac{T^3}{(\ln T)^2} + c$"
        f"\na = {popt_phic[0]:.4e}"
        f"\nb = {popt_phic[1]:.4e}"
        f"\nc = {popt_phic[2]:.4e}"
        f"\n$R^2 = {R2_phic:.6f}$"
    ),
)
ax.set_xlabel("T  (GeV)")
ax.set_ylabel("phi_esc")
ax.set_title(r"$\phi_{\mathrm{esc}}$ — combined ansatz")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── phi_esc residual ──
ax = axes4[1, 1]
res_phic = (phi_esc - yfit_phic) / np.abs(phi_esc) * 100
ax.plot(T, res_phic, "s-", markersize=4, color="steelblue")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("T  (GeV)")
ax.set_ylabel("Residual  (%)")
ax.set_title(f"phi_esc residual  (R²={R2_phic:.5f})")
ax.grid(True, alpha=0.3)

fig4.suptitle(f"Combined ansatze  –  fermion_only  ({dataset})", fontsize=13)
plt.tight_layout()
plt.savefig(f"figs/finiteTemp/fitting_combined_fermion_only_{dataset}.png", dpi=200)
plt.close(fig4)

print("\nPlots saved:")
print(f"  figs/finiteTemp/fitting_analysis_fermion_only_{dataset}.png")
print(f"  figs/finiteTemp/fitting_residuals_fermion_only_{dataset}.png")
print(f"  figs/finiteTemp/fitting_sqrtT_fermion_only_{dataset}.png")
print(f"  figs/finiteTemp/fitting_combined_fermion_only_{dataset}.png")
