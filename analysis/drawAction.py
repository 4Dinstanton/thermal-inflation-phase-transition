import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit


def format_e(n):
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def rev(x, a, b, c, d, e, f):
    # return a * x**2 + b * x + c
    return a + b * x + c * x**2 + d * x**3 + e * np.exp(-f * x)


def HubbleSquare(T, delV):
    MPL = 2.4e18
    chig2 = 30 / (math.pi**2 * 106.75)
    return (T**4 / chig2 + delV) / (3 * MPL**2)


def Hubble(T, delV):
    MPL = 2.4e18
    chig2 = 30 / (math.pi**2 * 106.75)
    if isinstance(T, np.ndarray):
        Hub2 = (T**4 / chig2 + delV) / (3 * MPL**2)
        return np.sqrt(Hub2.astype("float"))
    else:
        return np.sqrt((T**4 / chig2 + delV) / (3 * MPL**2))


lambdaSix = 0

param_set = "set8"
csv_path = f"data/tunneling/{param_set}"
MPL = 2.4e18


# df = pd.read_csv(f"{csv_path}/T-S_param_{param_set}_lambdaSix_{format_e(lambdaSix)}.csv")
potential_flag = "boson_only"
# potential_flag = "boson_fermion"
df = pd.read_csv(
    f"{csv_path}/T-S_param_{param_set}_lambdaSix_0E+00_{potential_flag}.csv"
).iloc[1:]
# df = pd.concat((df,df2)).iloc[:150].reset_index()
# print(df)
delV = 10**36 / 4
chig2 = 30 / (math.pi**2 * 106.75)
T_c = df["T"].max()
T_c = 15000
# print(df["T"].values.dtype)
# print(ho)

H2 = HubbleSquare(df["T"].values.astype(float), delV).astype("float")
# print(H2)
# H = Hubble(df["T"].values.astype(float), delV)
# delV = 10**24
# H = Hubble(df["T"].values, delV)
# print(H2)
# print(ho)
GAMMA = df["T"] ** 4 * (df["S3/T"] / (2 * math.pi)) ** (3 / 2) * np.exp(-df["S3/T"])
# print(np.log(1/H2**2))
G = (-df["S3/T"]) + 4 * np.log(df["T"]) + 3 * np.log(df["S3/T"] / (2 * math.pi)) / 2
G_H4 = (
    (-df["S3/T"])
    + 4 * np.log(df["T"])
    + 3 * np.log(df["S3/T"] / (2 * math.pi)) / 2
    - 2 * np.log(H2)
)
area = np.trapz(df["T"], np.exp(G_H4) / df["T"])
# print(ho)
# print(G_H4.values)
# print(ho)
t_arr = np.linspace(df["T"].min(), df["T"].max(), 400)
popt, pcov = curve_fit(rev, df["T"].values, G.values)
# print(popt)
# print(ho)
# print("gamma", G_H4)

# print("fitting", rev(t_arr, *popt)[0])
MPL = 2.4e18
chig2 = 30 / (math.pi**2 * 106.75)
H_ = Hubble(2_00_000, delV)
tp_arr = np.linspace(df["T"].min(), df["T"].max() + 5000, 400)
# print((np.exp(rev(tp_arr, *popt)) / (H_**4 * tp_arr))[:-50:])
# plt.plot(tp_arr, np.exp(rev(tp_arr, *popt)) / (H_**4))
# plt.plot(tp_arr, np.exp(rev(tp_arr, *popt)) / (H_**4) * (Tp / T - 1) ** 3)
# plt.yscale("log")
# plt.show()

# plt.plot(t_arr, Hubble(t_arr, delV))
# plt.show()
# print(ho)


def nT(T, *popt):
    # return np.trapz(T, np.exp(G_H4)/df["T"])
    # def f(y): return (a * y**4 + b * y**3 + c * y **2 + d * y + e)/y / ((y**8 / (900/(math.pi**2*106.75)**2) + delV)/(3*MPL**2)**2)
    def f(y):
        return np.exp(rev(y, *popt)) / (H_**4 * y)

    return quad(f, T, T_c)[0]


def inner_integral(T, Tp):
    return quad(lambda Tpp: 1 / Hubble(Tpp, delV), T, Tp)[0]


def percol(T, *popt):
    prefactor = 4 * math.pi / 3

    def outer_integrand(Tp):
        return np.exp(rev(Tp, *popt) - 4 * np.log(H_)) * (Tp / T - 1) ** 3 / T

    integral, err = quad(outer_integrand, T, T_c)
    # integral, err = quad(outer_integrand, T, 10000)

    return prefactor * integral


def I_of_T(T, Tc, v_b):
    prefactor = 4.0 * np.pi * v_b**3 / 3.0

    def outer_integrand(Tp):
        J = inner_integral(T, Tp)
        return Gamma(Tp) / (H(Tp) * Tp**4) * J**3

    integral, err = quad(outer_integrand, T, Tc)
    return prefactor * integral


# print(H2)
# plt.plot(df["T"], GAMMA)
# plt.show()


# print(nt_arr)
plt.figure(figsize=(10, 8))

# plt.plot(t_arr, (rev(t_arr, *popt)), linestyle='--', label=r'$\log(\Gamma(T)/H^4)$')
# print(np.exp(G_H4))
# print(G_H4)
# popt1, pcov1= curve_fit(rev, df["T"].values, G_H4)

# --- Step 1: coarse scan to locate approximate T_n (where Gamma/H^4 ~ 1) ---
t_coarse = np.linspace(df["T"].min(), df["T"].max() + 5000, 5000)
log_gamma_h4_coarse = rev(t_coarse, *popt) - 4 * np.log(Hubble(t_coarse, delV))
cross_idx = np.where(np.diff(np.sign(log_gamma_h4_coarse)))[0]
if len(cross_idx) > 0:
    T_n_approx = 0.5 * (t_coarse[cross_idx[-1]] + t_coarse[cross_idx[-1] + 1])
else:
    T_n_approx = 0.5 * (df["T"].min() + df["T"].max())
    print(
        f"  WARNING: Gamma/H^4=1 crossing not found, fallback T_n_approx={T_n_approx:.1f}"
    )
print(f"  Approximate T_n (Gamma/H^4=1): {T_n_approx:.2f} GeV")

# --- Step 2: fine grid around T_n_approx ---
HALF_WIDTH = 200.0
N_FINE = 500
t_arr = np.linspace(T_n_approx - HALF_WIDTH, T_n_approx + HALF_WIDTH, N_FINE)
nt_arr = np.empty(N_FINE)
perc_arr = np.empty(N_FINE)
for i, t in enumerate(t_arr):
    nt_arr[i] = nT(t, *popt)
    perc_arr[i] = percol(t, *popt)

# --- Step 3: determine plot boundaries from visible y-range ---
YLIM_LO, YLIM_HI = 1e-4, 1e4
gamma_h4_fine = np.exp(rev(t_arr, *popt)) / (Hubble(t_arr, delV) ** 4)
I_over_034 = perc_arr / 0.34

# Right boundary: first T where Gamma/H^4 drops below YLIM_LO
right_mask = np.where(gamma_h4_fine < YLIM_LO)[0]
idx_right = right_mask[0] if len(right_mask) > 0 else N_FINE - 1

# Left boundary: last T where I/0.34 exceeds YLIM_HI
left_mask = np.where(I_over_034 > YLIM_HI)[0]
idx_left = left_mask[-1] if len(left_mask) > 0 else 0

# Slight margin (~5 % of visible span, at least 3 points)
margin = max(3, int(0.05 * (idx_right - idx_left)))
idx_left = max(0, idx_left - margin)
idx_right = min(N_FINE - 1, idx_right + margin)

t_arr = t_arr[idx_left : idx_right + 1]
perc_arr = perc_arr[idx_left : idx_right + 1]
nt_arr = nt_arr[idx_left : idx_right + 1]
f_arr = rev(t_arr, *popt)
print(f"  Plot range: [{t_arr[0]:.2f}, {t_arr[-1]:.2f}] GeV  ({len(t_arr)} points)")
line1 = plt.plot(t_arr / 1000, nt_arr, color="red", label=r"$n(T)$")
plt.axhline(1, 0, 1, linestyle="--", color="black")
T_n = t_arr[np.argmin(abs(np.array(nt_arr) - 1))] / 1000
# print("T_n", T_n)
plt.axvline(T_n, 0, 1, color="red", linestyle="--")
plt.text(T_n + 0.001, 10, r"$T_n$", color="red", fontsize=12)

# print("max nt???", max(nt_arr))
# plt.yticks(np.geomspace(0.01, 10000 ,15).round())
plt.xlim([np.min(t_arr / 1000), np.max(t_arr / 1000)])
plt.ylim([0.0001, 10000])
plt.yscale("log")
line2 = plt.plot(
    t_arr / 1000,
    np.exp(rev(t_arr, *popt)) / (Hubble(t_arr, delV) ** 4),
    label=r"$\dfrac{\Gamma(T)}{H^4}$",
)
# plt.ylabel("number of bubble in a Hubble volume")
# plt.legend()
# plt.twinx()
plt.xlabel("T (TeV)")
plt.ylim([0.0001, 10000])
plt.yscale("log")

# plt.twinx()
# line2 = plt.plot(df["T"].iloc[:20], G_H4[:20], label=r"$\log(\Gamma(T)/H^4)$")
line3 = plt.plot(
    t_arr / 1000, np.array(perc_arr) / 0.34, label=r"$\dfrac{I}{0.34}$", color="green"
)
line4 = plt.plot(
    t_arr / 1000,
    np.exp(-np.array(perc_arr)) / 0.7,
    label=r"$\dfrac{P}{0.7}$",
    color="orange",
)
T_p = t_arr[np.argmin(abs(np.exp(-np.array(perc_arr)) / 0.7 - 1))] / 1000
print("T_p", T_p)
plt.axvline(T_p, 0, 1, color="orange", linestyle="--")
plt.text(T_p + 0.001, 10, r"$T_p$", color="orange", fontsize=12)

T_c1 = t_arr[np.argmin(abs(np.exp(-np.array(perc_arr)) / 0.7 - 10**-5))] / 1000
print("T_c1", T_c1)
plt.axvline(T_c1, 0, 1, color="black", linestyle="--")
plt.text(T_c1 - 0.003, 10, r"$T_{c_1}$", color="black", fontsize=12)
# print(rev(t_arr, *popt))
# print(rev(t_arr, *popt))
# line2 = plt.plot(df["T"].iloc[:75], G_H4[:75], label=r"$\log(\Gamma(T)/H^4)$")
# plt.ylabel(r"$\log(\Gamma(T))$")

lines = line1 + line2 + line3 + line4
labels = [line.get_label() for line in lines]
plt.legend(lines, labels)
# plt.title(r"$m_0 = $")
# plt.title(r"$\log(\Gamma(T))$")
plt.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_{potential_flag}.png", dpi=300)
plt.show()
# print(df)


t_arr = np.linspace(df["T"].min(), 200000, 1000)[:]
plt.plot(t_arr / 1000, np.exp(rev(t_arr, *popt)), label=r"$\Gamma$")
plt.xlabel("T (TeV)")
plt.ylabel(r"$\Gamma$")
plt.savefig(
    f"figs/finiteTemp/{param_set}_V0_{delV}_{potential_flag}_Gamma.png", dpi=300
)
# plt.show()
print("T_n", T_n, "T_p", T_p)
