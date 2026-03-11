import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# from scipy.differentiate import derivative


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


def drev(x, a, b, c, d, e, f):
    return b + 2 * c * x + 3 * d * x**2 - e * f * np.exp(-f * x)


def S3_small(x, a, b):
    return x / (a * 1.05**2) * (np.sqrt((a * x - b * 1000) / 2 * 1000))


def S3_smaller(x, a, b, c):
    return np.sqrt((a * x - b * 1000) / 2 * 1000)


def S3_large_quart(x, a):
    return x**4 / (a * 1000)


def S3_large_tri(x, a):
    return x**3 / (a * 1000)


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

param_set = "set6"
csv_path = f"data/tunneling/{param_set}"
MPL = 2.4e18

# "V_correct" = boson + fermion, "fermion_only" = fermion only
potential_flag = "boson_fermion"

# df = pd.read_csv(f"{csv_path}/T-S_param_{param_set}_lambdaSix_{format_e(lambdaSix)}.csv")
df = pd.read_csv(
    f"{csv_path}/T-S_param_{param_set}_lambdaSix_0E+00_{potential_flag}.csv"
)
# df = pd.concat((df,df2)).iloc[:150].reset_index()
print(df)
delV = 10**28
chig2 = 30 / (math.pi**2 * 106.75)
# print(df["T"].values.dtype)
# print(ho)

H2 = HubbleSquare(df["T"].values.astype(float), delV).astype("float")
print(H2)
# H = Hubble(df["T"].values.astype(float), delV)
# delV = 10**24
# H = Hubble(df["T"].values, delV)
# print(H2)
# print(ho)
GAMMA = df["T"] ** 4 * (df["S3/T"] / (2 * math.pi)) ** (3 / 2) * np.exp(-df["S3/T"])
# print(np.log(1/H2**2))
G = (-df["S3/T"]) + 4 * np.log(df["T"]) + 3 * np.log(df["S3/T"] / (2 * math.pi)) / 2
dG = np.diff(G, 1)

S3 = df["S3/T"] * df["T"]
# plt.plot(df["T"], df["S3/T"]*df["T"])
t_arr = np.linspace(df["T"].min(), 200000, 1000)[:]
cs = CubicSpline(df["T"].values, df["S3/T"].values * df["T"].values)
cs_1 = CubicSpline(df["T"].values, df["S3/T"].values)
# plt.plot(t_arr[:50], quartic(t_arr[:50], *po),linestyle="--")
# plt.show()
# print(ho)
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
print(popt)
# print(ho)
# print("gamma", G_H4)

print("fitting", rev(t_arr, *popt)[0])
# print(ho)


def nT(T, *popt):
    # return np.trapz(T, np.exp(G_H4)/df["T"])
    # def f(y): return (a * y**4 + b * y**3 + c * y **2 + d * y + e)/y / ((y**8 / (900/(math.pi**2*106.75)**2) + delV)/(3*MPL**2)**2)
    def f(y):
        return np.exp(rev(y, *popt)) / (Hubble(T, delV) ** 4 * y)

    return quad(f, T, 23000)[0]


def inner_integral(T, Tp):
    return quad(lambda T: 1 / Hubble(T, delV), T, Tp)[0]


def percol(T, *popt):
    prefactor = 4 * math.pi / 3

    def outer_integrand(Tp):
        J = inner_integral(T, Tp)
        return np.exp(rev(Tp, *popt)) / (Hubble(Tp, delV) * Tp**4) * J**3

    integral, err = quad(outer_integrand, T, 25000)

    return prefactor * integral

    def f(y):
        return (
            4
            * math.pi
            / 3
            * np.exp(rev(y, *popt))
            / (Hubble(T, delV) * y**4)
            * (quad(lambda T: 1 / Hubble(T, delV), y)[0] ** 3)
        )

    integral, err = quad(outer_integrand, T, Tc)
    return prefactor * integral
    return quad(f, T, 23000)[0]


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

# plt.plot(t_arr, np.log(quartic(t_arr, *popt))-np.log(HubbleSquare(t_arr,delV)**2), color="red", label="fitted", linestyle="--")
# print(quartic(t_arr, *popt))
t_arr = np.linspace(df["T"].min(), 7450, 500)[:]
# print(t_arr)
# print(ho)
nt_arr = []
perc_arr = []
for t in t_arr:
    nt_arr.append(nT(t, *popt))
    perc_arr.append(percol(t, *popt) / 0.34)

# print(nt_arr)
# print(np.min(abs(np.array(nt_arr) - 1)))
# print(t_arr[np.argmin(abs(np.array(nt_arr) - 1))])
print("????", t_arr[np.argmin(abs(np.array(nt_arr) - 1))])
print("????", nt_arr[np.argmin(abs(np.array(nt_arr) - 1))])
# print(ho)
neg_shift = 190
end_shift = 50
t_arr = t_arr[np.argmin(abs(np.array(nt_arr) - 1)) - neg_shift :]
perc_arr = perc_arr[np.argmin(abs(np.array(nt_arr) - 1)) - neg_shift :]
nt_arr = nt_arr[np.argmin(abs(np.array(nt_arr) - 1)) - neg_shift :]
f_arr = rev(t_arr, *popt)
print("", t_arr[np.argmin(abs(f_arr))])
print("2", nt_arr[np.argmin(abs(f_arr))])
print(perc_arr)
# print(ho)
"""
line1 = plt.plot(t_arr/1000, nt_arr, color='red', label=r"$n(T)$")
plt.axhline(1,0,1,linestyle='--', color='black')
#print("max nt???", max(nt_arr))
#plt.yticks(np.geomspace(0.01, 10000 ,15).round())
plt.xlim([np.min(t_arr/1000), np.max(t_arr/1000)])
plt.ylim([0.0001, 10000])
plt.yscale("log")
line2 = plt.plot(t_arr/1000, np.exp(rev(t_arr, *popt))/(Hubble(t_arr, delV)**4), label=r'$\dfrac{\Gamma(T)}{H^4}$')
#plt.ylabel("number of bubble in a Hubble volume")
#plt.legend()
#plt.twinx()
plt.xlabel("T (TeV)")
plt.ylim([0.0001, 10000])
plt.yscale("log")

#plt.twinx()
#line2 = plt.plot(df["T"].iloc[:20], G_H4[:20], label=r"$\log(\Gamma(T)/H^4)$")
line3 = plt.plot(t_arr/1000, np.array(perc_arr)/0.34, label=r'$\dfrac{I}{0.34}$', color='green')
line4 = plt.plot(t_arr/1000, np.exp(-np.array(perc_arr))/0.7, label=r'$\dfrac{P}{0.7}$', color='orange')
print(rev(t_arr, *popt))
#line2 = plt.plot(df["T"].iloc[:75], G_H4[:75], label=r"$\log(\Gamma(T)/H^4)$")
#plt.ylabel(r"$\log(\Gamma(T))$")

lines = line1 + line2 + line3 + line4
labels = [line.get_label() for line in lines]
plt.legend(lines, labels)
#plt.title(r"$m_0 = $")
#plt.title(r"$\log(\Gamma(T))$")
#plt.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}.png",dpi=300)
plt.show()
print(df)

"""

po, pc = curve_fit(quartic, df["T"].values, df["S3/T"].values * df["T"].values)
print(*popt)
# print(ho)
t_arr = np.linspace(df["T"].min(), 200000, 1000)[:]
H = Hubble(t_arr, delV)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].plot(t_arr / 1000, (cs(t_arr)), label=r"$S_3$")
ax[0].plot(
    t_arr / 1000,
    4 * t_arr * np.log(t_arr / H),
    label=r"$(S_3)_c$",
    color="red",
    linestyle="--",
)
ax[0].set_xlabel("T (TeV)")
ax[0].set_ylabel(r"$S_3$")
ax[0].legend()
ax[0].set_title(r"$S_3$")
# plt.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_S_3_full.png", dpi=300)
# plt.show()

ax[1].plot(t_arr / 1000, (rev(t_arr, *popt)), label=r"$\log(\Gamma)$")
ax[1].set_xlabel("T (TeV)")
ax[1].set_ylabel(r"$\log(\Gamma)$")
ax[1].legend()
ax[1].set_title(r"$\log(\Gamma)$")
# plt.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_Gamma_full.png", dpi=300)
# plt.show()

ax[2].plot(
    t_arr / 1000,
    -1 * t_arr * (drev(t_arr, *popt)),
    label=r"$-\dfrac{d\log(\Gamma)}{d\log T}$",
)
ax[2].set_xlabel("T (TeV)")
ax[2].set_ylabel(r"$\beta/H$")
ax[2].legend()
ax[2].set_title(r"$\beta/H$")
fig.savefig(
    f"figs/finiteTemp/{param_set}_V0_{delV}_tunneling_full_temp_{potential_flag}.png",
    dpi=300,
)
# plt.show()

plt.cla()
fl, al = plt.subplots(1, 2, figsize=(15, 7))
df_large = df[(df["T"] > 5000)]
sopt, popc = curve_fit(
    S3_large_tri, df_large["T"].values, df_large["S3/T"].values * df_large["T"].values
)
sopq, popq = curve_fit(
    S3_large_quart, df_large["T"].values, df_large["S3/T"].values * df_large["T"].values
)
al[0].plot(t_arr / 1000, (cs(t_arr)), label=r"$S_3$")
al[0].plot(
    t_arr[300:] / 1000,
    S3_large_tri(t_arr[300:], *sopt),
    linestyle="--",
    color="red",
    label=rf"$\dfrac{{T^3}}{{{np.round(sopt[0], 3)} m}}$",
)
al[0].plot(
    t_arr[300:] / 1000,
    S3_large_quart(t_arr[300:], *sopq),
    linestyle="--",
    color="orange",
    label=rf"$\dfrac{{T^4}}{{{np.round(sopq[0], 3)} m}}$",
)
# plt.plot(t_arr/1000, 4 * t_arr *  np.log(t_arr/H), label=r'$(S_3)_c$', color='red', linestyle='--')
al[0].set_xlabel("T (TeV)")
al[0].set_ylabel(r"$S_3$")
al[0].legend()
al[0].set_title(
    r"$\text{Numeric fitting of } S_3 \text { in the region of } \alpha T \gg m $"
)
# fl.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_S3_fit_full_temp.png", dpi=300)

T_up = 7700
t_arr = np.linspace(df["T"].min(), T_up, 1000)[:]
df_small = df[(df["T"] < T_up)]
sops, pops = curve_fit(
    S3_small, df_small["T"].values, df_small["S3/T"].values * df_small["T"].values
)
print(sops)
alp, gam = sops
alp = np.round(alp, 3)
gam = np.round(gam, 3)
al[1].plot(t_arr / 1000, (cs(t_arr)), label=r"$S_3$")
al[1].plot(
    t_arr / 1000,
    S3_small(t_arr, *sops),
    linestyle="--",
    color="red",
    label=rf"$\dfrac{{T}}{{{alp}m^2}}\sqrt{{(\dfrac{{{alp}T-{gam}m}}{{2m}})}}$",
)
# plt.plot(t_arr/1000, 4 * t_arr *  np.log(t_arr/H), label=r'$(S_3)_c$', color='red', linestyle='--')
al[1].set_xlabel("T (TeV)")
al[1].set_ylabel(r"$S_3$")
al[1].legend()
al[1].set_title(
    r"$\text{Numeric fitting of } S_3 \text { in the region of } \alpha T \sim m $"
)
fl.savefig(
    f"figs/finiteTemp/{param_set}_V0_{delV}_S3_fit_{potential_flag}.png", dpi=300
)


fl2, al2 = plt.subplots(1, 2, figsize=(15, 5))
T_up = 5100
t_arr = np.linspace(df["T"].min(), T_up, 1000)[:]
df_small = df[(df["T"] < T_up)]
sops, pops = curve_fit(
    S3_small, df_small["T"].values, df_small["S3/T"].values * df_small["T"].values
)
print(sops)
alp, gam = sops
alp = np.round(alp, 3)
gam = np.round(gam, 3)
al2[0].plot(t_arr / 1000, (cs_1(t_arr)), label=r"$S_3/T$")
# al2[0].plot(t_arr/1000, S3_small(t_arr, *sops), linestyle='--', color='red', label=rf"$\dfrac{{T}}{{{alp}m^2}}\sqrt{{(\dfrac{{{alp}T-{gam}m}}{{2m}})}}$")
# plt.plot(t_arr/1000, 4 * t_arr *  np.log(t_arr/H), label=r'$(S_3)_c$', color='red', linestyle='--')
al2[0].set_xlabel("T (TeV)")
al2[0].set_ylabel(r"$S_3/T$")
al2[0].legend()
al2[0].set_title(r"$\text{Numeric fitting of } S_3/T$")


# plt.show()

# plt.cla()
t_arr = np.linspace(df["T"].min(), 5000, 1000)[:]
H = Hubble(t_arr, delV)

fig1, ax1 = plt.subplots(1, 3, figsize=(12, 5))
ax1[0].plot(t_arr / 1000, (cs(t_arr)), label=r"$S_3$")
ax1[0].plot(
    t_arr / 1000,
    4 * t_arr * np.log(t_arr / H),
    label=r"$(S_3)_c$",
    color="red",
    linestyle="--",
)
ax1[0].set_xlabel("T (TeV)")
ax1[0].set_ylabel(r"$S_3$")
ax1[0].legend()
ax1[0].set_title(r"$S_3$")
# plt.savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_S_3_small.png", dpi=300)
# plt.show()

# print(pd.DataFrame({"temp":t_arr, "S_3/T":cs(t_arr)/t_arr}).iloc[10:25])
# print(t_arr)
# print(cs(t_arr)/t_arr)

ax1[1].plot(t_arr / 1000, (rev(t_arr, *popt)), label=r"$\log(\Gamma)$")
ax1[1].set_xlabel("T (TeV)")
ax1[1].set_ylabel(r"$\log(\Gamma)$")
ax1[1].legend()
ax1[1].set_title(r"$\log(\Gamma)$")
# ax1[1].savefig(f"figs/finiteTemp/{param_set}_V0_{delV}_Gamma_small.png", dpi=300)
# plt.show()


ax1[2].plot(
    t_arr / 1000,
    -1 * t_arr * drev(t_arr, *popt),
    label=r"$-\dfrac{d\log(\Gamma)}{d\log T}$",
)
ax1[2].set_xlabel("T (TeV)")
ax1[2].set_ylabel(r"$\beta/H$")
ax1[2].legend()
ax1[2].set_title(r"$\beta/H$")
fig1.savefig(
    f"figs/finiteTemp/{param_set}_V0_{delV}_tunneling_small_temp_{potential_flag}.png",
    dpi=300,
)
# plt.show()


# plt.cla()
df = pd.read_csv(f"{csv_path}/T-S_param_set6_lambdaSix_0E+00_with_small11.csv").iloc[
    ::3
]
df["T"].iloc[0] = 7345.1
# df


cs = CubicSpline(df["T"].values, df["S3/T"].values * df["T"].values)
cs_1 = CubicSpline(df["T"].values, df["S3/T"].values)

T_up = 7345.6
t_arr = np.linspace(df["T"].min(), T_up, 100)[:]
print(t_arr)
df_small = df[(df["T"] < T_up)]
print(df_small)
# plt.plot(df_small["T"].values, df_small["S3/T"].values* df_small["T"].values)
# plt.show()
sops, pops = curve_fit(
    S3_smaller,
    df_small["T"].values,
    df_small["S3/T"].values,
    maxfev=100000,
)

print(sops)
alp, gam, c = sops
alp = np.round(alp, 3)
gam = np.round(gam, 3)
# al2[1].plot(t_arr/1000, (cs_1(t_arr)), label=r'$S_3$')
# print(df)
# print(df["T"].values)
al2[1].scatter(df_small["T"].values / 1000, df_small["S3/T"].values)
al2[1].plot(
    t_arr / 1000,
    S3_smaller(t_arr, *sops) / 1.21,
    linestyle="--",
    color="red",
    label=rf"0.83 \times $\sqrt{{(\dfrac{{{alp}T-{gam}m}}{{2m}})}}$",
)
# plt.plot(t_arr/1000, 4 * t_arr *  np.log(t_arr/H), label=r'$(S_3)_c$', color='red', linestyle='--')
al2[1].set_xlabel("T (TeV)")
al2[1].set_ylabel(r"$S_3/T$")
al2[1].legend()
# al[1].set_title(r"$\text{Numeric fitting of } S_3 \text { in the region of } \alpha T \sim m $")
fl2.savefig(
    f"figs/finiteTemp/{param_set}_V0_{delV}_S3_fit_smaller_{potential_flag}.png",
    dpi=300,
)
