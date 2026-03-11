import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
from cosmoTransitions.tunneling1D import SingleFieldInstanton

phi0 = 1e11
m = 1e3
lam = 1.09
g = 1.05

V0 = m ** 2 * phi0 ** 2 / 3
a = m ** 2 / (6 * phi0 ** 4)

def Jb(y2): return quad(lambda x: x ** 2 * np.log(1 - np.exp(- np.sqrt(x ** 2 + y2))), 0, np.inf)[0]
def Jf(y2): return - quad(lambda x: x ** 2 * np.log(1 + np.exp(- np.sqrt(x ** 2 + y2))), 0, np.inf)[0]

y2 = np.logspace(-1, 14, num = 100)
Jb_spl = make_interp_spline(y2, np.vectorize(Jb)(y2))
Jf_spl = make_interp_spline(y2, np.vectorize(Jf)(y2))

def mb2(phi, T): return m ** 2 + 1 / 2 * lam ** 2 * phi ** 2 + (1 / 4 * lam ** 2 + 2 / 3 * g ** 2) * T ** 2
def mf2(phi, T): return 1 / 2 * lam ** 2 * phi ** 2 + 1 / 6 * g ** 2 * T ** 2

def V1(phi): return V0 - 1 / 2 * m ** 2 * phi ** 2 + a * phi ** 6
def VT(phi, T): return T ** 4 / (2 * np.pi ** 2) * (2 * Jb_spl(mb2(phi, T) / T ** 2) + Jf_spl(mf2(phi, T) / T ** 2))
def V(phi, T): return V1(phi) + VT(phi, T)

def S3(T):
    inst = SingleFieldInstanton(1_000_000_00, 0, lambda phi: V(phi, T))
    prof = inst.findProfile()
    return inst.findAction(prof)

T = np.linspace(1e4, 1e6, num = 100)
Gam = T ** 4 * np.exp(- np.vectorize(S3)(T) / T)

plt.plot(T, Gam, '.')
plt.xlabel('T')
plt.ylabel(r'$\Gamma$')
plt.show()