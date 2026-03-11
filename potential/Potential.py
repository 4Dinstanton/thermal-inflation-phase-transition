import sympy as sp
import numpy as np
import scipy
from scipy import integrate, interpolate
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import math


class zeroTemperaturePotential:
    def __init__(self, param_dict):
        self.lam = param_dict["lambda"]
        self.mphi = param_dict["mphi"]
        self.epsilon = param_dict["epsilon"]
        self.lambdaSix = param_dict["lambdaSix"]

    def V_tree(self, X):
        phi = X[..., 0]
        # return self.mphi**2 / 2 * phi**2 - self.lam * phi**4 - self.epsilon * phi**3 + self.lambdaSix * phi**6
        return self.lam / 4 * phi**4 - self.mphi**2 / 2 * phi**2

    @property
    def v(self):
        return np.sqrt(np.abs(self.mphi**2 / self.lam))
        # return np.sqrt(self.mphi**2 / (4 * self.lambdaSix))

    """
    def B(self):
        m2 = abs(self.lam/2 * self.v**2)
        mu = abs(self.epsilon / (2*self.v**3))
        e = 1 - (2 * self.lam/4 * m2) / mu**2
        return m2 * np.pi**2 / (12 * mu**2) * (1 + 10.07*e + 16.55*e**2) / e**3
    """


class finiteTemperaturePotential(zeroTemperaturePotential):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.bosonMassSquared = param_dict["bosonMassSquared"]
        self.bosonCoupling = param_dict["bosonCoupling"]
        self.bosonGaugeCoupling = param_dict["bosonGaugeCoupling"]

        self.fermionCoupling = param_dict["fermionCoupling"]
        self.fermionGaugeCoupling = param_dict["fermionGaugeCoupling"]

    def update_params(self, param_dict):
        self.bosonMassSquared = param_dict["bosonMassSquared"]
        self.bosonCoupling = param_dict["bosonCoupling"]
        self.bosonGaugeCoupling = param_dict["bosonGaugeCoupling"]

        self.fermionCoupling = param_dict["fermionCoupling"]
        self.fermionGaugeCoupling = param_dict["fermionGaugeCoupling"]

    def update_T(self, T):
        self.T = T

    def build_fast_thermal(self, x_max=120.0, n_pts=2048):
        """Pre-build CubicSpline interpolants for Jb, Jf, dJb, dJf.

        Replaces O(ms) numerical quadrature per call with O(ns) spline lookup,
        giving ~100-1000x speedup during tunneling profile iteration.
        """
        import time as _t
        _t0 = _t.time()
        x_grid = np.linspace(0.0, x_max, n_pts)
        Jb_vals = np.array([CTFT.Jb_exact(x) for x in x_grid])
        Jf_vals = np.array([CTFT.Jf_exact(x) for x in x_grid])
        dJb_vals = np.array([CTFT.dJb_exact(x) for x in x_grid])
        dJf_vals = np.array([CTFT.dJf_exact(x) for x in x_grid])
        self._Jb_spl = CubicSpline(x_grid, Jb_vals, bc_type="not-a-knot")
        self._Jf_spl = CubicSpline(x_grid, Jf_vals, bc_type="not-a-knot")
        self._dJb_spl = CubicSpline(x_grid, dJb_vals, bc_type="not-a-knot")
        self._dJf_spl = CubicSpline(x_grid, dJf_vals, bc_type="not-a-knot")
        self._fast_xmax = x_max
        self._fast_thermal = True
        self._fast_arrays = (x_grid, Jb_vals, Jf_vals, dJb_vals, dJf_vals)
        print(f"[fast_thermal] Splines built in {_t.time()-_t0:.2f}s "
              f"({n_pts} pts, x_max={x_max})")

    def set_fast_thermal_from_arrays(self, x_grid, Jb_vals, Jf_vals, dJb_vals, dJf_vals):
        """Reconstruct splines from pre-computed arrays (fast, no quadrature)."""
        self._Jb_spl = CubicSpline(x_grid, Jb_vals, bc_type="not-a-knot")
        self._Jf_spl = CubicSpline(x_grid, Jf_vals, bc_type="not-a-knot")
        self._dJb_spl = CubicSpline(x_grid, dJb_vals, bc_type="not-a-knot")
        self._dJf_spl = CubicSpline(x_grid, dJf_vals, bc_type="not-a-knot")
        self._fast_xmax = float(x_grid[-1])
        self._fast_thermal = True

    def _Jb_fast(self, x):
        x = np.asarray(x)
        return np.where(x <= self._fast_xmax, self._Jb_spl(np.clip(x, 0, self._fast_xmax)), 0.0)

    def _Jf_fast(self, x):
        x = np.asarray(x)
        return np.where(x <= self._fast_xmax, self._Jf_spl(np.clip(x, 0, self._fast_xmax)), 0.0)

    def _dJb_fast(self, x):
        x = np.asarray(x)
        return np.where(x <= self._fast_xmax, self._dJb_spl(np.clip(x, 0, self._fast_xmax)), 0.0)

    def _dJf_fast(self, x):
        x = np.asarray(x)
        return np.where(x <= self._fast_xmax, self._dJf_spl(np.clip(x, 0, self._fast_xmax)), 0.0)

    def bosonic_input(self, x):
        return (
            np.sqrt(
                (
                    self.bosonMassSquared
                    + 0.5 * self.bosonCoupling**2 * x**2
                    + (
                        0.25 * self.bosonCoupling**2
                        + 2 / 3 * self.bosonGaugeCoupling**2
                    )
                    * self.T**2
                )
            )
            / self.T
        )

    def fermionic_input(self, x):
        return (
            np.sqrt(
                0.5 * self.fermionCoupling**2 * x**2
                + 1 / 6 * self.fermionGaugeCoupling**2 * self.T**2
            )
            / self.T
        )

    def mphi2(self, X):
        phi = X[..., 0]
        # return (3 * self.lam * phi**2) - 6 * self.epsilon * phi + 30 * self.lambdaSix * phi**4
        return 3 * self.lam * phi**2
        # return self.mphi**2 + 12 - self.lam * phi**2 + 30 * self.lambdaSix * phi**4
        # return 30 * self.lambdaSix * phi**4 - 24*self.lambdaSix * self.v**2 * phi**3 + 4 * self.lambdaSix * self.v**2

    def V_T(self, X):
        return (self.T**4 / 2 * np.pi**2) * CTFT.Jb_exact2(self.mphi2(X) / self.T)

    def V_p(self, X):
        phi = X[..., 0]
        treeLevelPotential = self.V_tree(X)
        _Jb = self._Jb_fast if getattr(self, '_fast_thermal', False) else CTFT.Jb_exact
        _Jf = self._Jf_fast if getattr(self, '_fast_thermal', False) else CTFT.Jf_exact
        thermalCorrectionBosonPotential = _Jb(self.bosonic_input(phi))
        thermalCorrectionFermionPotential = _Jf(self.fermionic_input(phi))
        correctedPotential = treeLevelPotential + self.T**4 / (2 * math.pi**2) * (
            2 * thermalCorrectionBosonPotential - thermalCorrectionFermionPotential
        )
        """
        print(np.min(self.T**4 / \
            (2 * math.pi**2) * (2*thermalCorrectionBosonPotential - thermalCorrectionFermionPotential)))
        plt.plot(phi, self.T**4 / \
            (2 * math.pi**2) * (2*thermalCorrectionBosonPotential - thermalCorrectionFermionPotential))
        plt.plot(phi, correctedPotential,color='orange')
        plt.twinx()
        plt.plot(phi, treeLevelPotential,color='red')

        plt.show()
        print(np.max(treeLevelPotential))
        """
        return correctedPotential

    def V_p_correct(self, X):
        phi = X[..., 0]
        treeLevelPotential = self.V_tree(X)
        _Jb = self._Jb_fast if getattr(self, '_fast_thermal', False) else CTFT.Jb_exact
        _Jf = self._Jf_fast if getattr(self, '_fast_thermal', False) else CTFT.Jf_exact
        thermalCorrectionBosonPotential = _Jb(self.bosonic_input(phi))
        thermalCorrectionFermionPotential = _Jf(self.fermionic_input(phi))
        correctedPotential = treeLevelPotential + self.T**4 / (2 * math.pi**2) * (
            thermalCorrectionBosonPotential + thermalCorrectionFermionPotential
        )
        return correctedPotential

    def V_p_fermion_only(self, X):
        phi = X[..., 0]
        treeLevelPotential = self.V_tree(X)
        _Jf = self._Jf_fast if getattr(self, '_fast_thermal', False) else CTFT.Jf_exact
        thermalCorrectionFermionPotential = _Jf(self.fermionic_input(phi))
        return treeLevelPotential + self.T**4 / (2 * math.pi**2) * (
            thermalCorrectionFermionPotential
        )

    # def V(self, X):
    #    return self.V_tree(X) + self.V_T(X) + math.pi**2/30 * 100 * self.T**4 + self.mphi2(X)**2 / (64 * math.pi**2) * (np.log(self.mphi2(X) / self.T) - 3/2)

    def V(self, X):
        return (
            self.V_p(X)
            + math.pi**2 / 30 * 100 * self.T**4
            + self.mphi2(X) ** 2
            / (64 * math.pi**2)
            * (np.log(np.abs(self.mphi2(X)) / self.T) - 3 / 2)
        )

    def V_correct(self, X):
        return (
            self.V_p_correct(X)
            + math.pi**2 / 30 * 100 * self.T**4
            + self.mphi2(X) ** 2
            / (64 * math.pi**2)
            * (np.log(np.abs(self.mphi2(X)) / self.T) - 3 / 2)
        )

    def V_fermion_only(self, X):
        return (
            self.V_p_fermion_only(X)
            + math.pi**2 / 30 * 100 * self.T**4
            + self.mphi2(X) ** 2
            / (64 * math.pi**2)
            * (np.log(np.abs(self.mphi2(X)) / self.T) - 3 / 2)
        )

    def dVdphi(self, X):
        phi = X[..., 0]
        return self.lam * phi**3 - self.mphi**2 * phi
        # return self.lam * phi * (phi**2 - self.v**2)

    def dV_p(self, X):
        phi = X[..., 0]
        dVdphi = self.dVdphi(phi)
        _dJb = self._dJb_fast if getattr(self, '_fast_thermal', False) else CTFT.dJb_exact
        _dJf = self._dJf_fast if getattr(self, '_fast_thermal', False) else CTFT.dJf_exact
        thermalCorrectionBosonPotentialDerivative = _dJb(
            self.bosonic_input(phi)
        ) * self._dxdphi_boson(phi)
        thermalCorrectionFermionPotentialDerivative = _dJf(
            self.fermionic_input(phi)
        ) * self._dxdphi_fermion(phi)

        correctedPotentialDerivative = dVdphi + self.T**4 / (2 * math.pi**2) * (
            2 * thermalCorrectionBosonPotentialDerivative
            - thermalCorrectionFermionPotentialDerivative
        )
        return correctedPotentialDerivative

    def dV_p_correct(self, X):
        phi = X[..., 0]
        dVdphi = self.dVdphi(phi)
        _dJb = self._dJb_fast if getattr(self, '_fast_thermal', False) else CTFT.dJb_exact
        _dJf = self._dJf_fast if getattr(self, '_fast_thermal', False) else CTFT.dJf_exact
        thermalCorrectionBosonPotentialDerivative = _dJb(
            self.bosonic_input(phi)
        ) * self._dxdphi_boson(phi)
        thermalCorrectionFermionPotentialDerivative = _dJf(
            self.fermionic_input(phi)
        ) * self._dxdphi_fermion(phi)

        correctedPotentialDerivative = dVdphi + self.T**4 / (2 * math.pi**2) * (
            thermalCorrectionBosonPotentialDerivative
            + thermalCorrectionFermionPotentialDerivative
        )
        return correctedPotentialDerivative

    def dV_p_fermion_only(self, X):
        phi = X[..., 0]
        dVdphi = self.dVdphi(phi)
        _dJf = self._dJf_fast if getattr(self, '_fast_thermal', False) else CTFT.dJf_exact
        thermalCorrectionFermionPotentialDerivative = _dJf(
            self.fermionic_input(phi)
        ) * self._dxdphi_fermion(phi)

        return dVdphi + self.T**4 / (2 * math.pi**2) * (
            thermalCorrectionFermionPotentialDerivative
        )

    def _dxdphi_boson(self, x):
        return 0.5 * self.bosonCoupling**2 / self.T**2 * x

    def _dxdphi_fermion(self, x):
        return 0.5 * self.fermionCoupling**2 / self.T**2 * x

    def _dJbdphi_exact(self, x):
        def sqrt_part(y):
            return self.sqrt(
                y * y
                + (self.bosonMassSquared + 0.5 * self.bosonCoupling**2 * x * x)
                / self.T**2
                + (0.25 * self.bosonCoupling**2 + 2 / 3 * bosonGaugeCoupling**2)
            )

        def f(y):
            return (
                y
                * y
                * (1 / (1 - np.exp(-1 * sqrt_part(y))))
                * np.exp(-1 * sqrt_part(y))
                * (self.bosonCoupling / self.T) ** 2
                * x
                / sqrt_part(y)
            )

        if x.imag == 0:
            x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]

    def _dJfdphi_exact(self, x):
        def sqrt_part(y):
            return self.sqrt(
                y * y
                + 0.5 * self.FermionCoupling**2 * x**2 / self.T**2
                + 1 / 6 * self.fermionGaugeCoupling**2
            )

        def f(y):
            return (
                -y
                * y
                * (1 / (1 + np.exp(-1 * sqrt_part(y))))
                * -1
                * np.exp(-1 * sqrt_part(y))
                * ((self.fermionCoupling / self.T) ** 2 * x / sqrt_part(y))
            )

        if x.imag == 0:
            x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]

    # def mphi2(self, X):
    #    phi = X[..., 0]
    #    return (3 * self.lam * phi**2)

    def _dm2dphi_boson(self, X):
        phi = X[..., 0]
        return 6 * self.lam * phi

    def _dJb_exact2(x):
        f = lambda y: np.sqrt(
            (y * y * (np.exp(np.sqrt(y * y + x)) - 1) ** -1 / 2 * np.sqrt(y * y + x))
        )
        return integrate.quad(f, max(0, np.sqrt(-x)), np.inf)[0]

    def _dJb_exact2_scalar(self, x):
        def integrand(y):
            # if y*y+x < 0:
            #    print(y*y, x)
            z = np.sqrt(y * y + x)
            # if np.isnan(z):
            #    print("z?????????", z)

            # if np.real(z) > 50:
            #    return np.real(y*y / (z * np.exp(z)))  # since exp(z)-1 ~ exp(z)

            # handle huge z (avoid overflow in exp)

            denom = np.exp(z) - 1

            if abs(z) < 1e-6:
                return np.real(y * y * (1.0 / z**2 - 0.5))

            if abs(denom) < 1e-15:
                return 0.0

            val = y * y / (z * denom)
            # if np.isnan(val):
            #    print("val???", val)

            return np.real(val)

        # split integral
        # i1, _ = integrate.quad(integrand, 0.0, 1.0, limit=200)
        i2, _ = integrate.quad(integrand, max(0, np.sqrt(-x)), 200, limit=200)
        # if np.isnan(i2):
        #    print(i2)
        return -0.5 * i2

    def _dJb_exact2(self, x):
        """
        Vectorized wrapper around the scalar integrator.
        x can be a scalar or numpy array.
        Returns the same shape filled with the numeric real result for dJb/dx.
        """
        x_arr = np.array(x, copy=False, ndmin=1)
        # vectorize the scalar integrator (cache option could be used for performance)
        vec = np.vectorize(
            lambda s: self._dJb_exact2_scalar(np.float64(s)), otypes=[np.float64]
        )
        res = vec(x_arr)
        # if the input was scalar, return scalar
        if np.ndim(x) == 0:
            return float(res[0])
        return res.reshape(x_arr.shape)

    def dJb_exact2(self, theta):
        """Jb calculated directly form the integral; input is theta = x^2."""
        return CTFT.arrayFunc(self._dJb_exact2, theta)

    def find_new_minima(self):
        min1 = scipy.optimize.minimize(self.V, [self.v], method="Nelder-Mead")
        bnds = (-4 * self.v, -1.2 * self.v)
        min2 = scipy.optimize.minimize(self.V, bnds)
        # assert self.V(min2.x) > self.V(min1.x)
        return min1.x[0], min2.x[0]

    @property
    def tv(self):
        return scipy.optimize.minimize(self.V, [self.v]).x[0]

    @property
    def fv(self):
        return scipy.optimize.minimize(self.V, [-self.v]).x[0]
