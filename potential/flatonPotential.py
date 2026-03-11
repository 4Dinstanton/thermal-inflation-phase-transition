import numpy as np
import sympy as sp
import math
from cosmoTransitions import finiteT as CTFT
from cosmoTransitions import pathDeformation as CTPD
import matplotlib.pyplot as plt
import scipy
import math


class FlatonPotential:
    def __init__(self):
        self.log, self.exp, self.sqrt = np.log, np.exp, np.sqrt
        self.phiMass = 1
        self.phiMassSquared = self.phiMass**2
        self.thermalInflationPotential = 10
        self.lambdaSix = self.phiMass**6 / (54 * self.thermalInflationPotential**2)
        self.phiVEV = self.sqrt(3 * self.thermalInflationPotential) / self.phiMass

        self.bosonMassSquared = 1
        self.bosonCoupling = 15
        self.bosonGaugeCoupling = 10

        self.fermionCoupling = 15
        self.fermionGaugeCoupling = 10
        self.temperature = 1

        pass

    def set_parameters(self, param_dict):
        self.phiMass = param_dict["phiMass"]
        self.phiMassSquared = self.phiMass**2
        self.thermalInflationPotential = param_dict["thermalInflationPotential"]
        self.lambdaSix = self.phiMass**6 / (54 * self.thermalInflationPotential**2)
        try:
            self.phiVEV = self.sqrt(3 * self.thermalInflationPotential) / self.phiMass
        except:
            self.phiVEV = self.sqrt(3 * self.thermalInflationPotential
                                    / 10**10) / self.phiMass * 10**5

        self.bosonMassSquared = param_dict["bosonMassSquared"]
        self.bosonCoupling = param_dict["bosonCoupling"]
        self.bosonGaugeCoupling = param_dict["bosonGaugeCoupling"]

        self.fermionCoupling = param_dict["fermionCoupling"]
        self.fermionGaugeCoupling = param_dict["fermionGaugeCoupling"]

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_hubble(self, hubble):
        self.hubble = hubble

    def arrayFunc(self, f, x, typ=float):
        # This function allows a 1D array to be passed to something that
        # normally can't handle it
        i = 0
        try:
            n = len(x)
        except:
            return f(x)  # x isn't an array
        s = np.empty(n, typ)
        while (i < n):
            try:
                s[i] = f(x[i])
            except:
                s[i] = np.NaN
            i += 1
        return s

    def bosonic_input(self, x):
        return self.sqrt((self.bosonMassSquared + 0.5 * self.bosonCoupling**2 * x**2 + (0.25
                                                                                        * self.bosonCoupling**2 + 2 / 3 * self.bosonGaugeCoupling**2) * self.temperature**2)) / self.temperature

    def fermionic_input(self, x):
        return self.sqrt(0.5 * self.fermionCoupling**2 * x**2 + 1 / 6
                         * self.fermionGaugeCoupling**2 * self.temperature**2) / self.temperature

    def _Jf_exact(self, x):
        def f(y): return -1 * y * y * self.log(1 + self.exp(-self.sqrt(y * y + x * x)))
        # def f(y): return -y * y * self.log(1 + exp(-sqrt(x * x + y * y)))
        if (x.imag == 0):
            # x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]
        else:
            def f1(y): return -y * y * self.log(2 * abs(np.cos(self.sqrt(abs(x * x) - y * y) / 2)))
            return (
                scipy.integrate.quad(f1, 0, abs(x))[0]
                + scipy.integrate.quad(f, abs(x), np.inf)[0]
            )

    def _Jb_exact(self, x):
        def f(y): return y * y * self.log(1 - self.exp(-self.sqrt(y * y + x * x)))
        # def f(y): return y * y * self.log(1 - exp(-sqrt(x * x + y * y)))
        if (x.imag == 0):
            # x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]
        else:
            def f1(y): return y * y * self.log(2 * abs(np.sin(self.sqrt(abs(x * x) - y * y) / 2)))
            return (
                scipy.integrate.quad(f1, 0, abs(x))[0]
                + scipy.integrate.quad(f, abs(x), np.inf)[0]
            )

    def _dJf_exact(self, x):
        def f(y): return y * y * (self.exp(self.sqrt(y * y + x * x)) + 1)**- \
            1 * x / self.sqrt(y * y + x * x)
        return scipy.integrate.quad(f, 0, 100)[0]

    def _dJb_exact(self, x):
        def f(y): return y * y * (self.exp(self.sqrt(y * y + x * x)) - 1)**- \
            1 * x / self.sqrt(y * y + x * x)
        return scipy.integrate.quad(f, 0, 100)[0]

    def _dxdphi_boson(self, x):
        return 0.5 * self.bosonCoupling**2 / self.temperature**2 * x

    def _dxdphi_fermion(self, x):
        return 0.5 * self.fermionCoupling**2 / self.temperature**2 * x

    def _Jb_exact2(self, theta):
        # Note that this is a function of theta so that you can get negative values
        #print(self.log(1 - self.exp(-self.sqrt(theta))).real)
        def f(y): return y * y * self.log(1 - self.exp(-self.sqrt(y * y + theta))).real
        # print(f(1))
        if theta >= 0:
            return scipy.integrate.quad(f, 0, np.inf)[0]
        else:
            def f1(y): return y * y * self.log(2 * abs(np.sin(self.sqrt(-theta - y * y) / 2)))
            return (
                scipy.integrate.quad(f, abs(theta)**.5, np.inf)[0]
                + scipy.integrate.quad(f1, 0, abs(theta)**.5)[0]
            )

    def _Jf_exact2(self, theta):
        # Note that this is a function of theta so that you can get negative values
        def f(y): return -y * y * self.log(1 + self.exp(-self.sqrt(y * y + theta))).real
        if theta >= 0:
            return scipy.integrate.quad(f, 0, np.inf)[0]
        else:
            def f1(y): return -y * y * self.log(2 * abs(np.cos(self.sqrt(-theta - y * y) / 2)))
            return (
                scipy.integrate.quad(f, abs(theta)**.5, np.inf)[0]
                + scipy.integrate.quad(f1, 0, abs(theta)**.5)[0]
            )

    def _dJbdphi_exact(self, x):
        def sqrt_part(y): return self.sqrt(y * y + (self.bosonMassSquared + 0.5 * self.bosonCoupling**2
                                                    * x * x) / self.temperature**2 + (0.25 * self.bosonCoupling**2 + 2 / 3 * bosonGaugeCoupling**2))
        def f(y): return y * y * (1 / (1 - self.exp(-1 * sqrt_part(y)))) * self.exp(-1
                                                                                    * sqrt_part(y)) * (self.bosonCoupling / self.temperature)**2 * x / sqrt_part(y)

        if (x.imag == 0):
            x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]

    def _dJfdphi_exact(self, x):
        def sqrt_part(y): return self.sqrt(y * y + 0.5 * self.FermionCoupling**2 * x
                                           ** 2 / self.temperature**2 + 1 / 6 * self.fermionGaugeCoupling**2)

        def f(y): return -y * y * (1 / (1 + self.exp(-1 * sqrt_part(y)))) * -1 * self.exp(-1
                                                                                          * sqrt_part(y)) * ((self.fermionCoupling / self.temperature)**2 * x / sqrt_part(y))
        if (x.imag == 0):
            x = abs(x)
            return scipy.integrate.quad(f, 0, np.inf)[0]

    def Jf_exact(self, x):
        """Jf calculated directly from the integral."""
        return self.arrayFunc(self._Jf_exact, x)

    def Jb_exact(self, x):
        """Jb calculated directly from the integral."""
        return self.arrayFunc(self._Jb_exact, x)

    def dJfdphi_exact(self, x):
        return self.arrayFunc(self._dJfdphi_exact, x, complex)

    def dJbdphi_exact(self, x):
        return self.arrayFunc(self._dJbdphi_exact, x, complex)


    def dJf_exact(self, x):
        """dJf/dx calculated directly from the integral."""
        return self.arrayFunc(self._dJf_exact, x)

    def dJb_exact(self, x):
        """dJb/dx calculated directly from the integral."""
        return self.arrayFunc(self._dJb_exact, x)


    def V_t(self, X):
        phi = X[..., 0]
        treeLevelPotential = (
            self.thermalInflationPotential - 0.5 * self.phiMassSquared * phi**2 + self.lambdaSix * phi**6
        )
        return treeLevelPotential

    def dV_t(self, X):
        phi = X[..., 0]
        return (-self.phiMassSquared * phi + 6 * self.lambdaSix * phi**5)

    def V_p(self, X):
        phi = X[..., 0]
        treeLevelPotential = self.V_t(X)
        thermalCorrectionBosonPotential = self.Jb_exact(self.bosonic_input(phi))
        thermalCorrectionFermionPotential = self.Jf_exact(self.fermionic_input(phi))
        correctedPotential = treeLevelPotential + self.temperature**4 / \
            (2 * math.pi**2) * (2 * thermalCorrectionBosonPotential + thermalCorrectionFermionPotential)
        return correctedPotential

    def dV_p(self, X):
        phi = X[..., 0]
        dVdphi = self.dV_t(phi)
        thermalCorrectionBosonPotentialDerivative = self.dJb_exact(
            self.bosonic_input(phi)) * self._dxdphi_boson(phi)
        thermalCorrectionFermionPotentialDerivative = self.dJf_exact(
            self.fermionic_input(phi)) * self._dxdphi_fermion(phi)

        correctedPotentialDerivative = dVdphi + self.temperature**4 / \
            (2 * math.pi**2) * (2 * thermalCorrectionBosonPotentialDerivative
                                + thermalCorrectionFermionPotentialDerivative)
        # print(correctedPotentialDerivative)

        return correctedPotentialDerivative

    def find_new_minimum(self):
        min = scipy.optimize.minimize(self.V_p, [self.phiVEV])
        return min.x

    def P_T(self, action):
        def f(T): return (4 / 3 * math.pi) * (T**3 / self.hubble**4) * (T / self.temperature - 1)**3 * \
            self.exp(-action / self.temperature)
        return 4 * math.pi / 3 * scipy.integrate.quad(f, self.temperature, self.temperature * 10**3)[0]

    def bubble_fraction(self, action):
        return np.round(1 - self.exp(-self.P_T(action)), 4)


phiMass = 1000
PhiMassSquared = phiMass**2
thermalInflationPotential = 10**24

lambdaSix = phiMass**6 / (54 * thermalInflationPotential**2)
# print(LambdaSix)

phiVEV = math.sqrt(3 * thermalInflationPotential) / phiMass
# phi = np.arange(0, thermalInflationPotential, thermalInflationPotential / 1000)
# phi = np.arange(0, 8, 0.1)

bosonMassSquared = 1000
bosonCoupling = 1
bosonGaugeCoupling = 1
fermionCoupling = 1
fermionGaugeCoupling = 1

param_dict = {"phiMass": phiMass, "thermalInflationPotential": thermalInflationPotential,
              "lambdaSix": lambdaSix, "bosonMassSquared": bosonMassSquared,
              "bosonCoupling": bosonCoupling, "bosonGaugeCoupling" : bosonGaugeCoupling,
              "fermionCoupling": fermionCoupling, "fermionGaugeCoupling": fermionGaugeCoupling}

Potential = FlatonPotential()
Potential.set_parameters(param_dict)
# print(1 / (10**38 / thermalInflationPotential))
H_c = np.sqrt(1 / (3 * (10**38 / thermalInflationPotential)))
Potential.set_hubble(H_c)
Potential.set_temperature(300000)

x = np.arange(0, Potential.phiVEV * 1.2, Potential.phiVEV * 1.2 / 1000)
x = x.reshape(len(x), 1)

"""
bosonMassSquared = 1000
bosonCoupling = 1
bosonGaugeCoupling = 1
fermionCoupling = 1
fermionGaugeCoupling = 1

param_dict = {"phiMass": phiMass, "thermalInflationPotential": thermalInflationPotential,
              "lambdaSix": lambdaSix, "bosonMassSquared": bosonMassSquared,
              "bosonCoupling": bosonCoupling, "bosonGaugeCoupling" : bosonGaugeCoupling,
              "fermionCoupling": fermionCoupling, "fermionGaugeCoupling": fermionGaugeCoupling}


b1 = Potential.bosonic_input(x)
pb1 = Potential.Jb_exact(Potential.bosonic_input(x))
vp1 = Potential.V_p(x)
Potential.thermalInflationPotential = 10**24
b2 = Potential.bosonic_input(x)
pb2 = Potential.Jb_exact(Potential.bosonic_input(x))

vp2 = (Potential.V_p(x))
"""
# print(vp2)

# hehe

# Potential.set_parameters(param_dict)
# comp_p2 = Potential.V_p(x)
"""
plt.plot(x, comp_p2 - comp_p2[0] + comp_p1[0], label="Small boson/fermion coupling")
plt.title("Potential comparison depending on couplings")
plt.legend()
plt.savefig("/Users/instanton/Tunnel/coupling_comp.jpg", dpi=300)
"""
# plt.show()

temperature_list = np.arange(4620, 4630, 0.25)

action_list = []
fraction_list = []
x = np.arange(0, Potential.phiVEV * 0.001, Potential.phiVEV * 0.001 / 1000)
x = x.reshape(len(x), 1)
#comp_p1 = Potential.V_p(x)
#comp_p2 = Potential.V_p(x)
print("start")
print(x.shape)
temp_list = np.arange(1000, 21000, 2000)
for temp in temp_list:
    Potential.set_temperature(int(temp))
    x = np.arange(0, Potential.phiVEV * 0.0002, Potential.phiVEV * 0.0002/1000)
    x = x.reshape(len(x), 1)
    plt.plot(x, Potential.V_p(x), label=f'T={temp}')
    pp = Potential.V_p(x)
    print(pp[np.argwhere(pp>pp[0])])
    plt.xlim([x[0], x[-1]])
plt.legend()
#plt.ylim()
plt.show()
print(ho)
for temp in temperature_list:
    # print(980000 - temp)

    Potential.find_new_minimum()
    #print(ho)
    # print(Potential.temperature)
    # print(comp_p2)
    # plt.plot(x, Potential.V_p(x))

    print(f"temperature {temp}")
    tunneling_result = CTPD.fullTunneling(
        path_pts=np.array(
            [
                [Potential.find_new_minimum()[0]],
                [0],
            ]
        ),
        V=Potential.V_p,
        dV=Potential.dV_p,
        maxiter=1,
        V_spline_samples=120000,
        tunneling_init_params=dict(alpha=2),
        tunneling_findProfile_params=dict(
            xtol=0.0000000001, phitol=0.000000001, rmin=0.0000001, npoints=1000),
        deformation_class=CTPD.Deformation_Spline,
    )

    action_list.append(tunneling_result.action / temp)
    fraction_list.append(Potential.bubble_fraction(tunneling_result.action))
    # print("action", tunneling_result.action)
    # print("fraction", Potential.bubble_fraction(tunneling_result.action))
# print("action", tunneling_result.action / Potential.temperature)
print(action_list, fraction_list)
# plt.show()


plt.plot(temperature_list, action_list, label="Action results")
plt.xlabel("Temperature (GeV)")
plt.ylabel(r"$\dfrac{S_3}{T}$")
plt.tick_params(axis="y", labelcolor="blue")
frac_plot = plt.twinx()
frac_plot.plot(temperature_list, fraction_list, color="red",
               label="Bubble volume fraction results")
frac_plot.set_ylim(-0.1, 1.1)
frac_plot.set_ylabel("F(T)")
frac_plot.tick_params(axis="y", labelcolor="red")


plt.title("Thermal tunneling plots depending on temperature")
plt.savefig("/Users/instanton/Tunnel/thermal_tunneling_trial.jpg", dpi=300)
plt.show()
plt.cla()


# plt.savefig("/Users/instanton/Tunnel/bubble_volume_fraction.jpg", dpi=300)

plt.plot(x, corrected_potential / Potential.thermalInflationPotential)
# plt.plot(x, (pot_cor / Potential.thermalInflationPotential))
# plt.plot(phi_min, Potential.V_p(phi_min) / Potential.thermalInflationPotential, "x")
# plt.plot(x, thermalCorrectionBosonPotential + thermalCorrectionFermionPotential)
# plt.waitforbuttonpress()

print("zero_val", Potential.V_p(np.array([[0],])))
print("min_val", Potential.V_p(np.array([Potential.find_new_minimum(),])))

"""
x = np.arange(0, Potential.phiVEV * scaling , Potential.phiVEV * scaling / 500)

BosonY = sqrt((bosonMassSquared + 0.5 * bosonCoupling**2 * x ** 2 + (0.25
                                                                     * bosonCoupling**2 + 2 / 3 * bosonGaugeCoupling**2) * Potential.temperature**2)) / Potential.temperature

FermionY = sqrt(0.5 * fermionCoupling**2 * x**2 + 1 / 6
                * fermionGaugeCoupling**2 * Potential.temperature**2) / Potential.temperature
# print(BosonY**2)

ThermalCorrectionBoson = 1 / (2 * math.pi**2) * Potential.Jb_exact(BosonY)
ThermalCorrectionFermion = 1 / (2 * math.pi**2) * Potential.Jf_exact(FermionY)
# print(ThermalCorrectionBoson)

TreeLevelPotential = thermalInflationPotential - 0.5 * PhiMassSquared * x**2 + lambdaSix * x**6

V = TreeLevelPotential + Potential.temperature**4 * \
    (ThermalCorrectionBoson + ThermalCorrectionFermion)

print(V)
"""

# plt.plot(x, tlp / Potential.thermalInflationPotential)
# plt.plot(x, V)
# plt.plot(x, (pot_cor / Potential.thermalInflationPotential))
# plt.plot(phi_min, Potential.V_p(phi_min) / Potential.thermalInflationPotential, "x")
# plt.plot(x, thermalCorrectionBosonPotential + thermalCorrectionFermionPotential)
# plt.show()


"""
"""
