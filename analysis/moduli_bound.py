"""Final modulus abundance of double thermal inflation (professor's note of 27 Oct 2026, Secs. II-VI).
Assumptions of the note: modulus-dominated first onset with Phi_osc = M_Pl, prompt first reheating (r_S = 0),
radiation-dominated intermediate era, matter-like second flaton decay at T_d2, coherent regeneration at both exits
with couplings b_i and sudden-shift response S_i = 1; no direct/thermal production."""
import math
import numpy as np
M_PL = 2.435e18


def s_of_T(T, g=100.):
    return 2 * math.pi**2 / 45 * g * T**3


def nphi_s_osc(m_phi, Phi_osc=M_PL, g=100.):
    T_osc = (90 / (math.pi**2 * g))**0.25 * math.sqrt(m_phi * M_PL)
    return 0.125 * (Phi_osc / M_PL)**2 * T_osc / m_phi, T_osc


def abundance(N1, N2, Tc1, Tc2, Td2, m_phi=1000., b1=0.1, b2=0.1, S1=1., S2=1., Phi_osc=M_PL, g=100.):
    """Returns dict of (n_phi/s) contributions and V1, V2 (GeV^4). Vectorized in N1, N2 (broadcast)."""
    N1 = np.asarray(N1, float); N2 = np.asarray(N2, float)
    nosc, _ = nphi_s_osc(m_phi, Phi_osc, g)
    V1 = m_phi * nosc * s_of_T(Tc1, g) * np.exp(3 * N1)                 # Eq. (5), modulus-dominated first onset
    V2 = math.pi**2 * g / 30 * Tc2**4 * np.exp(4 * N2)                     # Eq. (13), radiation onset of stage 2
    TR1 = (30 * V1 / (math.pi**2 * g))**0.25
    D1 = (TR1 / Tc1)**3                                                    # Eq. (7)
    D2 = Tc2 / Td2 * np.exp(4 * N2)                                        # Eq. (7), matter-like second decay
    pri = nosc / (D1 * D2)                                                 # Eq. (9)
    c1 = 3 / 8 * b1**2 * V1 * TR1 / (m_phi**3 * M_PL**2) * S1 / D2         # Eq. (10)
    c2 = 3 / 8 * b2**2 * V2 * Td2 / (m_phi**3 * M_PL**2) * S2              # Eq. (11)
    return dict(pri=pri, c1=c1, c2=c2, tot=pri + c1 + c2, V1=V1, V2=V2, D1=D1, D2=D2, nosc=nosc)


if __name__ == '__main__':
    # --- reproduce the note: Tc1 = 2 TeV, Tc2 = 1 TeV, Td2 = 100 GeV, b = 0.1
    nosc, Tosc = nphi_s_osc(1000.)
    print(f"T_osc = {Tosc:.4e} GeV  (note 2.712e10),  (n/s)_osc = {nosc:.4e}  (note 3.390e6)")
    r = abundance(0., 0., 2000., 1000., 100.)
    print(f"pri prefactor = {r['pri']:.4f}  (note 5.816),  ln(5.816/1e-15) = {math.log(5.816e15):.2f} (note 36.30)")
    for N1, N2, lab in ((7.68, 4.81, 'LS 4e12,1e8 -> 8.02e-16'), (7.22, 5.16, 'LS 2e12,2e8 -> 5.67e-16'), (6.62, 4.71, 'Draft 1e12,1e8 -> 1.30e-14'), (8.15, 4.71, 'Draft 1e13,1e8 -> 4.13e-16')):
        r = abundance(N1, N2, 2000., 1000., 100.)
        print(f"{lab}: tot={float(r['tot']):.3e} (pri {float(r['pri']):.2e}, c1 {float(r['c1']):.2e}, c2 {float(r['c2']):.2e})")
    # fixed N2 = 5: allowed N1 (note: 7.24 < N1 < 10.39)
    N1g = np.linspace(3, 13, 2001); t = abundance(N1g, 5.0, 2000., 1000., 100.)['tot']
    ok = N1g[t < 1e-15]; print(f"N2=5: allowed N1 = {ok.min():.2f} .. {ok.max():.2f}  (note 7.24 .. 10.39)")
