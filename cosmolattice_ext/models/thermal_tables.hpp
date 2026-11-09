#ifndef THERMAL_TABLES_HPP
#define THERMAL_TABLES_HPP

/* Thermal one-loop integral tables for the CosmoLattice thermal-inflation model.
 *
 * This is a *standalone* C++ helper (no CosmoLattice / TempLat dependency) so it
 * can be unit-tested on its own. It loads the binary table produced by
 * tools/export_thermal_splines.py and assembles the full finite-temperature
 * effective potential V(phi,T), its first derivative V'(phi,T) and second
 * derivative V''(phi,T).
 *
 * Conventions (must match tools/export_thermal_splines.py and the numba solver
 * simulation/latticeSimeRescale_numba.py):
 *
 *   tree:     V_tree   = (lam/4)(phi^2 - phi0^2)^2
 *                       = (lam/4) phi^4 - (mphi^2/2) phi^2 + DeltaV
 *             with phi0^2 = mphi^2/lam, DeltaV = mphi^4/(4 lam)
 *             so V_tree(0)=+DeltaV (false vac) and V_tree(phi0)=0 (true vac).
 *             The additive DeltaV does not change V' (dynamics) but keeps
 *             CosmoLattice Energies::rho > 0 after nucleation so GW spectra
 *             (which divide by rho) stay non-negative.
 *   thermal:  V_th     = T^4/(2 pi^2) [ nb Jb(u_b) + nf Jf(u_f) ]
 *   radiation:V_rad    = pi^2/30 * gStarPot * T^4              (phi-independent)
 *   CW:       V_cw     = (m2)^2/(64 pi^2) (log(|m2|/T) - 3/2),  m2 = 3 lam phi^2
 *
 *   u_b = sqrt( mb2_0 + 0.5 yb^2 phi^2 + (0.25 yb^2 + 2/3 gb^2) T^2 ) / T
 *   u_f = sqrt( 0.5 yf^2 phi^2 + (1/6) gf^2 T^2 ) / T
 *
 *   du_b/dphi = 0.5 yb^2 phi / (T^2 u_b)        (correct chain rule; numba-consistent)
 *   du_f/dphi = 0.5 yf^2 phi / (T^2 u_f)
 *
 * NOTE: Potential.dV_p_correct in the Python repo uses a different (incorrect)
 * chain rule; we deliberately match the *correct* derivative (= derivative of V).
 *
 * Two-step tree (treeType = TwoStep, potential/twoStepPotential.py), x = |phi|/v:
 *   V_tree/V0 = 1 + k2 x^2 + k3 x^3 + k4 x^4 + k6 x^6
 *             = (x - 1)^2 (q0 + q1 x + q2 x^2 + q3 x^3 + q4 x^4)
 * The factorised form is used for V so V(v) = 0 holds without a V0 cancellation.
 * The odd term is c3 |phi|^3: with an analytic phi^3 a real field would have a
 * deeper vacuum at phi < 0. In TwoStep mode u_b,f drop the mb2_0 and T^2 terms,
 * so with yb = sqrt(2) g_B, yf = sqrt(2) g_F one has u = g |phi| / T (paper Eq. 4).
 * Use includeCW = false.
 */

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThermalInflation {

class ThermalTables {
public:
    // Model parameters needed to assemble V from the pure J(u) tables.
    struct Params {
        double lam = 1e-24;
        double mphi = 1000.0;
        double mb2_0 = 1.0e6;   // bosonMassSquared
        double yb = 1.09;       // bosonCoupling
        double gb = 1.05;       // bosonGaugeCoupling
        double yf = 1.09;       // fermionCoupling
        double gf = 1.05;       // fermionGaugeCoupling
        double nb = 20.0;       // boson multiplicity (0 => fermion-only)
        double nf = 20.0;       // fermion multiplicity
        double gStarPot = 100.0; // radiation free-energy multiplicity
        bool includeCW = true;   // include Coleman-Weinberg force/energy
        // Tree shape: LambdaPhi4 = (lam/4)(phi^2 - phi0^2)^2; TwoStep = sextic above.
        enum class TreeType { LambdaPhi4 = 0, TwoStep = 1 };
        TreeType treeType = TreeType::LambdaPhi4;
        double V0 = 0.0;                        // TwoStep: V_tree(0) (GeV^4)
        double vVac = 1.0;                      // TwoStep: global minimum v (GeV)
        double k2 = 0.0, k3 = 0.0, k4 = 0.0, k6 = 0.0;  // TwoStep: dimensionless
    };

    ThermalTables() = default;

    explicit ThermalTables(const std::string& binPath) { load(binPath); }

    ThermalTables(const std::string& binPath, const Params& p) {
        load(binPath);
        setParams(p);
    }

    void setParams(const Params& p) {
        params_ = p;
        // (x-1)^2 Q(x) = 1 + k2 x^2 + k3 x^3 + k4 x^4 + k6 x^6, matched from x^6 down.
        q_[4] = p.k6;
        q_[3] = 2.0 * q_[4];
        q_[2] = p.k4 + 2.0 * q_[3] - q_[4];
        q_[1] = p.k3 + 2.0 * q_[2] - q_[3];
        q_[0] = p.k2 + 2.0 * q_[1] - q_[2];
    }

    bool twoStep() const { return params_.treeType == Params::TreeType::TwoStep; }

    // Tree-level pieces (no thermal / radiation / CW).
    double Vtree(double phi) const {
        const Params& p = params_;
        if (twoStep()) {
            const double x = std::fabs(phi) / p.vVac;
            const double Q = q_[0] + x * (q_[1] + x * (q_[2] + x * (q_[3] + x * q_[4])));
            return p.V0 * (x - 1.0) * (x - 1.0) * Q;
        }
        if (p.lam > 0.0) {
            const double phi0_sq = p.mphi * p.mphi / p.lam;
            const double dphi2 = phi * phi - phi0_sq;
            return 0.25 * p.lam * dphi2 * dphi2;
        }
        return 0.25 * p.lam * std::pow(phi, 4) - 0.5 * p.mphi * p.mphi * phi * phi;
    }
    double VtreePrime(double phi) const {
        const Params& p = params_;
        if (twoStep()) {
            const double x = std::fabs(phi) / p.vVac;
            const double x2 = x * x;
            const double sgn = phi >= 0.0 ? 1.0 : -1.0;
            return sgn * p.V0 / p.vVac
                 * x * (2.0 * p.k2 + x * (3.0 * p.k3 + x * (4.0 * p.k4 + 6.0 * p.k6 * x2)));
        }
        return p.lam * std::pow(phi, 3) - p.mphi * p.mphi * phi;
    }
    double VtreeSecond(double phi) const {
        const Params& p = params_;
        if (twoStep()) {
            const double x = std::fabs(phi) / p.vVac;
            const double x2 = x * x;
            return p.V0 / (p.vVac * p.vVac)
                 * (2.0 * p.k2 + x * (6.0 * p.k3 + x * (12.0 * p.k4 + 30.0 * p.k6 * x2)));
        }
        return 3.0 * p.lam * phi * phi - p.mphi * p.mphi;
    }
    const Params& params() const { return params_; }

    // Load the little-endian binary table (see export_thermal_splines.py).
    void load(const std::string& binPath) {
        std::ifstream in(binPath, std::ios::binary);
        if (!in) throw std::runtime_error("ThermalTables: cannot open " + binPath);
        int64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(int64_t));
        in.read(reinterpret_cast<char*>(&umin_), sizeof(double));
        in.read(reinterpret_cast<char*>(&umax_), sizeof(double));
        if (n <= 1) throw std::runtime_error("ThermalTables: invalid grid size");
        n_ = static_cast<size_t>(n);
        auto readArr = [&](std::vector<double>& v) {
            v.resize(n_);
            in.read(reinterpret_cast<char*>(v.data()),
                    static_cast<std::streamsize>(n_ * sizeof(double)));
        };
        readArr(Jb_);
        readArr(Jf_);
        readArr(dJb_);
        readArr(dJf_);
        readArr(d2Jb_);
        readArr(d2Jf_);
        if (!in) throw std::runtime_error("ThermalTables: truncated table file");
        du_ = (umax_ - umin_) / static_cast<double>(n_ - 1);
        invdu_ = 1.0 / du_;
    }

    bool loaded() const { return n_ > 1; }

    // ---- Full assembled potential and derivatives -------------------------
    double V(double phi, double T) const {
        const Params& p = params_;
        const double pref = std::pow(T, 4) / (2.0 * M_PI * M_PI);
        const double ub = uBoson(phi, T);
        const double uf = uFermion(phi, T);
        // Shifted tree as (lam/4)(phi^2 - phi0^2)^2 — algebraically equal to
        // (lam/4)phi^4 - (mphi^2/2)phi^2 + DeltaV, but stable near phi=phi0
        // (expanded form suffers O(1e19) GeV^4 cancellation → spurious potAvI < 0
        // and negative P_GW via /rho).
        double v = Vtree(phi);
        // J interpolated with cubic Hermite using dJ as the nodal derivative.
        v += pref * (p.nb * hermite(Jb_, dJb_, ub) + p.nf * hermite(Jf_, dJf_, uf));
        v += M_PI * M_PI / 30.0 * p.gStarPot * std::pow(T, 4);
        if (p.includeCW) v += cw(phi, T);
        return v;
    }

    double Vprime(double phi, double T) const {
        const Params& p = params_;
        const double pref = std::pow(T, 4) / (2.0 * M_PI * M_PI);
        const double ub = uBoson(phi, T);
        const double uf = uFermion(phi, T);
        const double ubs = ub > 1e-20 ? ub : 1e-20;
        const double ufs = uf > 1e-20 ? uf : 1e-20;
        const double dub = 0.5 * p.yb * p.yb * phi / (T * T * ubs);
        const double duf = 0.5 * p.yf * p.yf * phi / (T * T * ufs);
        double vp = VtreePrime(phi);
        // dJ interpolated with cubic Hermite using d2J as the nodal derivative.
        vp += pref * (p.nb * hermite(dJb_, d2Jb_, ub) * dub
                    + p.nf * hermite(dJf_, d2Jf_, uf) * duf);
        if (p.includeCW) vp += dcw(phi, T);
        return vp;
    }

    double Vsecond(double phi, double T) const {
        const Params& p = params_;
        const double pref = std::pow(T, 4) / (2.0 * M_PI * M_PI);
        const double ub = uBoson(phi, T);
        const double uf = uFermion(phi, T);
        const double ubs = ub > 1e-20 ? ub : 1e-20;
        const double ufs = uf > 1e-20 ? uf : 1e-20;
        // u = sqrt(A + B phi^2)/T  ->  du/dphi = B phi/(T^2 u),
        //                              d2u/dphi2 = B/(T^2 u) - (B phi)^2/(T^4 u^3)
        const double Bb = 0.5 * p.yb * p.yb;
        const double Bf = 0.5 * p.yf * p.yf;
        double dub, duf, d2ub, d2uf;
        if (twoStep()) {
            // A = 0: u = sqrt(B)|phi|/T is piecewise linear (avoids 0 * 1/u at phi = 0).
            const double sgn = phi >= 0.0 ? 1.0 : -1.0;
            dub = sgn * std::sqrt(Bb) / T;
            duf = sgn * std::sqrt(Bf) / T;
            d2ub = 0.0;
            d2uf = 0.0;
        } else {
            dub = Bb * phi / (T * T * ubs);
            duf = Bf * phi / (T * T * ufs);
            d2ub = Bb / (T * T * ubs) - (Bb * phi) * (Bb * phi) / (std::pow(T, 4) * std::pow(ubs, 3));
            d2uf = Bf / (T * T * ufs) - (Bf * phi) * (Bf * phi) / (std::pow(T, 4) * std::pow(ufs, 3));
        }
        double vpp = VtreeSecond(phi);
        vpp += pref * (p.nb * (interp(d2Jb_, ub) * dub * dub + hermite(dJb_, d2Jb_, ub) * d2ub)
                     + p.nf * (interp(d2Jf_, uf) * duf * duf + hermite(dJf_, d2Jf_, uf) * d2uf));
        if (p.includeCW) vpp += d2cw(phi, T);
        return vpp;
    }

    // Dimensionless argument helpers (exposed for testing).
    double uBoson(double phi, double T) const {
        const Params& p = params_;
        // TwoStep follows the paper's m = g|phi| (no Debye/vacuum mass in J).
        const double th = twoStep() ? 0.0 : 1.0;
        const double m2 = th * p.mb2_0 + 0.5 * p.yb * p.yb * phi * phi
                          + th * (0.25 * p.yb * p.yb + 2.0 / 3.0 * p.gb * p.gb) * T * T;
        return std::sqrt(m2 > 0 ? m2 : 0.0) / T;
    }
    double uFermion(double phi, double T) const {
        const Params& p = params_;
        const double th = twoStep() ? 0.0 : 1.0;
        const double m2 = 0.5 * p.yf * p.yf * phi * phi + th * (1.0 / 6.0) * p.gf * p.gf * T * T;
        return std::sqrt(m2 > 0 ? m2 : 0.0) / T;
    }

    // Radial V'(rho) with chain rule to (phi1, phi2); optional Z_N angular term.
    void vPrimeComponents(double phi1, double phi2, double T,
                          double znOrder, double znStrength, bool znActive,
                          double& dV1, double& dV2) const {
        const double rho = std::sqrt(phi1 * phi1 + phi2 * phi2);
        const double rhoSafe = rho > 1e-30 ? rho : 1e-30;
        const double invRho = 1.0 / rhoSafe;
        double dVdr = Vprime(rho, T);
        dV1 = dVdr * phi1 * invRho;
        dV2 = dVdr * phi2 * invRho;
        if (znActive && znOrder > 0.5 && znStrength > 0.0 && rho > 1e-20) {
            const double theta = std::atan2(phi2, phi1);
            const double sinN = std::sin(znOrder * theta);
            const double invRho2 = invRho * invRho;
            dV1 += -znOrder * znStrength * sinN * phi2 * invRho2;
            dV2 += znOrder * znStrength * sinN * phi1 * invRho2;
        }
    }

private:
    // Linear interpolation on the dense uniform grid, flat extrapolation.
    double interp(const std::vector<double>& tab, double u) const {
        if (u <= umin_) return tab.front();
        if (u >= umax_) return tab.back();
        const double x = (u - umin_) * invdu_;
        const size_t i = static_cast<size_t>(x);
        const double frac = x - static_cast<double>(i);
        return tab[i] * (1.0 - frac) + tab[i + 1] * frac;
    }

    // Cubic Hermite interpolation of `val` using `deriv` as the exact nodal
    // derivative (d val / du). Near-exact (O(du^4)) on the dense uniform grid.
    // Flat extrapolation outside [umin, umax].
    double hermite(const std::vector<double>& val, const std::vector<double>& deriv,
                   double u) const {
        if (u <= umin_) return val.front();
        if (u >= umax_) return val.back();
        const double x = (u - umin_) * invdu_;
        const size_t i = static_cast<size_t>(x);
        const double t = x - static_cast<double>(i);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;
        // nodal derivatives w.r.t. the unit interval are deriv*du.
        return h00 * val[i] + h10 * du_ * deriv[i]
             + h01 * val[i + 1] + h11 * du_ * deriv[i + 1];
    }

    double cw(double phi, double T) const {
        const double m2 = 3.0 * params_.lam * phi * phi;
        const double m2abs = std::fabs(m2);
        if (m2abs <= 0.0) return 0.0;
        return m2 * m2 / (64.0 * M_PI * M_PI) * (std::log(m2abs / T) - 1.5);
    }
    double dcw(double phi, double T) const {
        const double m2 = 3.0 * params_.lam * phi * phi;
        const double dm2 = 6.0 * params_.lam * phi;
        const double m2abs = std::fabs(m2);
        if (m2abs <= 0.0) return 0.0;
        return dm2 * m2 / (64.0 * M_PI * M_PI) * 2.0 * (std::log(m2abs / T) - 1.0);
    }
    double d2cw(double phi, double T) const {
        // d/dphi of dcw, with m2 = 3 lam phi^2, dm2 = 6 lam phi, d2m2 = 6 lam.
        const double lam = params_.lam;
        const double m2 = 3.0 * lam * phi * phi;
        const double m2abs = std::fabs(m2);
        if (m2abs <= 0.0) return 0.0;
        const double dm2 = 6.0 * lam * phi;
        const double d2m2 = 6.0 * lam;
        const double L = std::log(m2abs / T) - 1.0;
        // dcw = dm2*m2/(64 pi^2)*2*L  ->  product rule
        const double pref = 1.0 / (64.0 * M_PI * M_PI) * 2.0;
        const double term1 = (d2m2 * m2 + dm2 * dm2) * L;
        const double term2 = dm2 * m2 * (dm2 / m2);  // m2*L' where L'=dm2/m2
        return pref * (term1 + term2);
    }

    Params params_;
    double q_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    size_t n_ = 0;
    double umin_ = 0.0, umax_ = 0.0, du_ = 0.0, invdu_ = 0.0;
    std::vector<double> Jb_, Jf_, dJb_, dJf_, d2Jb_, d2Jf_;
};

}  // namespace ThermalInflation

#endif  // THERMAL_TABLES_HPP
