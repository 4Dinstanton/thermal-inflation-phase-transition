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
 *   tree:     V_tree   = (lam/4) phi^4 - (mphi^2/2) phi^2
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
    };

    ThermalTables() = default;

    explicit ThermalTables(const std::string& binPath) { load(binPath); }

    ThermalTables(const std::string& binPath, const Params& p) : params_(p) {
        load(binPath);
    }

    void setParams(const Params& p) { params_ = p; }
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
        double v = 0.25 * p.lam * std::pow(phi, 4) - 0.5 * p.mphi * p.mphi * phi * phi;
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
        double vp = p.lam * std::pow(phi, 3) - p.mphi * p.mphi * phi;
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
        const double dub = Bb * phi / (T * T * ubs);
        const double duf = Bf * phi / (T * T * ufs);
        const double d2ub = Bb / (T * T * ubs) - (Bb * phi) * (Bb * phi) / (std::pow(T, 4) * std::pow(ubs, 3));
        const double d2uf = Bf / (T * T * ufs) - (Bf * phi) * (Bf * phi) / (std::pow(T, 4) * std::pow(ufs, 3));
        double vpp = 3.0 * p.lam * phi * phi - p.mphi * p.mphi;
        vpp += pref * (p.nb * (interp(d2Jb_, ub) * dub * dub + hermite(dJb_, d2Jb_, ub) * d2ub)
                     + p.nf * (interp(d2Jf_, uf) * duf * duf + hermite(dJf_, d2Jf_, uf) * d2uf));
        if (p.includeCW) vpp += d2cw(phi, T);
        return vpp;
    }

    // Dimensionless argument helpers (exposed for testing).
    double uBoson(double phi, double T) const {
        const Params& p = params_;
        const double m2 = p.mb2_0 + 0.5 * p.yb * p.yb * phi * phi
                          + (0.25 * p.yb * p.yb + 2.0 / 3.0 * p.gb * p.gb) * T * T;
        return std::sqrt(m2 > 0 ? m2 : 0.0) / T;
    }
    double uFermion(double phi, double T) const {
        const Params& p = params_;
        const double m2 = 0.5 * p.yf * p.yf * phi * phi + (1.0 / 6.0) * p.gf * p.gf * T * T;
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
    size_t n_ = 0;
    double umin_ = 0.0, umax_ = 0.0, du_ = 0.0, invdu_ = 0.0;
    std::vector<double> Jb_, Jf_, dJb_, dJf_, d2Jb_, d2Jf_;
};

}  // namespace ThermalInflation

#endif  // THERMAL_TABLES_HPP
