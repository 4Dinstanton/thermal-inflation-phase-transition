#ifndef THERMAL_INFLATION_H
#define THERMAL_INFLATION_H

/* Thermal-inflation model for CosmoLattice.
 *
 * A single real scalar phi with the FULL finite-temperature effective potential
 *
 *     V(phi,T) = V_tree(phi) + V_thermal(phi,T) + V_radiation(T) + V_CW(phi,T)
 *
 * matching potential/Potential.py (V_correct / V_fermion_only). The thermal part
 * uses the tabulated J-integrals (thermal_tables.hpp) produced by
 * tools/export_thermal_splines.py.
 *
 * Two modes via the "potential_type" parameter:
 *   - "V_correct"    : boson + fermion   (nb, nf both > 0)
 *   - "fermion_only" : nb forced to 0
 *
 * Temperature evolution is inflation-style, T = T0 / a, applied each step by the
 * custom stochastic evolver (cosmolattice_ext/evolvers/stochasticrk.h), which
 * also adds Langevin friction + FDT thermal noise. With the standard CosmoLattice
 * evolvers (VV/LF) this model still runs as a deterministic classical field with
 * T = T0/a, but without thermal noise.
 *
 * Program variables (see manual Sec. 4.1):
 *   fStar     = phi0 = initial_amplitudes[0]    (field scale; ~ tree VEV phi0 = gamma*M_Pl)
 *   omegaStar = mphi                            (= numba mu; sets dx = omega*dx_phys etc.)
 *   alpha     = 1
 */

#include "CosmoInterface/cosmointerface.h"

#include "thermal_tables.hpp"
#include "thermal_force.h"
#include "field_snapshot.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "TempLat/parallel/mpi/mpitypeconstants.h"
#ifndef NOMPI
#include <mpi.h>
#endif

namespace TempLat {

    struct ModelPars : public TempLat::DefaultModelPars {
        static constexpr size_t NScalars = 2;
        static constexpr size_t NPotTerms = 1;
    };

#define MODELNAME thermal_inflation

    template <class R>
    using Model = MakeModel(R, ModelPars);

    class MODELNAME : public Model<MODELNAME> {
    private:
        // Model-specific parameters.
        double lambda, mphi;
        double yb, gb, yf, gf;        // couplings (boson Yukawa/gauge, fermion Yukawa/gauge)
        double mb2_0;                 // bosonMassSquared
        double nb, nf;                // multiplicities
        double gStarPot;             // radiation free-energy multiplicity (Potential.py: 100)
        double T0_;                   // initial temperature (GeV)

        ThermalInflation::ThermalTables thermalTables;  // loaded once
        ThermalContext thermalCtx;                      // shared with the operators
        ThermalInflation::FieldSnapshotWriter snapshotWriter_;

    public:
        // ---- Langevin / temperature parameters exposed to the evolver --------
        double T0() const { return T0_; }
        double muScale() const { return mphi; }   // = omegaStar
        double fStarVal() const { return fStar; } // field rescaling, for FDT noise
        double etaPhys = 0.0;                       // friction (GeV); default = T0 (set below)
        double dxPhys = 1e-3;                       // physical spacing (GeV^-1), for FDT noise
        double gStarHubble = 106.75;               // g_* in the Friedmann radiation term
        double delV = 0.0;                          // vacuum energy ΔV (GeV^4) for H
        bool   includeCW = true;
        bool   thermalNoise = true;                 // enable FDT noise in stochastic evolver
        bool   icNumba = false;                     // replace kCutOff IC with phi=0.01 GeV white, pi=0
        double uniformPhiGeV = 0.0;                 // set all sites to this phi (GeV), pi=0
        double bubbleSeedPhiGeV = 0.0;              // cubic patch seed at lattice centre (GeV)
        double bubbleSeedBgGeV = 0.0;               // background phi outside patch (GeV); 0 = leave IC
        int    bubbleSeedRadius = 0;                // patch half-width in lattice cells (0 = centre site only)
        std::string stochasticScheme = "numba";     // "numba" (default) or "fused"
        int    nScalars_ = 1;                       // 1 = real scalar; 2 = complex phi1+i*phi2
        double znOrder_ = 0.0;                      // Z_N breaking order (0 = pure U(1))
        double znStrength_ = 0.0;                   // delta_V for cos(N theta) term
        double znTurnOnT_ = 0.0;                    // activate Z_N below this T (GeV)

        // Post-PT expansion staging: ti (thermal inflation) → md → rd.
        // legacy mode keeps H(T,delV) and T=T0/a forever.
        enum class ExpansionStage { TI = 0, MD = 1, RD = 2 };
        bool expansionStaged_ = false;
        ExpansionStage expansionStage_ = ExpansionStage::TI;
        double expansionTSwitch_ = 0.0;   // >0: enter MD when T <= this
        double expansionFSwitch_ = 1e-5;  // else enter MD when false-vac frac <= this
        double expansionPhiEsc_ = 1e4;    // |phi|/rho escape threshold (GeV)
        double TRh_ = 0.0;               // post–flaton-decay T_reh (GeV); 0 = stay in MD
        double aSwitch_ = 1.0;           // scale factor at ti→md
        double TSwitch_ = 0.0;           // bath T at ti→md
        double rhoMSwitch_ = 0.0;        // matter density at ti→md (= delV dump)
        double aRh_ = 1.0;               // scale factor at md→rd
        double TRhAnchor_ = 0.0;         // bath T after md→rd (= T_rh)

        int activeScalars() const { return nScalars_; }
        ExpansionStage expansionStage() const { return expansionStage_; }
        int expansionStageId() const { return static_cast<int>(expansionStage_); }
        bool expansionStaged() const { return expansionStaged_; }
        double rhoMatter() const {
            if (expansionStage_ != ExpansionStage::MD || aI <= 0.0 || aSwitch_ <= 0.0) {
                return 0.0;
            }
            const double ratio = aSwitch_ / aI;
            return rhoMSwitch_ * ratio * ratio * ratio;
        }
        double delVForHubble() const {
            // Vacuum energy contributes to H only during the TI / legacy stage.
            if (expansionStaged_ && expansionStage_ != ExpansionStage::TI) return 0.0;
            return delV;
        }
        bool znActiveNow() const {
            if (znOrder_ <= 0.0 || znStrength_ <= 0.0) return false;
            if (znTurnOnT_ <= 0.0) return true;
            return thermalCtx.T <= znTurnOnT_;
        }

        void vPrimeComponentGeV(double phi1GeV, double phi2GeV, int comp, double& dV) const {
            double dV1 = 0.0, dV2 = 0.0;
            thermalTables.vPrimeComponents(phi1GeV, phi2GeV, thermalCtx.T,
                                           znOrder_, znStrength_, znActiveNow(),
                                           dV1, dV2);
            dV = (comp == 0 ? dV1 : dV2);
        }

        // getSet() does not mark ghosts stale (unlike Field::operator=). After any
        // manual write we must setGhostsAreStale() so the next confirmGhostsUpToDate()
        // refreshes periodic/MPI neighbours used by the Laplacian.
        void markScalarGhostsStale() {
            ForLoop(n, 0, ModelPars::NScalars - 1,
                fldS(n).setGhostsAreStale();
                piS(n).setGhostsAreStale();
            );
        }

        void freezeInactiveScalars() {
            if (nScalars_ >= 2) return;
            auto& it = getToolBox()->itX();
            for (it.begin(); it.end(); ++it) {
                fldS(1_c).getSet(it()) = 0.0;
                piS(1_c).getSet(it()) = 0.0;
            }
            markScalarGhostsStale();
        }

        // Flat global site index (MPI-safe). Do NOT use it() — local memory offsets
        // repeat across ranks and would duplicate ICs / noise patterns.
        uint64_t globalSiteIndex(const std::vector<ptrdiff_t>& c) {
            const ptrdiff_t N = getToolBox()->mNGridPointsVec[0];
            auto to0 = [N](ptrdiff_t x) -> uint64_t {
                return static_cast<uint64_t>(x >= 0 ? x : x + N);
            };
            const uint64_t ix = to0(c[0]);
            const uint64_t iy = to0(c.size() > 1 ? c[1] : 0);
            const uint64_t iz = to0(c.size() > 2 ? c[2] : 0);
            const uint64_t Nu = static_cast<uint64_t>(N);
            return (ix * Nu + iy) * Nu + iz;
        }

        static uint64_t hashMix64(uint64_t x) {
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27;
            x *= 0x94D049BB133111EBULL;
            x ^= x >> 31;
            return x;
        }

        static double hashGaussian(uint64_t seed) {
            const uint64_t h1 = hashMix64(seed);
            const uint64_t h2 = hashMix64(seed ^ 0x9E3779B97F4A7C15ULL);
            const double u1 = (static_cast<double>(h1 >> 11) + 1.0) * (1.0 / 9007199254740992.0);
            const double u2 = (static_cast<double>(h2 >> 11)) * (1.0 / 9007199254740992.0);
            const double r = std::sqrt(-2.0 * std::log(u1 > 1e-300 ? u1 : 1e-300));
            return r * std::cos(6.28318530717958647692 * u2);
        }

        // Replace CosmoLattice spectral ICs with numba-style white noise (call after initialize).
        // Real scalar: phi = 0.01 * randn.
        // Complex: independent components with amp/sqrt(2) so <rho^2> matches the real case
        // (same as latticeSimComplex_numba radial init with init_rho=0.01).
        // Seeded from GLOBAL site index so MPI ranks do not share identical slabs.
        void applyNumbaInitialConditions() {
            if (!icNumba) return;
            const double amp =
                (nScalars_ >= 2) ? (0.01 / fStar) / std::sqrt(2.0) : (0.01 / fStar);
            constexpr uint64_t kIcSeed = 0x4E554D42412D4943ULL; // "NUMBA-IC"
            auto& it = getToolBox()->itX();
            for (it.begin(); it.end(); ++it) {
                const uint64_t site = globalSiteIndex(it.getVec());
                const uint64_t seed0 = site ^ kIcSeed;
                fldS(0_c).getSet(it()) = amp * hashGaussian(seed0);
                piS(0_c).getSet(it()) = 0.0;
                if (nScalars_ >= 2) {
                    const uint64_t seed1 = site ^ 0x9E3779B97F4A7C15ULL ^ kIcSeed;
                    fldS(1_c).getSet(it()) = amp * hashGaussian(seed1);
                    piS(1_c).getSet(it()) = 0.0;
                }
            }
            if (nScalars_ < 2) {
                for (it.begin(); it.end(); ++it) {
                    fldS(1_c).getSet(it()) = 0.0;
                    piS(1_c).getSet(it()) = 0.0;
                }
            }
            markScalarGhostsStale();
        }

        // Uniform phi (homogeneous roll test): all sites phi=uniformPhiGeV, pi=0.
        void applyUniformPhi() {
            if (uniformPhiGeV <= 0.0) return;
            const double phiProg = uniformPhiGeV / fStar;
            auto& it = getToolBox()->itX();
            ForLoop(n, 0, ModelPars::NScalars - 1,
                for (it.begin(); it.end(); ++it) {
                    fldS(n).getSet(it()) = phiProg;
                    piS(n).getSet(it()) = 0.0;
                }
            );
            markScalarGhostsStale();
        }

        // Cubic patch seed at lattice centre (roll / nucleation diagnostics).
        void applyBubbleSeed() {
            if (bubbleSeedPhiGeV <= 0.0) return;
            auto tb = getToolBox();
            auto& it = tb->itX();

            std::vector<ptrdiff_t> centerCoord;
            ptrdiff_t best = -1;
            ptrdiff_t bestDist2 = std::numeric_limits<ptrdiff_t>::max();
            for (it.begin(); it.end(); ++it) {
                const auto coord = tb->getCoordConfiguration(it());
                ptrdiff_t d2 = 0;
                for (const auto v : coord) d2 += v * v;
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    best = it();
                    centerCoord = coord;
                }
            }
            if (best < 0) return;

            const double bgProg = bubbleSeedBgGeV / fStar;
            const double hotProg = bubbleSeedPhiGeV / fStar;
            const int R = bubbleSeedRadius < 0 ? 0 : bubbleSeedRadius;

            for (it.begin(); it.end(); ++it) {
                if (bubbleSeedBgGeV > 0.0) {
                    fldS(0_c).getSet(it()) = bgProg;
                    piS(0_c).getSet(it()) = 0.0;
                }
                const auto coord = tb->getCoordConfiguration(it());
                bool inside = true;
                for (size_t d = 0; d < coord.size(); ++d) {
                    const ptrdiff_t delta = coord[d] - centerCoord[d];
                    if (delta < -R || delta > R) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    fldS(0_c).getSet(it()) = hotProg;
                    piS(0_c).getSet(it()) = 0.0;
                }
            }
            markScalarGhostsStale();
        }

        // Stage-aware bath temperature from scale factor (does not mutate stage).
        //   ti / legacy: T = T0 / a
        //   md:          T = T_sw (a_sw / a)^{3/2}
        //   rd:          T = T_reh (a_rh / a)  with T_reh = --T_rh
        double temperatureAtScaleFactor(double a) const {
            if (a <= 0.0) a = 1e-30;
            if (!expansionStaged_ || expansionStage_ == ExpansionStage::TI) {
                return T0_ / a;
            }
            if (expansionStage_ == ExpansionStage::MD) {
                const double ratio = aSwitch_ / a;
                return TSwitch_ * std::pow(ratio, 1.5);
            }
            // RD after flaton decay: cool from the CLI reheating temperature.
            return TRhAnchor_ * (aRh_ / a);
        }

        // Update the temperature from the current scale factor.
        // Called by the evolver each step (and once at construction).
        void updateTemperature(double a) { thermalCtx.T = temperatureAtScaleFactor(a); }
        void setCurrentTemperature(double T) { thermalCtx.T = T; }
        double currentT() const { return thermalCtx.T; }

        // Prescribed Hubble matching StochasticRK (for snapshots / diagnostics).
        double prescribedHubble() const {
            const double M_PL = 2.4e18;
            const double T_now = thermalCtx.T;
            if (expansionStaged_ && expansionStage_ == ExpansionStage::MD) {
                const double rhoM = rhoMatter();
                const double H2 = rhoM / (3.0 * M_PL * M_PL);
                return std::sqrt(H2 > 0 ? H2 : 0.0);
            }
            if (expansionStaged_ && expansionStage_ == ExpansionStage::RD) {
                // Radiation-only H from bath T (= T_reh (a_rh/a)); not from ΔV.
                const double chig2 = 30.0 / (M_PI * M_PI * gStarHubble);
                const double H2 = std::pow(T_now, 4) / chig2 / (3.0 * M_PL * M_PL);
                return std::sqrt(H2 > 0 ? H2 : 0.0);
            }
            // TI / legacy: radiation + vacuum
            const double chig2 = 30.0 / (M_PI * M_PI * gStarHubble);
            const double H2 = (std::pow(T_now, 4) / chig2 + delVForHubble())
                            / (3.0 * M_PL * M_PL);
            return std::sqrt(H2 > 0 ? H2 : 0.0);
        }

        // Fraction of sites still in the false vacuum (|amp| <= escape GeV).
        // MPI-safe: Allreduce SUM of local false / total counts.
        double falseVacuumFraction(double escapeGeV) {
            const double escProg = escapeGeV / fStar;
            const double esc2 = escProg * escProg;
            double localFalse = 0.0;
            double localTotal = 0.0;
            auto& it = getToolBox()->itX();
            for (it.begin(); it.end(); ++it) {
                const double p1 = static_cast<double>(fldS(0_c).get(it()));
                double amp2 = p1 * p1;
                if (nScalars_ >= 2) {
                    const double p2 = static_cast<double>(fldS(1_c).get(it()));
                    amp2 += p2 * p2;
                }
                if (amp2 <= esc2) localFalse += 1.0;
                localTotal += 1.0;
            }
#ifndef NOMPI
            {
                double buf[2] = {localFalse, localTotal};
                MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM,
                              getToolBox()->mGroup.getBaseComm());
                localFalse = buf[0];
                localTotal = buf[1];
            }
#endif
            if (localTotal <= 0.0) return 1.0;
            return localFalse / localTotal;
        }

        // After a is advanced: possibly ti→md or md→rd. Safe to call every step.
        void maybeAdvanceExpansionStage() {
            if (!expansionStaged_) return;

            if (expansionStage_ == ExpansionStage::TI) {
                bool enterMD = false;
                if (expansionTSwitch_ > 0.0) {
                    enterMD = (thermalCtx.T <= expansionTSwitch_);
                } else {
                    enterMD = (falseVacuumFraction(expansionPhiEsc_) <= expansionFSwitch_);
                }
                if (enterMD) {
                    expansionStage_ = ExpansionStage::MD;
                    aSwitch_ = aI > 0.0 ? aI : 1.0;
                    TSwitch_ = thermalCtx.T;
                    rhoMSwitch_ = delV > 0.0 ? delV : 0.0;
                    updateTemperature(aI);
                    if (getToolBox()->amIRoot()) {
                        std::cout << "\n*** expansion stage TI→MD at a=" << aSwitch_
                                  << " T=" << TSwitch_ << " GeV"
                                  << " rho_m=" << rhoMSwitch_ << " GeV^4 ***\n\n";
                    }
                }
                return;
            }

            if (expansionStage_ == ExpansionStage::MD && TRh_ > 0.0) {
                if (thermalCtx.T <= TRh_) {
                    // Flaton decay / reheating: T_rh is the true reheating
                    // temperature (arXiv:0801.4197), not (ρ_m)^{1/4}. Enter RD
                    // with T = T_rh and H from ρ_r(T); do not dump ΔV → T.
                    const double H_before = prescribedHubble();
                    expansionStage_ = ExpansionStage::RD;
                    aRh_ = aI > 0.0 ? aI : 1.0;
                    TRhAnchor_ = TRh_;
                    thermalCtx.T = TRhAnchor_;
                    const double H_after = prescribedHubble();
                    if (getToolBox()->amIRoot()) {
                        std::cout << "\n*** expansion stage MD→RD at a=" << aRh_
                                  << " T_reh=" << TRhAnchor_ << " GeV"
                                  << " H_md=" << H_before
                                  << " H_rd=" << H_after << " ***\n\n";
                    }
                }
            }
        }

        // RK2 scratch buffers for the Numba-parity evolver (not measured/snapshotted).
        FieldCollection<Field, double, 2, true> tmpFldS;
        FieldCollection<Field, double, 2, true> tmpPiS;

        // Convenience access used by the stochastic evolver.
        const ThermalContext& thermalContext() const { return thermalCtx; }

        double vPrimeGeV(double phiGeV) const {
            return thermalTables.Vprime(phiGeV, thermalCtx.T);
        }

        // Called from the patched Measurer each step (see run_cosmolattice --install).
        void saveFieldSnapshotIfDue(int n, double t) {
            snapshotWriter_.maybeSave(*this, n, t);
        }

        // Power-spectrum measurements FFT fldS/piS in place and leave ghost cells stale.
        void refreshFieldsAfterMeasurement() {
            ForLoop(n, 0, ModelPars::NScalars - 1,
                fldS(n).confirmGhostsUpToDate();
                piS(n).confirmGhostsUpToDate();
                tmpFldS(n).confirmGhostsUpToDate();
                tmpPiS(n).confirmGhostsUpToDate();
            );
            if (fldGWs != nullptr) {
                ForLoop(i, 0, NGWs - 1,
                    (*fldGWs)(i).confirmGhostsUpToDate();
                    (*piGWs)(i).confirmGhostsUpToDate();
                );
            }
        }

        MODELNAME(ParameterParser& parser, RunParameters<double>& runPar,
                  std::shared_ptr<MemoryToolBox> toolBox)
            : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt,
                               STRINGIFY(MODELLABEL)),
              tmpFldS("tmp_scalar", toolBox, runPar.getLatParams()),
              tmpPiS("tmp_pi_scalar", toolBox, runPar.getLatParams()) {
            /////////
            // Independent parameters
            /////////
            mphi = parser.get<double>("mphi", 1000.0);
            // lambda may be given directly or via gamma (phi0 = gamma*M_Pl, lambda = mphi^2/phi0^2)
            double gamma = parser.get<double>("gamma", -1.0);
            const double M_PL = 2.4e18;
            if (gamma > 0) {
                const double phi0 = gamma * M_PL;
                lambda = mphi * mphi / (phi0 * phi0);
                delV = 0.25 * lambda * phi0 * phi0 * phi0 * phi0;  // V0 = lam/4 phi0^4
            } else {
                lambda = parser.get<double>("lambda");
            }
            delV = parser.get<double>("delV", delV);

            yb = parser.get<double>("boson_coupling", 1.09);
            gb = parser.get<double>("boson_gauge_coupling", 1.05);
            yf = parser.get<double>("fermion_coupling", 1.09);
            gf = parser.get<double>("fermion_gauge_coupling", 1.05);
            mb2_0 = parser.get<double>("boson_mass_squared", 1.0e6);
            nb = parser.get<double>("nb", 20.0);
            nf = parser.get<double>("nf", 20.0);
            gStarPot = parser.get<double>("g_star_pot", 100.0);
            gStarHubble = parser.get<double>("g_star_hubble", 106.75);

            std::string potType = parser.get<std::string>("potential_type", "V_correct");
            if (potType == "fermion_only") nb = 0.0;

            T0_ = parser.get<double>("T0", 7350.0);
            etaPhys = parser.get<double>("eta_phys", T0_);  // default eta = T0 (numba convention)
            dxPhys = parser.get<double>("dx_phys", 1e-3);
            includeCW = parser.get<int>("include_cw", 1) != 0;
            thermalNoise = parser.get<int>("thermal_noise", 1) != 0;
            icNumba = parser.get<int>("ic_numba", 0) != 0;
            uniformPhiGeV = parser.get<double>("uniform_phi", 0.0);
            bubbleSeedPhiGeV = parser.get<double>("bubble_seed_phi", 0.0);
            bubbleSeedBgGeV = parser.get<double>("bubble_seed_bg", 0.0);
            bubbleSeedRadius = parser.get<int>("bubble_seed_radius", 0);
            stochasticScheme = parser.get<std::string>("stochastic_scheme", "numba");
            nScalars_ = parser.get<int>("n_scalars", 1);
            if (nScalars_ < 1) nScalars_ = 1;
            if (nScalars_ > 2) nScalars_ = 2;
            znOrder_ = parser.get<double>("zn_order", 0.0);
            znStrength_ = parser.get<double>("zn_strength", 0.0);
            znTurnOnT_ = parser.get<double>("zn_turn_on_T", 0.0);

            {
                const std::string mode =
                    parser.get<std::string>("expansion_mode", "legacy");
                expansionStaged_ = (mode == "staged" || mode == "STAGED");
                expansionTSwitch_ = parser.get<double>("expansion_T_switch", 0.0);
                expansionFSwitch_ = parser.get<double>("expansion_f_switch", 1e-5);
                expansionPhiEsc_ = parser.get<double>("expansion_phi_esc", 1e4);
                TRh_ = parser.get<double>("T_rh", 0.0);
                expansionStage_ = ExpansionStage::TI;
                aSwitch_ = 1.0;
                TSwitch_ = T0_;
                rhoMSwitch_ = 0.0;
                aRh_ = 1.0;
                TRhAnchor_ = TRh_;
            }

            std::string tablePath = parser.get<std::string>("thermal_table",
                                                            "../../data/thermal_splines/thermal_tables.bin");

            /////////
            // Initial homogeneous components
            /////////
            fldS0 = parser.get<double, 2>("initial_amplitudes", {0.0, 0.0});
            piS0 = parser.get<double, 2>("initial_momenta", {0.0, 0.0});

            /////////
            // Rescaling for program variables
            /////////
            alpha = 1;
            // fStar: use phi0 (tree VEV) if the homogeneous amplitude is ~0.
            const double phi0_scale = (std::abs(fldS0[0]) > 0 ? std::abs(fldS0[0])
                                                              : mphi / std::sqrt(lambda));
            fStar = phi0_scale;
            omegaStar = mphi;

            /////////
            // Set up thermal tables + shared operator context
            /////////
            ThermalInflation::ThermalTables::Params tp;
            tp.lam = lambda; tp.mphi = mphi; tp.mb2_0 = mb2_0;
            tp.yb = yb; tp.gb = gb; tp.yf = yf; tp.gf = gf;
            tp.nb = nb; tp.nf = nf; tp.gStarPot = gStarPot; tp.includeCW = includeCW;
            thermalTables.load(tablePath);
            thermalTables.setParams(tp);

            thermalCtx.tables = &thermalTables;
            thermalCtx.fStar = fStar;
            thermalCtx.omegaStar = omegaStar;
            thermalCtx.T = T0_;
            thermalCtx.znOrder = znOrder_;
            thermalCtx.znStrength = znStrength_;
            thermalCtx.znTurnOnT = znTurnOnT_;

            const bool saveSnapshots = parser.get<int>("save_snapshots", 0) != 0;
            const int snapshotSteps = parser.get<int>("snapshot_steps", 100000);
            const int snapshotStepsDense = parser.get<int>("snapshot_steps_dense", 0);
            const double phiThreshold = parser.get<double>("phi_threshold", -1.0);
            snapshotWriter_.configure(
                runPar.outFn, saveSnapshots,
                snapshotSteps, snapshotStepsDense, phiThreshold,
                fStar, runPar.N);

            setInitialPotentialAndMassesFromPotential();
        }

        /////////
        // Program potential (single combined term: tree + thermal + radiation + CW)
        /////////
        auto potentialTerms(Tag<0>) {
            return thermalV(fldS(0_c), &thermalCtx);
        }

        // First derivative wrt each scalar component (complex field via rho, theta).
        auto potDeriv(Tag<0>) {
            return thermalVprime1(fldS(0_c), fldS(1_c), &thermalCtx);
        }
        auto potDeriv(Tag<1>) {
            return thermalVprime2(fldS(0_c), fldS(1_c), &thermalCtx);
        }

        // Second derivative (diagonal approx on each component; used for IC masses).
        auto potDeriv2(Tag<0>) {
            return thermalVsecond(fldS(0_c), &thermalCtx);
        }
        auto potDeriv2(Tag<1>) {
            return thermalVsecond(fldS(1_c), &thermalCtx);
        }
    };

}  // namespace TempLat

#endif  // THERMAL_INFLATION_H
