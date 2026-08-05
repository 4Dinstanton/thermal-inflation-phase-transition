#ifndef THERMAL_INFLATION_V2_H
#define THERMAL_INFLATION_V2_H

/* Thermal-inflation model for CosmoLattice v2.0.
 *
 * Port of cosmolattice_ext/models/thermal_inflation.h. Physics is unchanged:
 * one or two real scalars (real flaton, or phi1 + i phi2 for a global U(1)
 * flaton) with the full finite-temperature effective potential
 *
 *     V(phi,T) = V_tree + V_thermal + V_radiation + V_CW
 *
 * from the tabulated J-integrals, a prescribed background T(a) and H(a) with
 * ti -> md -> rd staging, and Langevin friction + FDT noise supplied by
 * evolvers/stochasticrk.h.
 *
 * v1 -> v2 differences (see cosmolattice_ext_v2/README.md for the full list):
 *
 *  - Constructor takes `auto toolBox` instead of
 *    `std::shared_ptr<MemoryToolBox>` (the one-line change documented in
 *    CL 2.0 Sect. 5).
 *  - Every per-site raw loop is gone. TempLat v1.0.0 removed `itX()`,
 *    `Field::get(ptrdiff_t)` and `getSet(ptrdiff_t)` because fields now live in
 *    Kokkos views. All initial conditions and diagnostics are therefore written
 *    as lattice expressions, which is also what makes them MPI- and
 *    GPU-correct by construction:
 *      * numba-style white-noise IC -> RandomGaussianFieldConfig
 *      * uniform / frozen components -> plain field assignment
 *      * cubic bubble seed          -> SpatialCoordinate + heaviside mask
 *      * false-vacuum fraction      -> average(heaviside(...)), which already
 *                                      carries out the global MPI reduction
 *        (v1 needed a hand-written MPI_Allreduce here).
 *  - The custom `.raw` snapshot writer is kept (v2 host-view port of the v1
 *    writer) so `--steps` / `--phi_threshold` / `--steps_dense` still work and
 *    `tools/export_cl_snapshots.py` stays usable. Upstream HDF5 snapshots
 *    (`snapshots = S`) remain available as an optional extra.
 */

#include "CosmoInterface/cosmointerface.h"

#include "TempLat/lattice/algebra/coordinates/spatialcoordinate.h"
#include "TempLat/lattice/algebra/random/randomgaussianfield.h"

#include "thermal_force.h"
#include "thermal_tables.hpp"
#include "field_snapshot.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace TempLat {

    struct ModelPars : public TempLat::DefaultModelPars {
        static constexpr size_t NScalars = 2;
        static constexpr size_t NPotTerms = 1;
        // Opts into the v2 defect module. With Ns == 2 and NCs == 0 this makes
        // DefectsMeasurer emit `winfindLengthStrings` -- the total comoving string
        // length from plaquette winding numbers of Complexify(fldS0, fldS1) -- into
        // defects.txt whenever `measureDefectsStructure = true`, plus norm.txt with
        // <|phi|> and its variance. That is the in-simulation replacement for
        // tools/compute_strings_cl.py, which had to re-derive windings from exported
        // snapshots. resolutionPreservingFactor stays at 1 unless a fixed background
        // with fattening is requested, so the dynamics are unchanged.
        static constexpr bool DefectsModel = true;
    };

#define MODELNAME thermal_inflation_v2

    template <class R>
    using Model = MakeModel(R, ModelPars);

    class MODELNAME : public Model<MODELNAME> {
    private:
        double lambda, mphi;
        double yb, gb, yf, gf;
        double mb2_0;
        double nb, nf;
        double gStarPot;
        double T0_;

        ThermalInflation::ThermalTables thermalTables;
        ThermalContext thermalCtx;

    public:
        // ---- Langevin / temperature parameters exposed to the evolver --------
        double T0() const { return T0_; }
        double muScale() const { return mphi; }    // = omegaStar
        double fStarVal() const { return fStar; }  // field rescaling, for FDT noise
        double etaPhys = 0.0;                      // friction (GeV); default = T0
        double dxPhys = 1e-3;                      // physical spacing (GeV^-1)
        double gStarHubble = 106.75;               // g_* in the Friedmann radiation term
        double delV = 0.0;                         // vacuum energy dV (GeV^4) for H
        bool   includeCW = true;
        bool   thermalNoise = true;
        bool   icNumba = false;
        double uniformPhiGeV = 0.0;
        double bubbleSeedPhiGeV = 0.0;
        double bubbleSeedBgGeV = 0.0;
        int    bubbleSeedRadius = 0;
        std::string stochasticScheme = "ou";       // "ou" | "verlet" | "numba" (half-FDT)
        int    nScalars_ = 1;
        double znOrder_ = 0.0;
        double znStrength_ = 0.0;
        double znTurnOnT_ = 0.0;
        std::string baseSeed_ = "thermal";         // RNG stream label for the FDT noise
        ThermalInflation::FieldSnapshotWriter snapshotWriter_;

        // Optional: zero Langevin eta + FDT noise once conversion begins.
        bool   langevinOffAfterNucleation_ = false;
        double langevinOffFSwitch_ = 0.99;
        double langevinOffPhiEsc_ = -1.0;
        bool   langevinDisabled_ = false;
        double etaPhysSaved_ = 0.0;
        bool   thermalNoiseSaved_ = true;

        // Post-PT expansion staging: ti (thermal inflation) -> md -> rd.
        enum class ExpansionStage { TI = 0, MD = 1, RD = 2 };
        bool expansionStaged_ = false;
        ExpansionStage expansionStage_ = ExpansionStage::TI;
        double expansionTSwitch_ = 0.0;
        double expansionFSwitch_ = 1e-5;
        double expansionPhiEsc_ = 1e4;
        double TRh_ = 0.0;
        double aSwitch_ = 1.0;
        double TSwitch_ = 0.0;
        double rhoMSwitch_ = 0.0;
        double aRh_ = 1.0;
        double TRhAnchor_ = 0.0;

        int activeScalars() const { return nScalars_; }
        ExpansionStage expansionStage() const { return expansionStage_; }
        int expansionStageId() const { return static_cast<int>(expansionStage_); }
        bool expansionStaged() const { return expansionStaged_; }

        double rhoMatter() const {
            if (expansionStage_ != ExpansionStage::MD || aI <= 0.0 || aSwitch_ <= 0.0) return 0.0;
            const double ratio = aSwitch_ / aI;
            return rhoMSwitch_ * ratio * ratio * ratio;
        }

        double delVForHubble() const {
            if (expansionStaged_ && expansionStage_ != ExpansionStage::TI) return 0.0;
            return delV;
        }

        // ---- Initial conditions, as lattice expressions ----------------------

        /** Zero the second component when running a single real scalar. */
        void freezeInactiveScalars() {
            if (nScalars_ >= 2) return;
            fldS(1_c) = 0.;
            piS(1_c) = 0.;
        }

        /** numba-style white-noise IC: phi = 0.01 GeV * N(0,1) per site, pi = 0.
         *  RandomGaussianFieldConfig is a counter-based per-site generator keyed on
         *  the GLOBAL coordinate, so the realization does not depend on the MPI
         *  decomposition. This replaces the hand-rolled hash RNG of the v1 model. */
        void applyNumbaInitialConditions() {
            if (!icNumba) return;
            using RGFC = RandomGaussianFieldConfig<double, Model<MODELNAME>::NDim>;
            const double amp = (nScalars_ >= 2) ? (0.01 / fStar) / std::sqrt(2.0)
                                                : (0.01 / fStar);
            fldS(0_c) = amp * RGFC(baseSeed_ + "_ic0", getToolBox());
            piS(0_c) = 0.;
            if (nScalars_ >= 2) {
                fldS(1_c) = amp * RGFC(baseSeed_ + "_ic1", getToolBox());
                piS(1_c) = 0.;
            }
            freezeInactiveScalars();
        }

        /** Homogeneous roll test: all sites phi = uniformPhiGeV, pi = 0. */
        void applyUniformPhi() {
            if (uniformPhiGeV <= 0.0) return;
            const double phiProg = uniformPhiGeV / fStar;
            ForLoop(n, 0, ModelPars::NScalars - 1,
                fldS(n) = phiProg;
                piS(n) = 0.;
            );
        }

        /** Cubic patch of true vacuum at the coordinate origin (roll / nucleation
         *  diagnostics). Built as a product of heaviside masks on the signed
         *  spatial coordinates rather than by searching for the centre site. */
        void applyBubbleSeed() {
            if (bubbleSeedPhiGeV <= 0.0) return;
            constexpr size_t ND = Model<MODELNAME>::NDim;
            SpatialCoordinate<ND> x(getToolBox());
            const double half = static_cast<double>(bubbleSeedRadius < 0 ? 0 : bubbleSeedRadius) + 0.5;

            const double bgProg = bubbleSeedBgGeV / fStar;
            const double hotProg = bubbleSeedPhiGeV / fStar;

            if (bubbleSeedBgGeV > 0.0) {
                fldS(0_c) = bgProg;
                piS(0_c) = 0.;
            }
            // Product over dimensions of heaviside(half - |x_i|).
            if constexpr (ND == 3) {
                fldS(0_c) = fldS(0_c) +
                            (hotProg - fldS(0_c)) * heaviside(half - abs(x(1_c))) *
                                heaviside(half - abs(x(2_c))) * heaviside(half - abs(x(3_c)));
            } else if constexpr (ND == 2) {
                fldS(0_c) = fldS(0_c) + (hotProg - fldS(0_c)) * heaviside(half - abs(x(1_c))) *
                                            heaviside(half - abs(x(2_c)));
            } else {
                fldS(0_c) = fldS(0_c) + (hotProg - fldS(0_c)) * heaviside(half - abs(x(1_c)));
            }
            piS(0_c) = 0.;
        }

        // ---- Background: T(a), H(a), staging ---------------------------------

        double temperatureAtScaleFactor(double a) const {
            if (a <= 0.0) a = 1e-30;
            if (!expansionStaged_ || expansionStage_ == ExpansionStage::TI) return T0_ / a;
            if (expansionStage_ == ExpansionStage::MD) {
                return TSwitch_ * std::pow(aSwitch_ / a, 1.5);
            }
            return TRhAnchor_ * (aRh_ / a);
        }

        void updateTemperature(double a) { thermalCtx.T = temperatureAtScaleFactor(a); }
        void setCurrentTemperature(double T) { thermalCtx.T = T; }
        double currentT() const { return thermalCtx.T; }

        double prescribedHubble() const {
            const double M_PL = 2.4e18;
            const double chig2 = 30.0 / (M_PI * M_PI * gStarHubble);
            double rho = 0.0;
            if (expansionStaged_ && expansionStage_ == ExpansionStage::MD) {
                rho = rhoMatter();
            } else {
                rho = std::pow(thermalCtx.T, 4) / chig2 + delVForHubble();
            }
            const double H2 = rho / (3.0 * M_PL * M_PL);
            return std::sqrt(H2 > 0 ? H2 : 0.0);
        }

        /** Fraction of the volume still in the false vacuum (|phi| <= escapeGeV).
         *  average() performs the global (MPI) reduction internally. */
        double falseVacuumFraction(double escapeGeV) {
            const double escProg = escapeGeV / fStar;
            const double esc2 = escProg * escProg;
            if (nScalars_ >= 2) {
                return average(heaviside(esc2 - pow<2>(fldS(0_c)) - pow<2>(fldS(1_c))));
            }
            return average(heaviside(esc2 - pow<2>(fldS(0_c))));
        }

        /** Drop Langevin friction + FDT noise once conversion starts. Does not
         *  change V(phi,T) or the thermodynamic T_c1; the Hubble 3H term stays. */
        void maybeDisableLangevin() {
            if (!langevinOffAfterNucleation_ || langevinDisabled_) return;
            const double esc = (langevinOffPhiEsc_ > 0.0)
                                   ? langevinOffPhiEsc_
                                   : (expansionPhiEsc_ > 0.0 ? expansionPhiEsc_ : 1e4);
            const double fFalse = falseVacuumFraction(esc);
            if (fFalse > langevinOffFSwitch_) return;

            langevinDisabled_ = true;
            etaPhysSaved_ = etaPhys;
            thermalNoiseSaved_ = thermalNoise;
            etaPhys = 0.0;
            thermalNoise = false;
            if (getToolBox()->amIRoot()) {
                std::cout << "\n*** Langevin OFF (collision/GW stage): false-vac frac=" << fFalse
                          << " <= " << langevinOffFSwitch_ << " (was eta_phys=" << etaPhysSaved_
                          << " GeV, thermal_noise=" << (thermalNoiseSaved_ ? 1 : 0) << ") ***\n\n";
            }
        }

        bool langevinDisabled() const { return langevinDisabled_; }

        void maybeAdvanceExpansionStage() {
            if (!expansionStaged_) return;

            if (expansionStage_ == ExpansionStage::TI) {
                const bool enterMD = (expansionTSwitch_ > 0.0)
                                         ? (thermalCtx.T <= expansionTSwitch_)
                                         : (falseVacuumFraction(expansionPhiEsc_) <= expansionFSwitch_);
                if (enterMD) {
                    expansionStage_ = ExpansionStage::MD;
                    aSwitch_ = aI > 0.0 ? aI : 1.0;
                    TSwitch_ = thermalCtx.T;
                    rhoMSwitch_ = delV > 0.0 ? delV : 0.0;
                    updateTemperature(aI);
                    if (getToolBox()->amIRoot()) {
                        std::cout << "\n*** expansion stage TI->MD at a=" << aSwitch_ << " T=" << TSwitch_
                                  << " GeV rho_m=" << rhoMSwitch_ << " GeV^4 ***\n\n";
                    }
                }
                return;
            }

            if (expansionStage_ == ExpansionStage::MD && TRh_ > 0.0 && thermalCtx.T <= TRh_) {
                const double H_before = prescribedHubble();
                expansionStage_ = ExpansionStage::RD;
                aRh_ = aI > 0.0 ? aI : 1.0;
                TRhAnchor_ = TRh_;
                thermalCtx.T = TRhAnchor_;
                if (getToolBox()->amIRoot()) {
                    std::cout << "\n*** expansion stage MD->RD at a=" << aRh_ << " T_reh=" << TRhAnchor_
                              << " GeV H_md=" << H_before << " H_rd=" << prescribedHubble() << " ***\n\n";
                }
            }
        }

        const ThermalContext& thermalContext() const { return thermalCtx; }

        /** Called every lattice step from the patched main loop (dense switching). */
        void saveFieldSnapshotIfDue(int nStep, double tNow) {
            snapshotWriter_.maybeSave(*this, nStep, tNow);
        }

        MODELNAME(ParameterParser& parser, RunParameters<double>& runPar, auto toolBox)
            : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt,
                               STRINGIFY(MODELLABEL)) {
            mphi = parser.get<double>("mphi", 1000.0);
            const double gamma = parser.get<double>("gamma", -1.0);
            const double M_PL = 2.4e18;
            if (gamma > 0) {
                const double phi0 = gamma * M_PL;
                lambda = mphi * mphi / (phi0 * phi0);
                delV = 0.25 * lambda * phi0 * phi0 * phi0 * phi0;
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

            // ParameterParser::get returns a lazy ParameterGetter<T> in v2; bind it to a
            // concrete local before comparing or doing arithmetic on it.
            const std::string potType = parser.get<std::string>("potential_type", "V_correct");
            if (potType == "fermion_only") nb = 0.0;

            T0_ = parser.get<double>("T0", 7350.0);
            etaPhys = parser.get<double>("eta_phys", T0_);
            dxPhys = parser.get<double>("dx_phys", 1e-3);
            includeCW = parser.get<int>("include_cw", 1) != 0;
            thermalNoise = parser.get<int>("thermal_noise", 1) != 0;
            langevinOffAfterNucleation_ = parser.get<int>("langevin_off_after_nucleation", 0) != 0;
            langevinOffFSwitch_ = parser.get<double>("langevin_off_f_switch", 0.99);
            langevinOffPhiEsc_ = parser.get<double>("langevin_off_phi_esc", -1.0);
            langevinDisabled_ = false;
            etaPhysSaved_ = etaPhys;
            thermalNoiseSaved_ = thermalNoise;
            icNumba = parser.get<int>("ic_numba", 0) != 0;
            uniformPhiGeV = parser.get<double>("uniform_phi", 0.0);
            bubbleSeedPhiGeV = parser.get<double>("bubble_seed_phi", 0.0);
            bubbleSeedBgGeV = parser.get<double>("bubble_seed_bg", 0.0);
            bubbleSeedRadius = parser.get<int>("bubble_seed_radius", 0);
            stochasticScheme = parser.get<std::string>("stochastic_scheme", "ou");
            const int nScalarsIn = parser.get<int>("n_scalars", 1);
            nScalars_ = nScalarsIn < 1 ? 1 : (nScalarsIn > 2 ? 2 : nScalarsIn);
            znOrder_ = parser.get<double>("zn_order", 0.0);
            znStrength_ = parser.get<double>("zn_strength", 0.0);
            znTurnOnT_ = parser.get<double>("zn_turn_on_T", 0.0);

            const std::string mode = parser.get<std::string>("expansion_mode", "legacy");
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

            const std::string tablePath =
                parser.get<std::string>("thermal_table", "../../data/thermal_splines/thermal_tables.bin");

            fldS0 = parser.get<double, 2>("initial_amplitudes", {0.0, 0.0});
            piS0 = parser.get<double, 2>("initial_momenta", {0.0, 0.0});

            alpha = 1;
            fStar = (std::abs(fldS0[0]) > 0 ? std::abs(fldS0[0]) : mphi / std::sqrt(lambda));
            omegaStar = mphi;

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
        // Program potential: one combined term (tree + thermal + radiation + CW),
        // always evaluated at rho = sqrt(phi1^2 + phi2^2) so that Energies::rho is
        // correct for n_scalars = 2; for n_scalars = 1 the inactive phi2 stays 0.
        /////////
        auto potentialTerms(Tag<0>) { return thermalVrho(fldS(0_c), fldS(1_c), &thermalCtx); }

        auto potDeriv(Tag<0>) { return thermalVprime1(fldS(0_c), fldS(1_c), &thermalCtx); }
        auto potDeriv(Tag<1>) { return thermalVprime2(fldS(0_c), fldS(1_c), &thermalCtx); }

        auto potDeriv2(Tag<0>) { return thermalVsecond(fldS(0_c), &thermalCtx); }
        auto potDeriv2(Tag<1>) { return thermalVsecond(fldS(1_c), &thermalCtx); }
    };

}  // namespace TempLat

#endif  // THERMAL_INFLATION_V2_H
