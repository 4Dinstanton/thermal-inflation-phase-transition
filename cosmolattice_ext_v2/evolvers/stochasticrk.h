#ifndef COSMOINTERFACE_EVOLVERS_STOCHASTICRK_V2_H
#define COSMOINTERFACE_EVOLVERS_STOCHASTICRK_V2_H

/* Langevin (stochastic) evolver for the thermal-inflation model, CosmoLattice v2.0.
 *
 * Solves
 *
 *     phi'' + (3H + eta) phi' - a^{-2} Lap phi + V_{,phi}(phi,T) = xi(x,t)
 *     <xi xi> = 2 eta T / (a^3 dx_com^3) delta(t - t')           [FDT]
 *
 * with T = T(a) and H = H(a) prescribed by the model (thermal inflation is
 * vacuum + radiation dominated, so the lattice field energy alone would give a
 * negligible H).
 *
 * Schemes, selected with `stochastic_scheme`:
 *
 *   ou (default)
 *       Exact Ornstein-Uhlenbeck friction+noise (Strang split around Verlet).
 *       Samples the requested T: <pi^2>/<pi^2>_eq = 1.
 *
 *   verlet
 *       Explicit Velocity-Verlet friction+noise (v1 "fused" / "fdt" amplitude).
 *       Bias: <pi^2>/<pi^2>_eq = 1/(1 - eta*dt/4)  (~1.22 at production dt).
 *
 *   numba
 *       Same integrator as `verlet`, but noise amplitude × 1/sqrt(2) so the
 *       injected variance matches v1's default `numba` scheme (half FDT).
 *       Expected: <pi^2>/<pi^2>_eq ≈ 0.61 at production dt=0.1.
 *       This is statistical parity with old v1 runs, NOT a bit-for-bit port of
 *       the 4-pass RK2 / hash RNG (TempLat v2 removed the per-site API).
 *
 * Keep dt <= 0.2 for these parameters (Verlet core, independent of scheme).
 */

#include <cmath>
#include <string>

#include "CosmoInterface/definitions/averages.h"
#include "CosmoInterface/definitions/fixedbackgroundexpansion.h"
#include "CosmoInterface/evolvers/kernels/gwskernels.h"
#include "CosmoInterface/evolvers/kernels/kernels.h"
#include "CosmoInterface/runparameters.h"
#include "TempLat/lattice/algebra/random/randomgaussianfield.h"
#include "TempLat/util/rangeiteration/for_in_range.h"

namespace TempLat {

    template <class Model>
    class StochasticLangevin {
    public:
        using T = typename Model::FloatType;
        static constexpr size_t NDim = Model::NDim;
        using NoiseField = RandomGaussianFieldConfig<T, NDim>;

        enum class Scheme { Verlet, OU };

        StochasticLangevin(Model& model, RunParameters<T>& rPar)
            : expansion(rPar.expansion),
              fixedBackground(rPar.fixedBackground),
              aBackground(model, rPar),
              scheme_(Scheme::OU),
              halfFdtNoise_(false),
              noise0_(model.baseSeed_ + "_noise0", model.getToolBox()),
              noise1_(model.baseSeed_ + "_noise1", model.getToolBox()) {
            model.updateTemperature(model.aI);
            const std::string& sch = model.stochasticScheme;
            // `numba` = verlet integrator + half-FDT noise (v1 default amplitude).
            if (sch == "numba" || sch == "NUMBA") {
                scheme_ = Scheme::Verlet;
                halfFdtNoise_ = true;
            } else if (sch == "verlet" || sch == "fused" || sch == "fdt") {
                scheme_ = Scheme::Verlet;
            }
            // else: keep OU (exact FDT)
        }

        void evolve(Model& model, T tMinust0) {
            model.updateTemperature(model.aI);
            model.maybeDisableLangevin();

            if (scheme_ == Scheme::OU) {
                ouHalfStep(model);
                verletStep(model, tMinust0, /*friction=*/false);
                model.updateTemperature(model.aI);
                ouHalfStep(model);
            } else {
                verletStep(model, tMinust0, /*friction=*/true);
            }
            evolveGWs(model);
        }

        void sync(Model& model, T tMinust0) {
            if (fixedBackground) model.aDotI = aBackground.dot(tMinust0);
        }

    private:
        // ---- Langevin coefficients (all in program units) --------------------

        T hubble(const Model& model) const { return static_cast<T>(model.prescribedHubble()); }

        T scaleFactor(const Model& model) const {
            if (!expansion || model.aI <= 0) return 1.0;
            return model.aI;
        }

        /* Coefficients in CosmoLattice program variables.
         *
         * CL evolves pi = a^(3-alpha) dphiTilde/dEta, and dEta = a^(-alpha) omegaStar dt,
         * so pi = a^3 phidot / (fStar omegaStar) whatever alpha is. Pushing the physical
         * Langevin equation
         *      phiddot + (3H + eta_phys) phidot - Lap phi / a^2 + V' = xi,
         *      Var[d phidot] = 2 eta_phys T dt / (a^3 dx_com^3)
         * through that change of variables gives
         *      d pi / dEta = kernel - a^alpha (eta_phys/omegaStar) pi + noise,
         *      Var[d pi]   = 2 a^(3+alpha) (eta_phys/omegaStar) T dEta / (dx_com^3 fStar^2 omegaStar^2).
         * Two things to note against the v1 evolver:
         *   - the 3H drag is NOT added by hand. It is already inside pi's a^3, exactly as
         *     for every stock CL evolver; adding it again double counts (harmless in
         *     practice here, since 3H/eta ~ 1e-4, but wrong).
         *   - the 1/omegaStar^2 is essential. Without it the injected noise is too large
         *     by omegaStar^2 = mphi^2 = 1e6 for the thermal-inflation parameters. */

        /** Program-time friction coefficient, a^alpha * eta_phys / omegaStar. */
        T etaEff(const Model& model) const {
            const T e = model.etaPhys / model.muScale();
            return e * std::pow(scaleFactor(model), model.alpha);
        }

        /** Equipartition variance of the program momentum pi at the current T:
         *      <phidot^2> = T / (a^3 dx_com^3)  ->  <pi^2> = a^3 T / (dx_com^3 fStar^2 omegaStar^2). */
        T piVarianceEq(const Model& model) const {
            const T dx3 = std::pow(model.dxPhys, 3);
            const T f = model.fStarVal();
            const T mu = model.muScale();
            const T a = scaleFactor(model);
            return model.currentT() * a * a * a / (dx3 * f * f * mu * mu);
        }

        /** Std. dev. of the momentum kick delivered by the noise over `dEta` of
         *  program time, for the explicit (Verlet) scheme: Var = 2 etaEff varEq dEta.
         *  With halfFdtNoise_ (v1 `numba` parity), multiply by 1/sqrt(2). */
        T explicitNoiseSigma(const Model& model, T dEta) const {
            if (!model.thermalNoise) return 0.0;
            const T v = 2.0 * etaEff(model) * piVarianceEq(model) * dEta;
            T sig = std::sqrt(v > 0 ? v : 0.0);
            if (halfFdtNoise_) sig *= static_cast<T>(1.0 / std::sqrt(2.0));
            return sig;
        }

        // ---- Steps -----------------------------------------------------------

        /** Exact OU half-step on the momenta: pi <- c pi + sqrt((1-c^2) var) z. */
        void ouHalfStep(Model& model) {
            const T e = etaEff(model);
            if (e <= 0.0) return;
            const T c = std::exp(-e * model.dt / 2.0);
            const T sig = model.thermalNoise ? std::sqrt(std::max<T>(0.0, (1.0 - c * c) * piVarianceEq(model)))
                                             : static_cast<T>(0.0);
            applyToActiveScalars(model, c, sig);
        }

        void applyToActiveScalars(Model& model, T c, T sig) {
            const int ns = model.activeScalars();
            model.piS(0_c) = c * model.piS(0_c) + sig * noise0_;
            if (ns > 1) model.piS(1_c) = c * model.piS(1_c) + sig * noise1_;
        }

        /** Velocity-Verlet step. With `friction` the damping -eta_eff*pi and the
         *  FDT kick are folded into each half kick (v1 "fused" scheme). */
        void verletStep(Model& model, T tMinust0, bool friction) {
            const T halfdt = model.dt / 2.0;
            const T e = friction ? etaEff(model) : static_cast<T>(0.0);
            const T sig = friction ? explicitNoiseSigma(model, halfdt) : static_cast<T>(0.0);

            kickScalars(model, halfdt, e, sig);
            if (expansion) advanceScaleFactor(model, model.dt, tMinust0);
            driftScalars(model, model.dt);

            model.updateTemperature(model.aI);
            const T e2 = friction ? etaEff(model) : static_cast<T>(0.0);
            const T sig2 = friction ? explicitNoiseSigma(model, halfdt) : static_cast<T>(0.0);
            kickScalars(model, halfdt, e2, sig2);
        }

        void kickScalars(Model& model, T dEta, T etaEffVal, T sigma) {
            const int ns = model.activeScalars();
            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < ns) {
                    model.piS(n) += dEta * ScalarSingletKernels::get(model, n);
                    if (etaEffVal > 0) model.piS(n) += (-dEta * etaEffVal) * model.piS(n);
                }
            );
            if (sigma > 0) {
                model.piS(0_c) += sigma * noise0_;
                if (ns > 1) model.piS(1_c) += sigma * noise1_;
            }
        }

        void driftScalars(Model& model, T dEta) {
            const int ns = model.activeScalars();
            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < ns) {
                    model.fldS(n) += pow(model.aSI, model.alpha - 3) * (dEta * model.piS(n));
                }
            );
        }

        /** The GW sector never sees the thermal noise: only the field anisotropic
         *  stress sources it. Kick-drift-kick with the same dt as the fields. */
        void evolveGWs(Model& model) {
            if (model.fldGWs == nullptr) return;
            kickGWs(model);
            (*model.fldGWs) += pow(model.aSI, model.alpha - 3) * (model.dt * (*model.piGWs));
            kickGWs(model);
        }

        void kickGWs(Model& model) { (*model.piGWs) += (model.dt / 2) * GWsKernels::get(model); }

        void advanceScaleFactor(Model& model, T dEta, T tMinust0) {
            model.aIM = model.aI;
            if (fixedBackground) {
                model.aI = aBackground(tMinust0 + dEta);
                model.aSI = aBackground(tMinust0 + dEta / 2.0);
                model.aDotI = aBackground.dot(tMinust0 + dEta);
                return;
            }
            const T Hprog = hubble(model) / model.muScale();
            model.aSI = model.aI * std::exp(Hprog * dEta / 2.0);
            model.aI = model.aI * std::exp(Hprog * dEta);
            model.updateTemperature(model.aI);
            model.maybeAdvanceExpansionStage();
            model.aDotI = model.aI * hubble(model) / model.muScale();
        }

        bool expansion;
        bool fixedBackground;
        FixedBackgroundExpansion<T> aBackground;
        Scheme scheme_;
        bool halfFdtNoise_;  // true for stochastic_scheme=numba
        NoiseField noise0_;
        NoiseField noise1_;
    };

}  // namespace TempLat

#endif  // COSMOINTERFACE_EVOLVERS_STOCHASTICRK_V2_H
