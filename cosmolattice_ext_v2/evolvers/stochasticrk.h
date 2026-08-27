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
 *       Statistical amplitude parity with old v1 runs; NOT the 4-pass RK2.
 *
 *   fused_rk2  (aliases: rk2, numba_rk2)
 *       Expression-based 4-pass fused RK2 matching the Numba / v1 staging:
 *         predictor mid from (fld,pi), corrector with force at mid,
 *         two half-steps per dt, independent noise each corrector with
 *         amplitude 0.5 * sigma_full (sigma_full^2 = 2 eta <pi^2>_eq dt).
 *       Same half-FDT convention as Numba; expected T_eff ≈ 0.61 at dt=0.1.
 *       Not bit-identical (Kokkos RNG vs hash Box-Muller).
 *
 *   nonfused_rk2  (aliases: rk2_nonfused, nonfused)
 *       Single full-dt RK2 (2 passes): predictor at n, corrector at midpoint,
 *       one FDT kick of duration dt (Numba rk2_nonfused). Full FDT amplitude,
 *       not the half-FDT of fused_rk2.
 *
 * Keep dt <= 0.2 for these parameters (conservative core stability).
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "CosmoInterface/definitions/averages.h"
#include "CosmoInterface/definitions/fixedbackgroundexpansion.h"
#include "CosmoInterface/evolvers/kernels/gwskernels.h"
#include "CosmoInterface/evolvers/kernels/kernels.h"
#include "CosmoInterface/runparameters.h"
#include "TempLat/lattice/algebra/random/randomgaussianfield.h"
#include "TempLat/lattice/field/collections/fieldcollection.h"
#include "TempLat/lattice/field/field.h"
#include "TempLat/util/rangeiteration/for_in_range.h"

namespace TempLat {

    template <class Model>
    class StochasticLangevin {
    public:
        using T = typename Model::FloatType;
        static constexpr size_t NDim = Model::NDim;
        using NoiseField = RandomGaussianFieldConfig<T, NDim>;
        using ScalarCollection = FieldCollection<Field<T, NDim>, Model::Ns, true>;

        enum class Scheme { Verlet, OU, FusedRK2, NonfusedRK2 };

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
            if (sch == "fused_rk2" || sch == "FUSED_RK2" || sch == "rk2" || sch == "RK2" ||
                sch == "numba_rk2" || sch == "NUMBA_RK2") {
                scheme_ = Scheme::FusedRK2;
                rk2Temps_ = std::make_unique<Rk2Temps>(model, rPar);
            } else if (sch == "nonfused_rk2" || sch == "NONFUSED_RK2" ||
                       sch == "rk2_nonfused" || sch == "RK2_NONFUSED" ||
                       sch == "nonfused" || sch == "NONFUSED") {
                scheme_ = Scheme::NonfusedRK2;
                rk2Temps_ = std::make_unique<Rk2Temps>(model, rPar);
            } else if (sch == "numba" || sch == "NUMBA") {
                // Legacy v2 alias: Verlet + half-FDT amplitude (not the RK2 staging).
                scheme_ = Scheme::Verlet;
                halfFdtNoise_ = true;
            } else if (sch == "verlet" || sch == "fused" || sch == "fdt") {
                scheme_ = Scheme::Verlet;
            }
            if (model.etaFollowsT_ && model.getToolBox()->amIRoot()) {
                std::cout << "eta_follows_T: eta_phys(t) = " << model.etaPhys
                          << " * T(t)/T0  (eta ~ T)\n";
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
            } else if (scheme_ == Scheme::FusedRK2) {
                fusedRk2Step(model, tMinust0);
            } else if (scheme_ == Scheme::NonfusedRK2) {
                nonfusedRk2Step(model, tMinust0);
            } else {
                verletStep(model, tMinust0, /*friction=*/true);
            }
            evolveGWs(model);
        }

        void sync(Model& model, T tMinust0) {
            if (fixedBackground) model.aDotI = aBackground.dot(tMinust0);
        }

    private:
        struct Rk2Temps {
            ScalarCollection midFld;
            ScalarCollection midPi;
            ScalarCollection endFld;
            Rk2Temps(Model& model, RunParameters<T>& rPar)
                : midFld("srk_midFld", model.getToolBox(), rPar.getLatParams()),
                  midPi("srk_midPi", model.getToolBox(), rPar.getLatParams()),
                  endFld("srk_endFld", model.getToolBox(), rPar.getLatParams()) {}
        };

        // ---- Langevin coefficients (all in program units) --------------------

        T hubble(const Model& model) const { return static_cast<T>(model.prescribedHubble()); }

        T scaleFactor(const Model& model) const {
            if (!expansion || model.aI <= 0) return 1.0;
            return model.aI;
        }

        /** Program-time friction coefficient, a^alpha * eta_phys(T) / omegaStar. */
        T etaEff(const Model& model) const {
            const T e = static_cast<T>(model.etaPhysNow()) / model.muScale();
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

        /** Std. dev. of the momentum kick over `dEta` for explicit Verlet.
         *  With halfFdtNoise_ (scheme=numba), multiply by 1/sqrt(2). */
        T explicitNoiseSigma(const Model& model, T dEta) const {
            if (!model.thermalNoise) return 0.0;
            const T v = 2.0 * etaEff(model) * piVarianceEq(model) * dEta;
            T sig = std::sqrt(v > 0 ? v : 0.0);
            if (halfFdtNoise_) sig *= static_cast<T>(1.0 / std::sqrt(2.0));
            return sig;
        }

        /** Numba / fused_rk2 half-kick: 0.5 * sigma_full with sigma_full^2 = 2 eta varEq dt. */
        T fusedHalfNoiseSigma(const Model& model) const {
            if (!model.thermalNoise) return 0.0;
            const T v = 2.0 * etaEff(model) * piVarianceEq(model) * model.dt;
            return static_cast<T>(0.5) * std::sqrt(v > 0 ? v : 0.0);
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

        /** Two half-steps of predictor-corrector RK2 with Numba noise staging.
         *  Scale factor is held fixed during the RK2 passes and advanced once at
         *  the end (matches latticeSimeRescale_numba / v1 numbaRK2). */
        void fusedRk2Step(Model& model, T tMinust0) {
            const T halfdt = model.dt / 2.0;
            const T a0 = model.aI;
            const T T_now = model.currentT();
            const T e0 = etaEff(model);
            // Noise from T at the start of the step (Numba fused_inline).
            const T halfNoise = fusedHalfNoiseSigma(model);

            T a1 = a0;
            T T_mid = T_now;
            if (expansion) {
                const T Hprog = hubble(model) / model.muScale();
                a1 = a0 * std::exp(Hprog * model.dt);
                T_mid = static_cast<T>(model.temperatureAtScaleFactor(a1));
            }

            model.setCurrentTemperature(T_now);
            fusedRk2Half(model, halfdt, e0, halfNoise);

            model.setCurrentTemperature(T_mid);
            const T e1 = etaEff(model);  // a still a0 during second half
            fusedRk2Half(model, halfdt, e1, halfNoise);

            if (expansion) {
                model.aIM = a0;
                if (fixedBackground) {
                    model.aI = aBackground(tMinust0 + model.dt);
                    model.aSI = aBackground(tMinust0 + model.dt / 2.0);
                    model.aDotI = aBackground.dot(tMinust0 + model.dt);
                } else {
                    model.aSI = std::sqrt(a0 * a1);
                    model.aI = a1;
                    model.updateTemperature(model.aI);
                    model.maybeAdvanceExpansionStage();
                    model.aDotI = model.aI * hubble(model) / model.muScale();
                }
            } else {
                model.updateTemperature(model.aI);
            }
        }

        void fusedRk2Half(Model& model, T halfdt, T e, T halfNoise) {
            auto& temps = *rk2Temps_;
            const int ns = model.activeScalars();
            const T driftFac = std::pow(model.aI, model.alpha - 3);

            // Pass 1 (predictor) at current (fld, pi).
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        temps.midPi(n) =
                            model.piS(n) +
                            halfdt * (ScalarSingletKernels::get(model, n) + (-e) * model.piS(n));
                        temps.midFld(n) = model.fldS(n) + halfdt * driftFac * model.piS(n);
                    });

            // End-of-half field from predictor momentum (needs old fld).
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        temps.endFld(n) = model.fldS(n) + halfdt * driftFac * temps.midPi(n);
                    });

            // Pass 2 (corrector): force at midpoint.
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) { model.fldS(n) = temps.midFld(n); });

            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        model.piS(n) +=
                            halfdt * (ScalarSingletKernels::get(model, n) + (-e) * temps.midPi(n));
                    });
            if (halfNoise > 0) {
                model.piS(0_c) += halfNoise * noise0_;
                if (ns > 1) model.piS(1_c) += halfNoise * noise1_;
            }

            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) { model.fldS(n) = temps.endFld(n); });
        }

        /** Single full-dt RK2 (Numba rk2_nonfused): 2 passes, one FDT kick of dt. */
        void nonfusedRk2Step(Model& model, T tMinust0) {
            const T dt = model.dt;
            const T halfdt = dt / 2.0;
            const T a0 = model.aI;
            const T T_now = model.currentT();
            const T e0 = etaEff(model);
            T fullNoise = 0.0;
            if (model.thermalNoise) {
                const T v = 2.0 * e0 * piVarianceEq(model) * dt;
                fullNoise = std::sqrt(v > 0 ? v : 0.0);
            }

            T a1 = a0;
            T T_mid = T_now;
            if (expansion) {
                const T Hprog = hubble(model) / model.muScale();
                a1 = a0 * std::exp(Hprog * dt);
                T_mid = static_cast<T>(model.temperatureAtScaleFactor(std::sqrt(a0 * a1)));
            }

            auto& temps = *rk2Temps_;
            const int ns = model.activeScalars();
            const T driftFac = std::pow(model.aI, model.alpha - 3);

            model.setCurrentTemperature(T_now);
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        temps.midPi(n) =
                            model.piS(n) +
                            halfdt * (ScalarSingletKernels::get(model, n) + (-e0) * model.piS(n));
                        temps.midFld(n) = model.fldS(n) + halfdt * driftFac * model.piS(n);
                    });
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        temps.endFld(n) = model.fldS(n) + dt * driftFac * temps.midPi(n);
                    });

            model.setCurrentTemperature(T_mid);
            const T e1 = etaEff(model);
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) { model.fldS(n) = temps.midFld(n); });
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) {
                        model.piS(n) +=
                            dt * (ScalarSingletKernels::get(model, n) + (-e1) * temps.midPi(n));
                    });
            if (fullNoise > 0) {
                model.piS(0_c) += fullNoise * noise0_;
                if (ns > 1) model.piS(1_c) += fullNoise * noise1_;
            }
            ForLoop(n, 0, Model::Ns - 1,
                    if (static_cast<int>(n) < ns) { model.fldS(n) = temps.endFld(n); });

            if (expansion) {
                model.aIM = a0;
                if (fixedBackground) {
                    model.aI = aBackground(tMinust0 + dt);
                    model.aSI = aBackground(tMinust0 + dt / 2.0);
                    model.aDotI = aBackground.dot(tMinust0 + dt);
                } else {
                    model.aSI = std::sqrt(a0 * a1);
                    model.aI = a1;
                    model.updateTemperature(model.aI);
                    model.maybeAdvanceExpansionStage();
                    model.aDotI = model.aI * hubble(model) / model.muScale();
                }
            } else {
                model.updateTemperature(model.aI);
            }
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
        bool halfFdtNoise_;  // true for stochastic_scheme=numba (Verlet path)
        NoiseField noise0_;
        NoiseField noise1_;
        std::unique_ptr<Rk2Temps> rk2Temps_;
    };

}  // namespace TempLat

#endif  // COSMOINTERFACE_EVOLVERS_STOCHASTICRK_V2_H
