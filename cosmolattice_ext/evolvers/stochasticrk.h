#ifndef COSMOINTERFACE_EVOLVERS_STOCHASTICRK_H
#define COSMOINTERFACE_EVOLVERS_STOCHASTICRK_H

/* Stochastic RK evolver for the thermal-inflation model.
 *
 * Scheme::NumbaRK2 (default): explicit comoving Langevin + fused 4-pass RK2 matching
 * simulation/latticeSimeRescale_numba.py (rk2_fused_inline):
 *
 *     k_pi = (1/a^2) Lap phi - eta_eff pi - V'(phi,T)/mu^2
 *     phi += (dt/2) pi
 *     pi  += (dt/2) k_pi + 0.5 noise_scale z
 *
 * with eta_eff = eta + 3H/mu, noise_scale^2 = 2 eta_eff T dt / (a^3 mu^2 dx_phys^3),
 * and scale-factor update once per step (before the RK2 passes), as in numba.
 *
 * Noise: two half-kicks per step (passes 2 and 4), each 0.5*noise_scale*z with
 * independent z (rk2_fused_inline). noise_scale is evaluated once from T at the
 * start of the step (not T_mid), matching numba fused_inline.
 *
 * RNG: per-site hash Box-Muller (rk2_fused_inline), independent at each site/step/pass:
 *     pass 2: seed = site * 73856093  XOR step
 *     pass 4: seed = site * 19349669 XOR step
 * where site is the flat GLOBAL lattice index i*Ny*Nz+j*Nz+k (0-based),
 * not the local MPI memory offset it() — using it() duplicates noise across ranks.
 *
 * stochastic_scheme:
 *   numba (default) — bare numba noise_scale (nucleation parity with numba reference)
 *   fdt             — sqrt(2) noise correction for fixed-T equipartition tests
 *   fused           — legacy FusedRK2 symplectic kernel
 *
 * Program-field mapping (alpha = 1, phi_GeV = fStar * fldS):
 *     pi_numba = fStar * piS          (piS = d fldS / dt at a = 1)
 *     phi_GeV += (dt/2) pi_numba  <=>  fldS += (dt/2) piS
 *
 * Scheme::FusedRK2: legacy CosmoLattice symplectic kernel + two half-steps
 * (ScalarSingletKernels + a^{alpha-3} drift). Kept for comparison / fixed-T tests.
 */

#include <cmath>
#include <cstdint>
#include <string>
#include "TempLat/util/rangeiteration/for_in_range.h"
#include "TempLat/lattice/algebra/spatialderivatives/latticelaplacian.h"
#include "TempLat/lattice/algebra/helpers/getvalue.h"
#include "CosmoInterface/evolvers/kernels/kernels.h"
#include "CosmoInterface/evolvers/kernels/gwskernels.h"
#include "CosmoInterface/runparameters.h"
#include "CosmoInterface/definitions/averages.h"
#include "CosmoInterface/definitions/fixedbackgroundexpansion.h"

namespace TempLat {

    template <typename T = double>
    class StochasticRK {
    public:
        enum class Scheme { NumbaRK2, FusedRK2, EulerMaruyama };

        template <class Model>
        StochasticRK(Model& model, RunParameters<T>& rPar)
            : expansion(rPar.expansion),
              fixedBackground(rPar.fixedBackground),
              aBackground(model, rPar),
              scheme_(Scheme::NumbaRK2),
              useFdtNoise_(false),
              latticeStep_(0) {
            model.updateTemperature(model.aI);
            const std::string& sch = model.stochasticScheme;
            if (sch == "fused" || sch == "FusedRK2" || sch == "cl") {
                scheme_ = Scheme::FusedRK2;
            } else if (sch == "fdt" || sch == "FDT") {
                useFdtNoise_ = true;
            }
        }

        void setScheme(Scheme s) { scheme_ = s; }

        template <class Model>
        void evolve(Model& model, T tMinust0) {
            model.updateTemperature(model.aI);

            switch (scheme_) {
                case Scheme::NumbaRK2:
                default:
                    numbaRK2(model, tMinust0);
                    break;
                case Scheme::FusedRK2:
                    fusedRK2(model, tMinust0);
                    break;
                case Scheme::EulerMaruyama:
                    eulerMaruyama(model, tMinust0);
                    break;
            }
        }

        template <class Model>
        void sync(Model& model, T tMinust0) {
            if (fixedBackground) model.aDotI = aBackground.dot(tMinust0);
        }

    private:
        // ---- Shared coefficients ------------------------------------------------
        template <class Model>
        T etaThermal(Model& model) const {
            return model.etaPhys / model.muScale();
        }

        template <class Model>
        T hubble(Model& model) const {
            // Prefer model-prescribed Hubble when staged expansion is available.
            if (model.expansionStaged()) {
                return static_cast<T>(model.prescribedHubble());
            }
            const T M_PL = 2.4e18;
            const T T_now = model.currentT();
            const T chig2 = 30.0 / (M_PI * M_PI * model.gStarHubble);
            const T H2 = (std::pow(T_now, 4) / chig2 + model.delVForHubble())
                       / (3.0 * M_PL * M_PL);
            return std::sqrt(H2 > 0 ? H2 : 0.0);
        }

        template <class Model>
        T hubbleRate(Model& model) const {
            if (fixedBackground && model.aI > 0) {
                return model.aDotI * model.muScale() / model.aI;
            }
            return hubble(model);
        }

        template <class Model>
        T etaEffNoise(Model& model) const {
            T eeff = etaThermal(model);
            if (expansion) {
                eeff += 3.0 * hubbleRate(model) / model.muScale();
            }
            return eeff;
        }

        // Numba noise amplitude on pi_numba (no sqrt(2); two half-kicks).
        template <class Model>
        T numbaNoiseScale(Model& model, T T_now, T inv_a3, T eta_eff) const {
            if (!model.thermalNoise) return 0.0;
            const T mu = model.muScale();
            const T dt = model.dt;
            const T dx3 = std::pow(model.dxPhys, 3);
            const T val = 2.0 * eta_eff * T_now * dt * inv_a3 / (mu * mu * dx3);
            return std::sqrt(val > 0 ? val : 0.0);
        }

        // Legacy FusedRK2 noise on piS (includes sqrt(2) for two half-steps).
        template <class Model>
        T invA3(Model& model) const {
            if (!expansion) return 1.0;
            const T a = model.aI;
            if (a <= 0) return 1.0;
            return 1.0 / (a * a * a);
        }

        template <class Model>
        T fusedNoiseScale(Model& model) const {
            if (!model.thermalNoise) return 0.0;
            const T mu = model.muScale();
            const T eeff = etaEffNoise(model);
            const T T_now = model.currentT();
            const T dt = model.dt;
            const T dx3 = std::pow(model.dxPhys, 3);
            const T val = 2.0 * eeff * T_now * dt * invA3(model) / (mu * mu * dx3);
            return std::sqrt(2.0) * std::sqrt(val > 0 ? val : 0.0) / model.fStarVal();
        }

        // Program-unit Laplacian at one site (matches LatLapl 2nd-order stencil).
        // getJumpsInMemoryOrder() returns one +1 stride per dimension; both ±
        // neighbours are required (previously only +j was summed → 3-point
        // broken stencil, spurious -3 phi/dx^2 mass, checkerboard domains).
        template <class Model, class FieldT>
        T lapProgAt(Model& model, const FieldT& fld, ptrdiff_t i) const {
            const T inv_dx2 = 1.0 / (model.dx * model.dx);
            const T c = fld.get(i);
            T sum = -static_cast<T>(2 * Model::NDim) * c;
            const auto& jumps =
                model.getToolBox()->mLayouts.getConfigSpaceJumps().getJumpsInMemoryOrder();
            for (const ptrdiff_t j : jumps) {
                sum += fld.get(i + j);
                sum += fld.get(i - j);
            }
            return sum * inv_dx2;
        }

        // Flat global site index in [0, N^3), matching numba i*Ny*Nz+j*Nz+k
        // with (i,j,k) in [0,N). Uses it.getVec() signed coords — NOT it()
        // (local memory offset), which repeats across MPI ranks and duplicates noise.
        template <class Model, class LooperT>
        static uint64_t globalSiteIndex(Model& model, LooperT& it) {
            const auto c = it.getVec();
            const ptrdiff_t N = model.getToolBox()->mNGridPointsVec[0];
            auto to0 = [N](ptrdiff_t x) -> uint64_t {
                return static_cast<uint64_t>(x >= 0 ? x : x + N);
            };
            const uint64_t ix = to0(c[0]);
            const uint64_t iy = to0(c.size() > 1 ? c[1] : 0);
            const uint64_t iz = to0(c.size() > 2 ? c[2] : 0);
            const uint64_t Nu = static_cast<uint64_t>(N);
            return (ix * Nu + iy) * Nu + iz;
        }

        // Numba rk2_fused_inline hash RNG (see latticeSimeRescale_numba._hash_rng_pair).
        static uint64_t hashMix64(uint64_t x) {
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27;
            x *= 0x94D049BB133111EBULL;
            x ^= x >> 31;
            return x;
        }

        static void hashRngPair(uint64_t seed, T& u1, T& u2) {
            uint64_t x = seed;
            x = hashMix64(x);
            u1 = static_cast<T>((x >> 11) * (1.0 / 9007199254740992.0));
            x = x * 0x2545F4914F6CDD1DULL ^ 0x9E3779B97F4A7C15ULL;
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 31;
            u2 = static_cast<T>((x >> 11) * (1.0 / 9007199254740992.0));
            if (u1 < 1e-12) u1 = static_cast<T>(1e-12);
        }

        static T hashGaussian(uint64_t seed) {
            T u1, u2;
            hashRngPair(seed, u1, u2);
            return std::sqrt(-2.0 * std::log(u1))
                 * std::cos(6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341359642961730265646132941876892 * u2);
        }

        static constexpr uint64_t kNoiseMulPass2 = 73856093ULL;
        static constexpr uint64_t kNoiseMulPass4 = 19349669ULL;

        template <class Model>
        T numbaHalfNoise(Model& model, T T_now, T inv_a3, T eta_eff) const {
            if (!model.thermalNoise) return 0.0;
            T half = 0.5 * numbaNoiseScale(model, T_now, inv_a3, eta_eff) / model.fStarVal();
            if (useFdtNoise_) half *= std::sqrt(2.0);
            return half;
        }

        // ---- Numba fused 4-pass RK2 ---------------------------------------------
        template <class Model>
        void numbaRK2(Model& model, T /*tMinust0*/) {
            const T mu = model.muScale();
            const T fStar = model.fStarVal();
            const T halfdt = model.dt / 2.0;
            const T piScale = fStar;
            const T invPiScale = 1.0 / piScale;

            const T a0 = model.aI;
            model.updateTemperature(a0);
            const T T_now = model.currentT();

            T inv_a2 = 1.0;
            T inv_a3 = 1.0;
            T eta_eff = etaThermal(model);
            T T_mid = T_now;
            T a1 = a0;

            if (expansion) {
                inv_a2 = 1.0 / (a0 * a0);
                inv_a3 = 1.0 / (a0 * a0 * a0);
                eta_eff = etaEffNoise(model);
                const T H = hubbleRate(model);
                a1 = a0 * (1.0 + H * model.dt / mu);
                T_mid = static_cast<T>(model.temperatureAtScaleFactor(a1));
            }

            const T invMu2 = 1.0 / (mu * mu);
            const T halfNoise = numbaHalfNoise(model, T_now, inv_a3, eta_eff);
            const uint64_t stepSeed = latticeStep_;
            ++latticeStep_;
            const int nScalars = model.activeScalars();

            auto& it = model.getToolBox()->itX();

            auto phiGeVAt = [&](const auto& f0, const auto& f1, ptrdiff_t i) {
                const T p1 = fStar * f0.get(i);
                const T p2 = (nScalars > 1) ? fStar * f1.get(i) : static_cast<T>(0);
                return std::pair<T, T>(p1, p2);
            };

            auto rkPass1 = [&](double T_step) {
                model.setCurrentTemperature(T_step);
                model.tmpPiS(0_c) = LatLapl<Model::NDim>(model.fldS(0_c));
                if (nScalars > 1) {
                    model.tmpPiS(1_c) = LatLapl<Model::NDim>(model.fldS(1_c));
                }
                for (it.begin(); it.end(); ++it) {
                    const ptrdiff_t i = it();
                    const auto phiGeV = phiGeVAt(model.fldS(0_c), model.fldS(1_c), i);
                    const T lap0 = inv_a2 * fStar * model.tmpPiS(0_c).get(i);
                    const T pi0 = piScale * model.piS(0_c).get(i);
                    T dV0 = 0.0;
                    model.vPrimeComponentGeV(phiGeV.first, phiGeV.second, 0, dV0);
                    const T kpi0 = lap0 - eta_eff * pi0 - dV0 * invMu2;
                    model.tmpFldS(0_c).getSet(i) = model.fldS(0_c).get(i) + halfdt * model.piS(0_c).get(i);
                    model.tmpPiS(0_c).getSet(i) = model.piS(0_c).get(i) + halfdt * kpi0 * invPiScale;
                    if (nScalars > 1) {
                        const T lap1 = inv_a2 * fStar * model.tmpPiS(1_c).get(i);
                        const T pi1 = piScale * model.piS(1_c).get(i);
                        T dV1 = 0.0;
                        model.vPrimeComponentGeV(phiGeV.first, phiGeV.second, 1, dV1);
                        const T kpi1 = lap1 - eta_eff * pi1 - dV1 * invMu2;
                        model.tmpFldS(1_c).getSet(i) = model.fldS(1_c).get(i) + halfdt * model.piS(1_c).get(i);
                        model.tmpPiS(1_c).getSet(i) = model.piS(1_c).get(i) + halfdt * kpi1 * invPiScale;
                    }
                }
                model.tmpFldS(0_c).setGhostsAreStale();
                if (nScalars > 1) model.tmpFldS(1_c).setGhostsAreStale();
            };

            auto rkPass2 = [&](double T_step, uint64_t noiseMul) {
                model.tmpFldS(0_c).confirmGhostsUpToDate();
                if (nScalars > 1) model.tmpFldS(1_c).confirmGhostsUpToDate();
                model.setCurrentTemperature(T_step);
                for (it.begin(); it.end(); ++it) {
                    const ptrdiff_t i = it();
                    const auto phiGeV = phiGeVAt(model.tmpFldS(0_c), model.tmpFldS(1_c), i);
                    const T lap0 = inv_a2 * fStar * lapProgAt(model, model.tmpFldS(0_c), i);
                    const T pi0 = piScale * model.tmpPiS(0_c).get(i);
                    T dV0 = 0.0;
                    model.vPrimeComponentGeV(phiGeV.first, phiGeV.second, 0, dV0);
                    const T kpi0 = lap0 - eta_eff * pi0 - dV0 * invMu2;
                    model.fldS(0_c).getSet(i) += halfdt * model.tmpPiS(0_c).get(i);
                    T piVal0 = model.piS(0_c).get(i) + halfdt * kpi0 * invPiScale;
                    // Box-Muller pair: independent z0,z1 (matches latticeSimComplex_numba).
                    // Previously both components reused the same seed → identical kicks
                    // along (1,1), inflating radial noise by ~sqrt(2) vs real scalar.
                    T z0 = 0, z1 = 0;
                    if (halfNoise > 0) {
                        const uint64_t site = globalSiteIndex(model, it);
                        const uint64_t seed = site * noiseMul ^ stepSeed;
                        T u1, u2;
                        hashRngPair(seed, u1, u2);
                        const T r = std::sqrt(-2.0 * std::log(u1));
                        const T th = static_cast<T>(6.28318530717958647692) * u2;
                        z0 = r * std::cos(th);
                        z1 = r * std::sin(th);
                        piVal0 += halfNoise * z0;
                    }
                    model.piS(0_c).getSet(i) = piVal0;
                    if (nScalars > 1) {
                        const T lap1 = inv_a2 * fStar * lapProgAt(model, model.tmpFldS(1_c), i);
                        const T pi1 = piScale * model.tmpPiS(1_c).get(i);
                        T dV1 = 0.0;
                        model.vPrimeComponentGeV(phiGeV.first, phiGeV.second, 1, dV1);
                        const T kpi1 = lap1 - eta_eff * pi1 - dV1 * invMu2;
                        model.fldS(1_c).getSet(i) += halfdt * model.tmpPiS(1_c).get(i);
                        T piVal1 = model.piS(1_c).get(i) + halfdt * kpi1 * invPiScale;
                        if (halfNoise > 0) {
                            piVal1 += halfNoise * z1;
                        }
                        model.piS(1_c).getSet(i) = piVal1;
                    }
                }
                model.fldS(0_c).setGhostsAreStale();
                if (nScalars > 1) model.fldS(1_c).setGhostsAreStale();
            };

            model.fldS(0_c).confirmGhostsUpToDate();
            if (nScalars > 1) model.fldS(1_c).confirmGhostsUpToDate();

            rkPass1(T_now);
            rkPass2(T_now, kNoiseMulPass2);
            model.fldS(0_c).confirmGhostsUpToDate();
            if (nScalars > 1) model.fldS(1_c).confirmGhostsUpToDate();
            rkPass1(T_mid);
            rkPass2(T_mid, kNoiseMulPass4);

            if (expansion) {
                model.aIM = a0;
                model.aI = a1;
                model.aSI = std::sqrt(a0 * a1);
                model.updateTemperature(model.aI);
                model.maybeAdvanceExpansionStage();
                // Recompute H after possible stage change so aDotI stays consistent.
                model.aDotI = model.aI * hubble(model) / mu;
            }

            // GW source (PITensor) reads forwDiff(fldS); refresh ghosts first.
            // GWs do not feed back into the scalar EOM / Hubble in StochasticRK.
            model.fldS(0_c).confirmGhostsUpToDate();
            if (nScalars > 1) model.fldS(1_c).confirmGhostsUpToDate();
            evolveGWs(model);
        }

        // ---- Legacy FusedRK2 (CosmoLattice symplectic form) ---------------------
        template <class Model>
        void fusedRK2(Model& model, T tMinust0) {
            const T etaKick = etaThermal(model);
            const T halfNoise = 0.5 * fusedNoiseScale(model);

            halfStep(model, etaKick, halfNoise, tMinust0);

            model.updateTemperature(model.aI);
            halfStep(model, etaKick, 0.5 * fusedNoiseScale(model), tMinust0 + model.dt / 2.0);
            evolveGWs(model);
        }

        template <class Model>
        void halfStep(Model& model, T eeff, T halfNoise, T tMinust0) {
            const T halfdt = model.dt / 2.0;

            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < model.activeScalars()) {
                    model.piS(n) += halfdt * ScalarSingletKernels::get(model, n);
                    model.piS(n) += (-halfdt * eeff) * model.piS(n);
                    if (halfNoise > 0) addGaussian(model, model.piS(n), halfNoise);
                }
            );

            if (expansion) advanceScaleFactor(model, halfdt, tMinust0 + model.dt / 2.0);
            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < model.activeScalars()) {
                    model.fldS(n) += pow(model.aSI, model.alpha - 3) * (halfdt * model.piS(n));
                }
            );
        }

        template <class Model>
        void eulerMaruyama(Model& model, T tMinust0) {
            const T eeff = etaThermal(model);
            const T ns = fusedNoiseScale(model);
            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < model.activeScalars()) {
                    model.piS(n) += model.dt * ScalarSingletKernels::get(model, n);
                    model.piS(n) += (-model.dt * eeff) * model.piS(n);
                    if (ns > 0) addGaussian(model, model.piS(n), ns);
                }
            );
            if (expansion) advanceScaleFactor(model, model.dt, tMinust0 + model.dt);
            ForLoop(n, 0, Model::Ns - 1,
                if (static_cast<int>(n) < model.activeScalars()) {
                    model.fldS(n) += pow(model.aSI, model.alpha - 3) * (model.dt * model.piS(n));
                }
            );
            evolveGWs(model);
        }

        // Velocity-Verlet-style GW sub-step (no thermal noise on GW sector).
        template <class Model>
        void evolveGWs(Model& model) {
            if (model.fldGWs == nullptr) return;
            kickGWs(model, 1.0);
            driftGWs(model, 1.0);
            kickGWs(model, 1.0);
        }

        template <class Model>
        void kickGWs(Model& model, T w) {
            ForLoop(i, 0, Model::NGWs - 1,
                (*model.piGWs)(i) += (w * model.dt / 2) * GWsKernels::get(model, i);
            );
        }

        template <class Model>
        void driftGWs(Model& model, T w) {
            (*model.fldGWs) += pow(model.aSI, model.alpha - 3) * (model.dt * w * (*model.piGWs));
        }

        template <class Model>
        void advanceScaleFactor(Model& model, T dEta, T tMinust0) {
            model.aIM = model.aI;
            if (fixedBackground) {
                model.aI = aBackground(tMinust0);
                model.aSI = aBackground(tMinust0 + dEta / 2.0);
                model.aDotI = aBackground.dot(tMinust0);
                return;
            }
            const T Hprog = hubble(model) / model.muScale();
            model.aSI = model.aI * std::exp(Hprog * dEta / 2.0);
            model.aI = model.aI * std::exp(Hprog * dEta);
            model.updateTemperature(model.aI);
            model.maybeAdvanceExpansionStage();
            model.aDotI = model.aI * hubble(model) / model.muScale();
        }

        template <class Model, class FieldT>
        void addGaussian(Model& model, FieldT&& field, T sigma) {
            // Per-site hash from global coords (MPI-safe). Do not use a shared
            // sequential RNG — that duplicates noise across ranks.
            auto& it = model.getToolBox()->itX();
            const uint64_t stepMix = latticeStep_ * 0x9E3779B97F4A7C15ULL;
            for (it.begin(); it.end(); ++it) {
                const uint64_t seed = globalSiteIndex(model, it) ^ stepMix;
                field.getSet(it()) += sigma * hashGaussian(seed);
            }
        }

        bool expansion;
        bool fixedBackground;
        FixedBackgroundExpansion<T> aBackground;
        Scheme scheme_;
        bool useFdtNoise_;
        uint64_t latticeStep_;
    };

}  // namespace TempLat

#endif  // COSMOINTERFACE_EVOLVERS_STOCHASTICRK_H
