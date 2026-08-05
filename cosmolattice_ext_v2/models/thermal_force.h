#ifndef THERMAL_FORCE_V2_H
#define THERMAL_FORCE_V2_H

/* CosmoLattice v2.0 port of cosmolattice_ext/models/thermal_force.h.

   Same physics: TempLat unary/binary operators that evaluate the tabulated
   finite-temperature effective potential per lattice site, so a T-dependent,
   non-algebraic V(phi,T) can appear in the symbolic EOM.

   What changed for v2 / TempLat v1.0.0:

     v1 (CL 1.x)                          v2 (TempLat)
     -----------------------------------  ------------------------------------
     auto get(ptrdiff_t i)                auto eval(const IDX&... idx)
     GetValue::get(mR, i)                 DoEval::eval(mR, idx...)
     (no device annotation)               DEVICE_FORCEINLINE_FUNCTION
     virtual std::string operatorString() same, but must be `override`

   Device restriction: the table lookup dereferences host memory
   (ThermalTables holds std::vector). That is fine for the Serial / Threads /
   OpenMP Kokkos backends, i.e. any CPU build. A CUDA/HIP build will fail to
   compile here, which is the intended loud failure: the tables would have to
   be mirrored into a Kokkos::View first. See README for the plan. */

#include "TempLat/lattice/algebra/constants/zerotype.h"
#include "TempLat/lattice/algebra/helpers/doeval.h"
#include "TempLat/lattice/algebra/helpers/isvariadicindex.h"
#include "TempLat/lattice/algebra/operators/binaryoperator.h"
#include "TempLat/lattice/algebra/operators/unaryoperator.h"
#include "TempLat/parallel/device.h"

#include "thermal_tables.hpp"

namespace TempLat {

    /** \brief Shared runtime context for the thermal operators: a pointer to the
     *  loaded tables, the current temperature, and the program-variable scales. */
    struct ThermalContext {
        const ThermalInflation::ThermalTables* tables = nullptr;
        double T = 1.0;
        double fStar = 1.0;
        double omegaStar = 1.0;
        double znOrder = 0.0;
        double znStrength = 0.0;
        double znTurnOnT = 0.0;

        double invFOmega2() const { return 1.0 / (fStar * omegaStar * omegaStar); }
        double invF2Omega2() const { return 1.0 / (fStar * fStar * omegaStar * omegaStar); }
        double invOmega2() const { return 1.0 / (omegaStar * omegaStar); }

        bool znActive() const {
            if (znOrder <= 0.0 || znStrength <= 0.0) return false;
            if (znTurnOnT <= 0.0) return true;
            return T <= znTurnOnT;
        }
    };

    namespace Operators {

        // 0: potential value, 1: first derivative, 2: second derivative.
        enum class ThermalKind { Value = 0, Deriv = 1, Deriv2 = 2 };

        template <typename R>
        class ThermalOp : public UnaryOperator<R> {
        public:
            using UnaryOperator<R>::mR;

            ThermalOp(const R& r, const ThermalContext* ctx, ThermalKind kind)
                : UnaryOperator<R>(r), mCtx(ctx), mKind(kind) {}

            template <typename... IDX>
                requires requires(std::decay_t<R> t, IDX... idx) {
                    requires IsVariadicIndex<IDX...>;
                    DoEval::eval(t, idx...);
                }
            DEVICE_FORCEINLINE_FUNCTION auto eval(const IDX&... idx) const -> double {
                const double phiTilde = static_cast<double>(DoEval::eval(mR, idx...));
                const double phi = mCtx->fStar * phiTilde;
                const auto& tab = *mCtx->tables;
                switch (mKind) {
                    case ThermalKind::Value:
                        return tab.V(phi, mCtx->T) * mCtx->invF2Omega2();
                    case ThermalKind::Deriv:
                        return tab.Vprime(phi, mCtx->T) * mCtx->invFOmega2();
                    case ThermalKind::Deriv2:
                    default:
                        return tab.Vsecond(phi, mCtx->T) * mCtx->invOmega2();
                }
            }

            std::string operatorString() const override { return "thermal"; }

        private:
            const ThermalContext* mCtx;
            ThermalKind mKind;
        };

        template <typename R0, typename R1>
        class ThermalComponentOp : public BinaryOperator<R0, R1> {
        public:
            using BinaryOperator<R0, R1>::mR;
            using BinaryOperator<R0, R1>::mT;

            ThermalComponentOp(const R0& r0, const R1& r1, const ThermalContext* ctx, int comp)
                : BinaryOperator<R0, R1>(r0, r1), mCtx(ctx), mComp(comp) {}

            template <typename... IDX>
                requires requires(std::decay_t<R0> a, std::decay_t<R1> b, IDX... idx) {
                    requires IsVariadicIndex<IDX...>;
                    DoEval::eval(a, idx...);
                    DoEval::eval(b, idx...);
                }
            DEVICE_FORCEINLINE_FUNCTION auto eval(const IDX&... idx) const -> double {
                const double phi1 = mCtx->fStar * static_cast<double>(DoEval::eval(mR, idx...));
                const double phi2 = mCtx->fStar * static_cast<double>(DoEval::eval(mT, idx...));
                double dV1 = 0.0, dV2 = 0.0;
                mCtx->tables->vPrimeComponents(phi1, phi2, mCtx->T,
                                               mCtx->znOrder, mCtx->znStrength, mCtx->znActive(),
                                               dV1, dV2);
                return (mComp == 0 ? dV1 : dV2) * mCtx->invFOmega2();
            }

            std::string operatorString() const override { return "thermalComp"; }

        private:
            const ThermalContext* mCtx;
            int mComp;
        };

        // V(sqrt(phi1^2+phi2^2), T) for Energies::rho with complex / two-component fields.
        template <typename R0, typename R1>
        class ThermalRhoValueOp : public BinaryOperator<R0, R1> {
        public:
            using BinaryOperator<R0, R1>::mR;
            using BinaryOperator<R0, R1>::mT;

            ThermalRhoValueOp(const R0& r0, const R1& r1, const ThermalContext* ctx)
                : BinaryOperator<R0, R1>(r0, r1), mCtx(ctx) {}

            template <typename... IDX>
                requires requires(std::decay_t<R0> a, std::decay_t<R1> b, IDX... idx) {
                    requires IsVariadicIndex<IDX...>;
                    DoEval::eval(a, idx...);
                    DoEval::eval(b, idx...);
                }
            DEVICE_FORCEINLINE_FUNCTION auto eval(const IDX&... idx) const -> double {
                const double phi1 = mCtx->fStar * static_cast<double>(DoEval::eval(mR, idx...));
                const double phi2 = mCtx->fStar * static_cast<double>(DoEval::eval(mT, idx...));
                const double rho = device::sqrt(phi1 * phi1 + phi2 * phi2);
                return mCtx->tables->V(rho, mCtx->T) * mCtx->invF2Omega2();
            }

            std::string operatorString() const override { return "thermalRhoV"; }

        private:
            const ThermalContext* mCtx;
        };

    }  // namespace Operators

    /** \brief Convenience factory functions mirroring exp()/sqrt() style. */
    template <typename R>
    inline auto thermalV(const R& r, const ThermalContext* ctx) {
        return Operators::ThermalOp<R>(r, ctx, Operators::ThermalKind::Value);
    }
    template <typename R>
    inline auto thermalVprime(const R& r, const ThermalContext* ctx) {
        return Operators::ThermalOp<R>(r, ctx, Operators::ThermalKind::Deriv);
    }
    template <typename R>
    inline auto thermalVsecond(const R& r, const ThermalContext* ctx) {
        return Operators::ThermalOp<R>(r, ctx, Operators::ThermalKind::Deriv2);
    }
    template <typename R0, typename R1>
    inline auto thermalVrho(const R0& r0, const R1& r1, const ThermalContext* ctx) {
        return Operators::ThermalRhoValueOp<R0, R1>(r0, r1, ctx);
    }
    template <typename R0, typename R1>
    inline auto thermalVprime1(const R0& r0, const R1& r1, const ThermalContext* ctx) {
        return Operators::ThermalComponentOp<R0, R1>(r0, r1, ctx, 0);
    }
    template <typename R0, typename R1>
    inline auto thermalVprime2(const R0& r0, const R1& r1, const ThermalContext* ctx) {
        return Operators::ThermalComponentOp<R0, R1>(r0, r1, ctx, 1);
    }

}  // namespace TempLat

#endif  // THERMAL_FORCE_V2_H
