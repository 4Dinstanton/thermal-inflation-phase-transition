#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H

/* This file is part of the thermal-inflation extension to CosmoLattice.
   It defines TempLat unary operators that evaluate the temperature-dependent
   thermal effective potential (and its derivatives) per lattice site, using the
   tabulated thermal integrals in thermal_tables.hpp.

   CosmoLattice's potential machinery expects the (program-variable) potential
   and its field-derivatives to be returned as symbolic field expressions from
   the model's potentialTerms()/potDeriv()/potDeriv2() functions. Because our
   thermal potential is tabulated and T-dependent (it cannot be written as a
   fixed algebraic expression), we wrap a per-site table lookup in custom unary
   operators, exactly the way CosmoLattice implements exp(), sqrt(), tanh(),
   etc. (see src/include/TempLat/lattice/algebra/operators/exponential.h).

   The operators read the PROGRAM field value phiTilde at site i, convert to the
   physical field phi = fStar * phiTilde, evaluate the physical quantity from the
   tables at the current temperature T, and return it converted to program units:

       Vtilde(phiTilde)   = V_phys(phi) / (fStar^2 * omegaStar^2)
       dVtilde/dphiTilde  = V'_phys(phi) / (fStar * omegaStar^2)
       d2Vtilde/dphiTilde2= V''_phys(phi) / (omegaStar^2)

   The current temperature is read through a pointer so the model/evolver can
   update it every step (T = T0 / a). */

#include "TempLat/util/tdd/tdd.h"
#include "TempLat/lattice/algebra/operators/unaryoperator.h"
#include "TempLat/lattice/algebra/operators/binaryoperator.h"
#include "TempLat/lattice/algebra/helpers/getvalue.h"
#include "TempLat/lattice/algebra/constants/zerotype.h"

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

            inline auto get(ptrdiff_t i) -> double {
                const double phiTilde = GetValue::get(mR, i);
                const double phi = mCtx->fStar * phiTilde;
                const double T = mCtx->T;
                const auto& tab = *mCtx->tables;
                switch (mKind) {
                    case ThermalKind::Value:
                        return tab.V(phi, T) * mCtx->invF2Omega2();
                    case ThermalKind::Deriv:
                        return tab.Vprime(phi, T) * mCtx->invFOmega2();
                    case ThermalKind::Deriv2:
                    default:
                        return tab.Vsecond(phi, T) * mCtx->invOmega2();
                }
            }

            virtual std::string operatorString() const {
                return "thermal";
            }

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

            inline auto get(ptrdiff_t i) -> double {
                const double phi1 = mCtx->fStar * GetValue::get(mR, i);
                const double phi2 = mCtx->fStar * GetValue::get(mT, i);
                double dV1 = 0.0, dV2 = 0.0;
                mCtx->tables->vPrimeComponents(phi1, phi2, mCtx->T,
                                               mCtx->znOrder, mCtx->znStrength, mCtx->znActive(),
                                               dV1, dV2);
                const double dV = (mComp == 0 ? dV1 : dV2);
                return dV * mCtx->invFOmega2();
            }

            virtual std::string operatorString() const { return "thermalComp"; }

        private:
            const ThermalContext* mCtx;
            int mComp;
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
    inline auto thermalVprime1(const R0& r0, const R1& r1, const ThermalContext* ctx) {
        return Operators::ThermalComponentOp<R0, R1>(r0, r1, ctx, 0);
    }
    template <typename R0, typename R1>
    inline auto thermalVprime2(const R0& r0, const R1& r1, const ThermalContext* ctx) {
        return Operators::ThermalComponentOp<R0, R1>(r0, r1, ctx, 1);
    }

}  // namespace TempLat

#endif  // THERMAL_FORCE_H
