#ifndef OPE_BLACK_SCHOLES_HPP
#define OPE_BLACK_SCHOLES_HPP

// ============================================================================
// black_scholes.hpp
//
// Closed-form European option pricing and Greeks under the Black-Scholes model.
// Header-only, inline definitions (this pricer sits on the hot path of the
// parallel Greeks grid, so cross-translation-unit inlinability matters).
//
// Error handling (two-tier, see types.hpp for rationale):
//   - try_price_and_greeks() : noexcept, returns PricingResult{greeks, status}.
//     This is the ONLY entry point the parallel Greeks grid may call. An
//     exception escaping the element callable of a C++17 parallel algorithm
//     calls std::terminate; a throwing pricer inside std::execution::par
//     turns one bad quote into a dead process.
//   - price_and_greeks() and the scalar accessors: throwing convenience
//     wrappers for scalar / interactive use. They convert a non-Ok status
//     into std::domain_error.
//
// Conventions (raw mathematical derivatives; caller scales for quoting):
//   - vega  : per unit volatility            (NOT per 1%; divide by 100 to quote)
//   - theta : dV/dt, per year                (NOT per day; divide by 365 or 252)
//   - rho   : per unit rate                  (NOT per 1%; divide by 100 to quote)
//   - vanna : dVega/dSpot = dDelta/dVol      (raw)
//   - volga : dVega/dVol  (a.k.a. vomma)     (raw)
//
// Assumptions / scope:
//   - No dividends (q = 0). Continuous yield can later be added by replacing
//     r with (r - q) in d1/d2 and discounting the spot leg by e^{-qT}.
//   - Flat, continuously compounded risk-free rate r (any finite real value;
//     negative rates are permitted).
//   - C++17 floor. std::erf/std::exp are NOT constexpr before C++26 (P1383R2),
//     so these functions are runtime (inline, noexcept where they cannot throw),
//     not constexpr.
//
// Numerical precision floor (deep wings):
//   The price is formed as omega * (S*N(omega*d1) - K*e^{-rT}*N(omega*d2)).
//   For deep out-of-the-money options both terms are tiny and nearly cancel:
//   absolute error stays near machine epsilon, but RELATIVE error grows as
//   the price shrinks toward the underflow regime (e.g. a 5-sigma OTM call
//   prices at ~1e-21). This is the same double-precision floor documented for
//   the IV solver in implied_vol.hpp, seen from the pricing side. It is a
//   property of the formulation, not a bug; Jaeckel's "Let's Be Rational"
//   formulation of the Black function is the production fix for both ends.
// ============================================================================

#include <cmath>      // std::erfc, std::exp, std::log, std::sqrt, std::isfinite
#include <stdexcept>  // std::domain_error
#include <string>     // std::string (exception message assembly)

#include "types.hpp"  // OptionType, Greeks, Status, PricingResult

namespace ope {

// All inputs for a single European option.
struct BsInputs {
    double S;            // spot price,                must be > 0 and finite
    double K;            // strike price,              must be > 0 and finite
    double T;            // time to expiry in years,   must be >= 0 and finite
    double r;            // risk-free rate (cont. comp.), any FINITE real
    double sigma;        // annualized volatility,     must be >= 0 and finite
    OptionType type;
};

namespace detail {

// 1/sqrt(2) and 1/sqrt(2*pi) as exact literals. Hardcoded rather than computed
// because std::sqrt is not guaranteed constexpr before C++26, and this avoids
// pulling in M_PI (non-standard) or std::numbers (C++20, above our floor).
inline constexpr double INV_SQRT2    = 0.7071067811865475244;  // 1 / sqrt(2)
inline constexpr double INV_SQRT_2PI = 0.3989422804014326779;  // 1 / sqrt(2*pi)

// Standard normal CDF. The erfc form 0.5*erfc(-x/sqrt2) preserves accuracy in
// the left tail, unlike 0.5*(1 + erf(x/sqrt2)) which suffers cancellation there.
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * INV_SQRT2);
}

// Standard normal PDF.
inline double norm_pdf(double x) noexcept {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// +1 for a call, -1 for a put. Lets price/Greeks share one code path via the
// identity price = omega * (S*N(omega*d1) - K*e^{-rT}*N(omega*d2)).
inline double omega_of(OptionType t) noexcept {
    return (t == OptionType::Call) ? 1.0 : -1.0;
}

// Domain check, noexcept. Returns Status::Ok or Status::BadInput.
//
// Two layers of defense:
//   1. std::isfinite on every double rejects NaN AND +/-inf. The previous
//      revision's `!(x > 0)` trick rejected NaN for S/K/T/sigma but let
//      NaN r through unchecked (silently pricing to NaN) and accepted
//      infinities everywhere (inf > 0 is true), producing garbage output.
//   2. The domain comparisons proper (S > 0, K > 0, T >= 0, sigma >= 0).
inline Status validate(const BsInputs& in) noexcept {
    if (!std::isfinite(in.S) || !std::isfinite(in.K) || !std::isfinite(in.T) ||
        !std::isfinite(in.r) || !std::isfinite(in.sigma)) {
        return Status::BadInput;
    }
    if (!(in.S > 0.0))      return Status::BadInput;
    if (!(in.K > 0.0))      return Status::BadInput;
    if (!(in.T >= 0.0))     return Status::BadInput;
    if (!(in.sigma >= 0.0)) return Status::BadInput;
    return Status::Ok;
}

}  // namespace detail

// ----------------------------------------------------------------------------
// try_price_and_greeks: noexcept, status-returning. The grid entry point.
//
// Full price + Greeks in a single pass. d1, d2, and phi(d1) are computed once
// and reused across every Greek, which is materially cheaper than calling the
// individual accessors (each of which recomputes those shared terms).
//
// On BadInput the returned Greeks are all quiet NaN (never zero), so a caller
// that ignores the status cannot mistake a failure for a valid flat price.
// ----------------------------------------------------------------------------
inline PricingResult try_price_and_greeks(const BsInputs& in) noexcept {
    const Status st = detail::validate(in);
    if (st != Status::Ok) {
        return PricingResult{nan_greeks(), st};
    }

    const double S     = in.S;
    const double K     = in.K;
    const double T     = in.T;
    const double r     = in.r;
    const double sigma = in.sigma;
    const double omega = detail::omega_of(in.type);

    const double disc  = std::exp(-r * T);    // discount factor e^{-rT}
    const double sqrtT = std::sqrt(T);
    const double vol   = sigma * sqrtT;       // total volatility over the life

    Greeks g{};

    // ---- Degenerate (no diffusion): T == 0 or sigma == 0 --------------------
    // The terminal payoff is deterministic. Its present value is the discounted
    // forward intrinsic. IMPORTANT: at sigma == 0 with T > 0 this is
    // max(omega*(S - K*e^{-rT}), 0), NOT the spot intrinsic max(omega*(S-K), 0).
    // The two coincide only at T == 0. Greeks are set to their limiting values
    // so they remain consistent with the main branch as vol -> 0. At the exact
    // ATM-forward kink (fwd == 0) Greeks are not well defined; we return 0 there.
    if (vol <= 0.0) {
        const double fwd = omega * (S - K * disc);   // forward intrinsic (signed)
        const bool   itm = fwd > 0.0;
        g.price = itm ? fwd : 0.0;
        g.delta = itm ? omega : 0.0;
        g.gamma = 0.0;
        g.vega  = 0.0;
        g.theta = itm ? (-omega * r * K * disc) : 0.0;  // lim of full theta
        g.rho   = itm ? ( omega * K * T * disc) : 0.0;  // lim of full rho
        g.vanna = 0.0;
        g.volga = 0.0;
        return PricingResult{g, Status::Ok};
    }

    // ---- Standard branch ----------------------------------------------------
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol;
    const double d2 = d1 - vol;

    const double Nd1o = detail::norm_cdf(omega * d1);  // N(omega * d1)
    const double Nd2o = detail::norm_cdf(omega * d2);  // N(omega * d2)
    const double pdf  = detail::norm_pdf(d1);          // phi(d1)

    g.price = omega * (S * Nd1o - K * disc * Nd2o);
    g.delta = omega * Nd1o;
    g.gamma = pdf / (S * vol);                          // phi(d1) / (S*sigma*sqrtT)
    g.vega  = S * pdf * sqrtT;                           // per unit vol
    g.theta = -(S * pdf * sigma) / (2.0 * sqrtT)        // time-decay term
              - omega * r * K * disc * Nd2o;            // carry term; per year
    g.rho   = omega * K * T * disc * Nd2o;             // per unit rate
    g.vanna = -pdf * d2 / sigma;                        // dVega/dSpot
    g.volga = g.vega * d1 * d2 / sigma;                 // dVega/dVol (vomma)

    return PricingResult{g, Status::Ok};
}

// ----------------------------------------------------------------------------
// Throwing convenience wrapper for scalar / interactive use.
//
// NEVER call this (or the scalar accessors below) as the element function of a
// parallel algorithm: an exception escaping that callable is std::terminate,
// not a catchable error. Use try_price_and_greeks() there.
// ----------------------------------------------------------------------------
inline Greeks price_and_greeks(const BsInputs& in) {
    const PricingResult res = try_price_and_greeks(in);
    if (!res.ok()) {
        throw std::domain_error(std::string("black_scholes: ") +
                                to_string(res.status) +
                                " (S>0, K>0, T>=0, sigma>=0, all finite)");
    }
    return res.greeks;
}

// ----------------------------------------------------------------------------
// Convenience scalar accessors (throwing).
//
// Each recomputes the shared terms internally, so calling several of these for
// the same option is wasteful. Use price_and_greeks() (scalar) or
// try_price_and_greeks() (grids) when you need more than one quantity.
// ----------------------------------------------------------------------------
inline double price(const BsInputs& in) { return price_and_greeks(in).price; }
inline double delta(const BsInputs& in) { return price_and_greeks(in).delta; }
inline double gamma(const BsInputs& in) { return price_and_greeks(in).gamma; }
inline double vega (const BsInputs& in) { return price_and_greeks(in).vega;  }
inline double theta(const BsInputs& in) { return price_and_greeks(in).theta; }
inline double rho  (const BsInputs& in) { return price_and_greeks(in).rho;   }

}  // namespace ope

#endif  // OPE_BLACK_SCHOLES_HPP