#ifndef OPE_BLACK_SCHOLES_HPP
#define OPE_BLACK_SCHOLES_HPP

// ============================================================================
// black_scholes.hpp
//
// Closed-form European option pricing and Greeks under the Black-Scholes model.
// Header-only, inline definitions (this pricer sits on the hot path of the
// parallel Greeks grid, so cross-translation-unit inlinability matters).
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
//   - Flat, continuously compounded risk-free rate r (any real value; negative
//     rates are permitted).
//   - C++17 floor. std::erf/std::exp are NOT constexpr before C++26 (P1383R2),
//     so these functions are runtime (inline, noexcept where they cannot throw),
//     not constexpr.
//
// NOTE: OptionType and Greeks are defined here for now. When the binomial tree
// and Monte Carlo modules are added they should share these types; consider
// promoting OptionType (and possibly Greeks) to a common "types.hpp" then.
// ============================================================================

#include <cmath>      // std::erfc, std::exp, std::log, std::sqrt
#include <stdexcept>  // std::domain_error

namespace ope {

enum class OptionType { Call, Put };

// All inputs for a single European option.
struct BsInputs {
    double S;            // spot price,                must be > 0
    double K;            // strike price,              must be > 0
    double T;            // time to expiry in years,   must be >= 0
    double r;            // risk-free rate (cont. comp.), any real
    double sigma;        // annualized volatility,     must be >= 0
    OptionType type;
};

// Price plus first- and second-order Greeks. See header conventions for units.
struct Greeks {
    double price;
    double delta;        // dV/dS                       (first order)
    double gamma;        // d2V/dS2                      (second order)
    double vega;         // dV/dSigma, per unit vol      (first order)
    double theta;        // dV/dt, per year              (first order)
    double rho;          // dV/dr, per unit rate         (first order)
    double vanna;        // d2V/dS dSigma                (second order)
    double volga;        // d2V/dSigma2 (vomma)          (second order)
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

// Throws std::domain_error on any input outside the model's valid domain.
// Note the comparisons are written so that NaN inputs also fail (NaN > 0 is
// false), rather than silently propagating.
inline void validate(const BsInputs& in) {
    if (!(in.S > 0.0))      throw std::domain_error("black_scholes: spot S must be > 0");
    if (!(in.K > 0.0))      throw std::domain_error("black_scholes: strike K must be > 0");
    if (!(in.T >= 0.0))     throw std::domain_error("black_scholes: time T must be >= 0");
    if (!(in.sigma >= 0.0)) throw std::domain_error("black_scholes: sigma must be >= 0");
}

}  // namespace detail

// ----------------------------------------------------------------------------
// Full price + Greeks in a single pass.
//
// d1, d2, and phi(d1) are computed once and reused across every Greek, which is
// materially cheaper than calling the individual accessors (each of which
// recomputes those shared terms). Prefer this in any loop.
// ----------------------------------------------------------------------------
inline Greeks price_and_greeks(const BsInputs& in) {
    detail::validate(in);

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
        return g;
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

    return g;
}

// ----------------------------------------------------------------------------
// Convenience scalar accessors.
//
// Each recomputes the shared terms internally, so calling several of these for
// the same option is wasteful. Use price_and_greeks() when you need more than
// one quantity (especially inside the parallel Greeks grid).
// ----------------------------------------------------------------------------
inline double price(const BsInputs& in) { return price_and_greeks(in).price; }
inline double delta(const BsInputs& in) { return price_and_greeks(in).delta; }
inline double gamma(const BsInputs& in) { return price_and_greeks(in).gamma; }
inline double vega (const BsInputs& in) { return price_and_greeks(in).vega;  }
inline double theta(const BsInputs& in) { return price_and_greeks(in).theta; }
inline double rho  (const BsInputs& in) { return price_and_greeks(in).rho;   }

}  // namespace ope

#endif  // OPE_BLACK_SCHOLES_HPP