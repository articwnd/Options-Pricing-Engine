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
 
#include <cmath> // std::erfc, std::exp,, std::sqrt
#include <stdexcept> //std::domain_error

namespace ope { Call, Put };

// All inputs for a single European option.
struct BsInput {
    double S; // spot price, must be > 0
    double K; // strike price, must be > 0
    double T; // time to expiry in years, must be > 0
    double r;
    double sigma;
    OptionType type;
}

// Price plus first- and second-order Greeks
struct Greeks {
    double price;
    double delta;        // dV/dS                       (first order)
    double gamma;        // d2V/dS2                      (second order)
    double vega;         // dV/dSigma, per unit vol      (first order)
    double theta;        // dV/dt, per year              (first order)
    double rho;          // dV/dr, per unit rate         (first order)
    double vanna;        // d2V/dS dSigma                (second order)
    double volga;        // d2V/dSigma2 (vomma)          (second order)
}

namespace detail { 
    // 1/sqrt(2) and 1/sqrt(2*pi) as exact literals. Hardcoded rather than computed
    // becaused std::sqrt is not guaranteed constexpr before C++26, and this avoids
    // pulling in M_PI or std::numbers 
    inline constexpr double INV_SQRT2 = 0.7071067811865475244;
    inline constexpr double INV_SQRT_2PI = 0.3989422804014326779;

    // standard normal CDF. The erfc form 0.5(erfc(-x/sqrt2) preserves accuracy in
    // left tail, unlike 0.5*(1 + erf(x/sqrt2)) which suffers cancellation there.
    inline double norm_cdf(double x) no except {
        return 0.5 * std::erfc(-x * INV_SQRT2);
    }
}

// Standard normal PDF
inline double norm_pdf(double x) noexcept {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// +1 for a call, -1 for a put. Lets price/greeks share one code path via the
// indentity price = omega * (S*N(omega*d1) - K*e^t{-rT}*N(omega*d2)).
inline double omega_of(OptionType t) noexcept {
    return(t == OptionType::Call) ? 1.0 : -1.0;
}

// Throws std::domain_error on any input outside the model's valid domain.
// Note the comparisons are written so that NaN inputs also fail (NaN > 0 is 
// false), rather than silently propagating.
inline void validate(const BsInputs& in) {
    if (!(in.S > 0.0)) throw std::domain_error("black_scholes: spot S must be > 0");
    if (!(in.K > 0.0)) throw std::domain_error("black_scholes: strike K must be > 0");
    if (!(in.T >= 0.0)) throw std::domain_error("black_scholes: time T must be >= 0");
    if (!(in.sigma >= 0.0)) throw std::domain_error("black_scholes: sigma must be >= 0");
}




