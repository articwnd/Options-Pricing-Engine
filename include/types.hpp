#ifndef OPE_TYPES_HPP
#define OPE_TYPES_HPP

// ============================================================================
// types.hpp

// Shared types for the pricing engine. OptioType and Greeks were promoted 
// here from black_scholes.hpp so the binomial tree, Monte Carlo, and IV
// modules can share them without a dependency on the closed-form pricer.

// Status is the engine-wide error channel. The design rule:
//     - Grid / hot-path entry points ar noexcepy and return a status.
//       An excpetion escaping the element calleable of a C++17 parrallel
//       algorithm calls std::terminate ([algorithms.parrallel.exceptions]);
//       one malformed quote must not kill a million-option grid.
//     - scalar convenience entry points may throw. They are thin wrappers
//       that convert a non-Ok status into std::error.

// On any non-Ok status, numeric fields of the accompanying result are set to 
// quiet NaN, never zero. Zero is a valid price/Greek and silently using it 
// downstream would hie the failure; NaN propagates loudly.
// ============================================================================

#include <limits> // std::numerica_limits

namespace ope {

enum class OptionType { Call, Put };

// Engine-wide status codes. Kept intentionally small; extend only when a new 
// module has a failure mode not expressible below.
enum class Status : unsigned char {
    Ok = 0,
    BadInput,       // input outside the model domain, or non-finite (NaN/inf)
    NoConvergence,  // iterative solver exhausted iterations (IV: Newton+Brent)
    Unresolvable    // answer exists but lies below the numerical precision
                    // floor (deep-wing IV: vega underflow / catastrophic
                    // cancellation in the price formula). Retrying cannot
                    // help; this is a property of double precision, not of
                    // the solver. See implied_vol.hpp.

};

// Human-readable status, for logs and test diagnositics.
inline constexpr const char* to_string(Status s) noexcept {
    switch (s) { 
        case Status::Ok:                    return "Ok";
        case Status::BadInput:              return "BadInput";
        case Status::NoConvergence:         return "NoConvergence";
        case Status::Unresolvable:          return "Unresolvable";
    }
    return "UnknownStatus"; // unreachable for valid enum values
}

// Price plus first- and second-order Greeks.
// Units are raw mathematical derivatives; see black_scholes.hpp header for 
// quoting conventions (vega per unit vol, theta per year, rho per unit rate).
struct Greeks {
    double price;
    double delta;   // dV/dS                    (first order)
    double gamma;   // d2V/dS2                  (second order)
    double vega;    // dV/dSigma, per unit vol  (first order)
    double theta;   // dV/dt, per year          (first order)
    double rho;     // dV/dr, per unit rate     (first order)
    double vanna;   // d2V/dS dSigma            (second order)
    double volga;   // d2V/dSigma2 (vomma)      (second order)

};

// Returns a Greeks structure with every field set to quiet NaN. Used as the
// payload for any non-Ok result so failures cannot masquerade as prices. 
inline constexpr Greeks nan_greeks() noexcept {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    return Greeks{nan, nan, nan, nan, nan, nan, nan, nan};
};

// Result of a noexcept priicng call. Check 'status' (or use operator bool)
// before consuming 'greeks'; on non-Ok status all fields are NaN.
struct PricingResult {
    Greeks greeks;
    Status status;

    explicit operator bool() const noexcept { return status == Status::Ok; }
    bool ok() const noexcept { return status == Status::Ok; }
};

}; // namespace ope

#endif // OPE_TYPES_HPP
