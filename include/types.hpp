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

