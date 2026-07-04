# Options Pricing Engine (C++)

A production-oriented C++ implementation of core derivative pricing models,
covering European and American options via Black-Scholes (closed-form), the
Cox-Ross-Rubinstein (CRR) binomial tree, and Monte Carlo simulation. Includes
full first- and second-order Greeks, an implied volatility solver, and a
parallel Greeks calculator benchmarked against a serial baseline.

---

## Problem Statement

Derivatives desks, risk systems, and quantitative researchers all share a common
need: fast, accurate, and numerically stable option pricing across large
universes of instruments. Python implementations are convenient for research but
impose two hard constraints that matter in production. First, the interpreter
overhead and GIL make true multi-core parallelism on a hot path difficult.
Second, garbage collection means memory layout is not under the developer's
control, which matters when pricing thousands of options in a tight loop.

The specific gap this project targets is the absence of a self-contained,
well-documented C++ pricing engine that a student or junior quant developer can
read, run, extend, and reason about. Most open-source C++ quant code either
wraps QuantLib (obscuring the underlying mechanics) or demonstrates a single
method without connecting the pieces. This engine builds all three primary
pricing methods from first principles, shows exactly where each one breaks down,
and provides a direct numerical comparison between them so the tradeoffs are
empirically visible rather than just stated.

---

## Approach

### Models Implemented

**1. Black-Scholes (Closed-Form)**
- European calls and puts via the analytical B-S formula
- All five first-order Greeks (delta, gamma, vega, theta, rho) in closed form
- Implied volatility inversion via Newton-Raphson with Brent's method fallback
  for cases where the Newton step overshoots (near-zero vega, deep ITM/OTM)
- Runtime: O(1) per option; the baseline for speed comparisons

**2. Cox-Ross-Rubinstein Binomial Tree**
- European and American options on the same code path; early exercise handled
  by comparing continuation value to intrinsic at each node
- Tree depth configurable; default 500 steps balances accuracy vs. build time
- Demonstrates convergence behavior: oscillates around the B-S price as steps
  increase, converging at roughly O(1/N) rate (sawtooth, not monotone)
- Greeks computed by finite-difference bumping on the tree (bump-and-reprice).
  Because the tree is deterministic given its inputs, no random-number control
  is required here (contrast Monte Carlo below)

**3. Monte Carlo Simulation**
- Geometric Brownian Motion path simulation under the risk-neutral measure
- European payoffs use exact terminal sampling: under GBM the terminal price is
  exactly lognormal, so a single draw to expiry `T` is exact and there is no
  discretization error. The multi-step path engine (default 252 steps) is
  retained for path-dependent payoffs on the roadmap (Asian, barrier, lookback),
  where intermediate points are required. Running 252 steps for a vanilla
  European option is therefore wasted work and is not the default for those
  payoffs.
- Variance reduction via antithetic variates (pairs of `+Z` / `-Z` draws). This
  reduces variance, substantially so for payoffs that are monotone in the
  driver, at near-zero extra cost. The size of the reduction is payoff-dependent
  and is measured, not assumed (see Validation).
- Greeks via finite-difference bump-and-reprice using **common random numbers
  (CRN)**: the same set of normal draws is reused across the base and bumped
  repricings (`S +/- dS`). This is mandatory, not optional. Without CRN the base
  and bumped estimates carry independent simulation error, and the finite
  difference is dominated by Monte Carlo noise rather than the derivative,
  which makes delta and gamma unusable at practical path counts. Pathwise
  differentiation / AAD is the production upgrade path (see Tradeoffs).
- Parallelized using `std::execution::par` (C++17 parallel STL); a serial
  baseline is provided for direct benchmarking (see Parallelization).

### Implied Volatility Solver

Newton-Raphson iteration starting from a Brenner-Subrahmanyam initial guess
(an at-the-money approximation):

```
sigma_0 = sqrt(2 * pi / T) * (C / S)
```

Fallback to Brent's method when:
- Vega falls below a numerical threshold (flat payoff region)
- Newton step produces a negative or > 5 sigma trial value
- Iteration count exceeds the configured maximum

### Error Handling: Two-Tier API (noexcept grid path, throwing scalar path)

Every pricing module exposes two entry points:

- **`try_*` (noexcept, status-returning):** returns a result struct carrying
  the numeric payload plus an `ope::Status` (`Ok`, `BadInput`,
  `NoConvergence`, `Unresolvable`). On any non-Ok status the numeric fields
  are quiet NaN, never zero, so an ignored failure cannot masquerade as a
  valid flat price. This is the **only** entry point the parallel Greeks grid
  is permitted to call.
- **Throwing wrappers** (`price_and_greeks`, scalar accessors): thin
  convenience layers for scalar and interactive use that convert a non-Ok
  status into `std::domain_error`.

The split is not stylistic. Under the C++17 parallel algorithms, an exception
that escapes the element callable calls `std::terminate`
([algorithms.parallel.exceptions]); it cannot be caught around the algorithm.
A throwing pricer inside `std::execution::par` therefore turns one malformed
quote into a dead process. This was verified empirically: a 1,000-option grid
with a single `sigma = -1` row aborted with SIGABRT under the old throwing
API, and completes with exactly one `BadInput` row flagged under the
status API.

Input validation rejects non-finite values (`std::isfinite` on S, K, T, r,
sigma) in addition to domain checks (S > 0, K > 0, T >= 0, sigma >= 0). NaN
and infinity are checked explicitly because ordinary comparisons let them
slip through: `inf > 0` is true, and an unchecked NaN rate prices silently
to NaN.

Shared types (`OptionType`, `Greeks`, `Status`, `PricingResult`) live in
`types.hpp` so the tree, Monte Carlo, and IV modules use one error channel.

### Parallelization

Greeks across a 3D option grid (strike x maturity x vol) are computed using
`std::for_each` with `std::execution::par`, calling the noexcept
`try_price_and_greeks` path (see Error Handling above).

`par` (not `par_unseq`) is the correct policy here. The per-element work is a
full option repricing, which allocates working buffers (tree arrays, Monte Carlo
path state) and may throw. `par_unseq` adds the *unsequenced* guarantee, which
forbids the element function from allocating, taking locks, or throwing;
violating that is undefined behavior, not merely slow. `par` permits
multi-threaded execution while still giving each element a well-defined,
sequenced execution context, which is what our pricing callables need.

No SIMD claim is made for this loop. In current libstdc++ and libc++
implementations, `par_unseq` is largely treated like `par` and does not
auto-vectorize an arbitrary callable, so the realistic benefit here is thread
parallelism. SIMD over a flat, allocation-free numeric kernel is a separate
optimization noted as future work.

### Project Structure

```
options-pricing-engine/
├── include/
│   ├── types.hpp                # Shared types: OptionType, Greeks, Status, PricingResult
│   ├── black_scholes.hpp        # Closed-form pricer and Greeks
│   ├── binomial_tree.hpp        # CRR tree, European and American
│   ├── monte_carlo.hpp          # GBM simulation, antithetic variates, CRN Greeks
│   ├── implied_vol.hpp          # Newton-Raphson + Brent solver
│   └── greeks_grid.hpp          # Parallel Greeks across option universe
├── src/
│   ├── black_scholes.cpp
│   ├── binomial_tree.cpp
│   ├── monte_carlo.cpp
│   ├── implied_vol.cpp
│   └── greeks_grid.cpp
├── tests/
│   ├── test_black_scholes.cpp   # Validates against known analytical values
│   ├── test_binomial_tree.cpp   # Convergence check vs. B-S price
│   ├── test_monte_carlo.cpp     # Distributional checks, standard error bounds
│   └── test_implied_vol.cpp     # Round-trip: price -> IV -> reprice
├── benchmarks/
│   └── bench_parallel_greeks.cpp  # Serial vs. par wall-clock comparison
├── CMakeLists.txt
└── README.md
```

### Build Requirements

| Dependency | Version | Purpose |
|---|---|---|
| C++ standard | C++17 (minimum) | `std::execution`, `std::optional` |
| C++ standard (optional) | C++20 | `std::span`, used only when compiling in C++20 mode |
| CMake | >= 3.20 | Build system |
| Intel TBB / libtbb-dev | Any recent (oneTBB) | Parallel STL backend for GCC/libstdc++ on Linux/macOS |
| Catch2 (optional) | >= 3.0 | Unit test framework |

> **Note on `std::span`:** `std::span` is a C++20 feature, not C++17. The engine
> targets C++17 as a floor and does not require `std::span` in that mode. If you
> build in C++20 mode, `std::span` is used for non-owning buffer views; otherwise
> equivalent pointer-plus-size or `gsl::span` style interfaces are used.

```bash
# Ubuntu / Debian
sudo apt-get install libtbb-dev cmake build-essential

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run tests
./build/tests/test_black_scholes
./build/tests/test_implied_vol

# Run benchmark
./build/benchmarks/bench_parallel_greeks
```

> **Note on parallel STL backends (read before trusting the benchmark):**
> `std::execution::par` requires a parallel STL backend at link time, and
> availability differs sharply by toolchain.
>
> - **GCC / libstdc++ (Linux, or GCC on macOS):** requires Intel TBB
>   (`libtbb-dev` / oneTBB). Link against TBB or the parallel policies fall back
>   to serial.
> - **MSVC:** the parallel backend is built in. No extra dependency.
> - **Clang / libc++ (default on macOS):** parallel algorithms are **not**
>   provided by default. Apple's Xcode libc++ ships none, and upstream libc++
>   requires `LIBCXX_ENABLE_PARALLEL_ALGORITHMS` plus a PSTL backend selected at
>   libc++ build time. On a default Mac toolchain, `std::execution::par` may
>   compile but run serially, or fail to link.
>
> Recommended portable paths: on Linux use GCC + TBB; on macOS use GCC +
> Homebrew TBB, or oneDPL / oneTBB. Whatever the toolchain, confirm the parallel
> benchmark actually beats the serial baseline before reporting a speedup. A
> silent serial fallback produces a "1x speedup" that looks like a bug but is
> really a missing backend.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `S` | 100.0 | Spot price |
| `K` | 100.0 | Strike price |
| `T` | 1.0 | Time to expiry (years) |
| `r` | 0.05 | Risk-free rate (continuously compounded) |
| `sigma` | 0.20 | Volatility (annualized) |
| `N_steps` | 500 | Binomial tree depth |
| `N_paths` | 100,000 | Monte Carlo path count |
| `N_timesteps` | 1 (European), 252 (path-dependent) | MC time steps per path. European payoffs use a single exact terminal step; multi-step is used only when the payoff needs intermediate points |
| `iv_tol` | 1e-6 | Newton-Raphson convergence tolerance |
| `iv_max_iter` | 100 | Maximum IV solver iterations |

---

## Tradeoffs

### Black-Scholes vs. Binomial Tree vs. Monte Carlo

| Dimension | Black-Scholes | Binomial Tree | Monte Carlo |
|---|---|---|---|
| Speed | Fastest (O(1)) | Moderate (O(N^2) nodes) | Slowest (O(P x T)) |
| American options | No | Yes | Yes (with LSM) |
| Path-dependent payoffs | No | No | Yes |
| Parallelizable | Trivially | Poorly (tree has data deps) | Highly |
| Accuracy | Exact (under assumptions) | Converges to B-S | Converges as 1/sqrt(P) |

### Newton-Raphson vs. Brent's Method (IV Solver)

Newton-Raphson converges quadratically near the root but requires a vega
evaluation at each step and can diverge for deep ITM/OTM options where vega is
near zero. Brent's method is slower (linear-superlinear convergence) but is
guaranteed to converge given a valid bracket. The hybrid approach here uses
Newton first and falls back to Brent only when Newton becomes numerically
unreliable, keeping average-case cost low without sacrificing robustness.

### Antithetic Variates vs. Quasi-Random (Sobol) Sequences

Antithetic variates reduce variance at near-zero extra cost: each path generates
both a `+Z` and `-Z` draw, and the negative correlation between paired payoffs
cancels variance. The reduction is largest for payoffs that are monotone in the
driver and shrinks for non-monotone payoffs, so the realized improvement is
measured per payoff rather than assumed. Sobol sequences (low-discrepancy
quasi-Monte Carlo) achieve faster asymptotic convergence
(O(log(N)^d / N) vs. O(1/sqrt(N))) but require a correct scrambled Sobol
generator, dimension management across the path, and care around the
dimensionality limit. Antithetic variates are implemented here as the reliable
baseline. Sobol is documented as a known upgrade path.

### Parallel STL vs. OpenMP vs. CUDA

`std::execution::par` is used because it is standard C++17, requires no compiler
pragma or external annotation, and is portable across MSVC, GCC, and (with a
parallel backend) Clang. The cost is dependency on a parallel STL backend (TBB
for libstdc++; an explicitly enabled PSTL for libc++; built in for MSVC), which
means availability is toolchain-dependent (see the build note above). OpenMP
would require `#pragma omp parallel for` annotations and an `-fopenmp` flag but
has wider, more predictable compiler support and no libc++ gap. CUDA would
achieve the largest speedup on GPU hardware but adds non-trivial build
complexity and an NVIDIA hardware dependency. For a CPU-bound portfolio project,
parallel STL is a reasonable tradeoff, with OpenMP as the fallback if the libc++
backend situation is a blocker on a target machine.

### Bump-and-Reprice Greeks vs. Pathwise / AAD

All Greeks outside Black-Scholes closed form are computed by finite-difference
bumping: reprice at `S + dS` and `S - dS`, take the centered difference. For
Monte Carlo this is done with common random numbers (the same draws across base
and bumped repricings) so the finite difference reflects the derivative rather
than simulation noise. This is correct and simple but scales as
O(num_greeks x pricing_cost). Pathwise differentiation and Adjoint Algorithmic
Differentiation (AAD) compute all Greeks in a single pass at roughly the cost of
one pricing call, which is how production risk engines work. AAD is noted as the
primary upgrade path for the Monte Carlo Greeks, since it is the standard used
by major banks and the main advertised feature of libraries like QuantLib's
AAD-enabled forks.

---

## Known Limitations

- **Deep-wing precision floor applies to the price formula, not just IV.**
  The B-S price is formed as `omega * (S*N(omega*d1) - K*exp(-rT)*N(omega*d2))`.
  For deep out-of-the-money options both terms are tiny and nearly cancel:
  absolute error stays near machine epsilon, but relative error grows as the
  price shrinks (a 5-sigma OTM call prices at ~1e-21). The IV solver's
  documented wing failures are this same double-precision floor seen from the
  inversion side. Jaeckel's "Let's Be Rational" formulation of the Black
  function is the production fix for both, and remains the documented
  upgrade path.
- **No dividends.** Continuous dividend yield can be incorporated into B-S by
  replacing `r` with `r - q`. Discrete dividends require tree adjustment or
  jump conditions in the PDE. Not implemented.
- **Constant volatility.** The B-S assumption. Local vol (Dupire) and stochastic
  vol (Heston, SABR) are the correct next steps for smile-consistent pricing.
- **No term structure.** A flat risk-free rate is used. A real rates desk would
  bootstrap a yield curve and discount each cashflow at its tenor-matched rate.
- **Monte Carlo Greeks are slow.** Bump-and-reprice at 100k paths x 5 Greeks is
  expensive even with common random numbers. CRN fixes the *accuracy* problem
  (noise), not the *cost* problem; pathwise / AAD is the production solution for
  cost.
- **Parallel backend is toolchain-dependent.** On Clang / libc++ the parallel
  policies may run serially without a backend. The benchmark must be checked
  against the serial baseline on the actual target machine, not assumed.
- **No exotic payoffs yet.** The multi-step Monte Carlo engine exists to support
  path-dependent options (Asian, barrier, lookback), but those payoffs are not
  implemented. Until they are, European pricing uses exact terminal sampling.
- **American MC requires LSM.** The binomial tree handles American options
  correctly. Monte Carlo requires the Longstaff-Schwartz method (regression-based
  continuation value estimation), which is a separate implementation.

---

## Validation

Each model is validated at build time:

- **B-S closed form:** Checked against the put-call parity identity
  `C - P = S - K * exp(-rT)` and known textbook values (Hull, Ch. 15).
- **Binomial tree:** Convergence test confirms the tree price is within 0.01 of
  the B-S price at N = 500 steps for a standard ATM European option, and that
  the error envelope decreases at roughly O(1/N).
- **Monte Carlo (price):** Standard error of the price estimate verified to be
  within the expected `sigma_payoff / sqrt(N_paths)` band at the 95% confidence
  level. The antithetic variance reduction is reported as a measured ratio of
  estimator variance (antithetic vs. plain) for the test payoff, rather than
  assumed.
- **Monte Carlo (Greeks):** Delta and gamma computed via CRN bump-and-reprice
  are checked against the Black-Scholes closed-form Greeks for a European option,
  asserting agreement within a tolerance that scales with `1/sqrt(N_paths)`. A
  control run without CRN is included to demonstrate that the non-CRN estimate
  fails this tolerance, documenting why CRN is required.
- **IV solver:** Round-trip test: compute B-S price from a known vol, invert to
  recover vol, assert `|recovered - known| < iv_tol`.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate
  Liabilities.* Journal of Political Economy.
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified
  Approach.* Journal of Financial Economics.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.*
  Springer. (Common random numbers, antithetic variates, pathwise Greeks.)
- Longstaff, F. & Schwartz, E. (2001). *Valuing American Options by Simulation.*
  Review of Financial Studies.
- Hull, J. (2022). *Options, Futures, and Other Derivatives.* 11th ed. Pearson.
- NVIDIA Accelerated Quant Finance (ISO C++ parallelism examples):
  https://github.com/NVIDIA/accelerated-quant-finance
- QuantLib (C++ financial library, AAD-enabled forks):
  https://www.quantlib.org

---

## Document Revision Log

This spec was revised to correct technical errors in the prior draft. Changes:

1. **Monte Carlo Greeks now specify common random numbers (CRN).** The prior
   draft described bump-and-reprice without fixing the random draws across bumps.
   That makes the finite difference dominated by simulation noise. CRN is now
   required, and a non-CRN control is added to validation to prove the point.
2. **`std::span` corrected to C++20.** It was previously listed under C++17.
   The standard floor is C++17 (`std::execution`, `std::optional`); `std::span`
   is gated to C++20 builds.
3. **Parallel policy changed from `par_unseq` to `par`.** The per-element
   pricing callable allocates and may throw, which is undefined behavior under
   the unsequenced guarantee of `par_unseq`. The unsubstantiated SIMD claim was
   removed.
4. **Portability claim corrected.** Clang / libc++ does not ship parallel
   algorithms by default (Apple Xcode libc++ has none; upstream requires opt-in).
   The build note now states this and gives portable paths plus a silent-serial
   warning.
5. **Antithetic variates reworded.** "Halves standard error" was overstated
   (halving SE needs roughly a 4x variance cut). Now stated as variance
   reduction, payoff-dependent, and measured in validation.
6. **European Monte Carlo uses exact terminal sampling.** 252 timesteps for a
   vanilla European payoff is wasted work since the GBM terminal price is exactly
   lognormal. The multi-step engine is retained explicitly for path-dependent
   payoffs on the roadmap.
7. **Two-tier error API introduced; throwing pricer removed from the grid
   path.** The prior design had `validate()` throw `std::domain_error`
   unconditionally. An exception escaping the element callable of a C++17
   parallel algorithm calls `std::terminate`, so one bad row killed the whole
   grid (verified: SIGABRT on a 1,000-option grid with a single bad sigma).
   `try_price_and_greeks()` is now the noexcept, status-returning grid entry
   point; the throwing API remains as a scalar convenience wrapper. Shared
   types promoted to `types.hpp`.
8. **Input validation hardened to reject non-finite values.** The prior
   `!(x > 0)` comparisons rejected NaN for S/K/T/sigma but left `r` entirely
   unchecked (NaN rate priced silently to NaN) and accepted infinities for
   every field (`inf > 0` is true). All five inputs now require
   `std::isfinite` in addition to the domain checks.
9. **Deep-wing cancellation documented on the pricing side.** The limitation
   was previously framed as an IV solver issue only. It is the same
   double-precision floor in the price formula itself, and is now listed
   under Known Limitations with the Jaeckel formulation as the shared fix.
