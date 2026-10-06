# Changelog

All notable changes to MSM.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.2.0] - 2026-10-06

Starting with the branch `julia_1.13`, the bug fixes and the update to Julia
1.13 were made with the assistance of Claude Opus 5.5 (Anthropic), used through
Claude Code. 

### Added

- `check_problem(sMMProblem)` checks that a problem is set up correctly, and
  throws an informative error otherwise. `msm_optimize!` and `msm_multistart!`
  call it first; skip it with `check = false`. The objective function returns
  the penalty value on any error, so such mistakes previously gave a flat
  objective function, and the optimization ran to completion without an error.
  It checks that the priors, the empirical moments, the simulation function and
  the objective function are set, and that `W` is k×k for k empirical moments
  (the default `W` is 1×1). It then simulates the moments once, at the initial
  values of the priors, outside of the objective function. It fails if the
  simulation throws (the original error is logged with its stack trace), does
  not return every empirical moment, or returns non-finite moments. It warns
  about simulated moments that are not empirical moments (they are ignored),
  and when the distance at the initial values exceeds a finite `penaltyValue`
  (failed simulations would then look better than successful ones).

- `MSMOptions(lambda = ...)`: number of points evaluated per generation by the
  natural evolution strategies (`:dxnes`, the default, `:xnes` and
  `:separable_nes`), and `nes_lambda` to compute it. The points of a generation
  are evaluated in parallel, and the next generation starts when all of them
  are done. BlackBoxOptim's default depends only on the number of parameters
  (8 for 5 parameters, 10 for 10 parameters with `:dxnes`), so with more
  workers than that, most of them were idle. With `lambda = 0` (default), it is
  now rounded up to a multiple of the number of workers (`:dxnes` requires an
  even value: with an odd number of workers, one worker is idle). With
  `verbose = true`, `msm_optimize!` logs the value used and the number of
  generations it implies. **With several workers, this changes the global
  optimization, and so its results.** `maxFuncEvals` still counts
  evaluations: a larger `lambda` means fewer generations for the same budget,
  so scale `maxFuncEvals` with `lambda`. With one worker, nothing changes.
- `is_nes_optimizer(s::Symbol)`.
- A notebook estimating a small DSGE model (a real business cycle model written
  with MacroModelling.jl): `notebooks/models/RBC.ipynb`, with its own environment.
  It recovers known parameters from simulated data, and checks the coverage of
  the confidence intervals and the size of the J-test in a Monte Carlo
  experiment.
- A notebook estimating the same model with Dynare.jl instead of
  MacroModelling.jl to solve and simulate it:
  `notebooks/models/dynare/RBCDynare.ipynb`, with its own environment and the
  model file `RBC.mod`. Each worker loads the model in its own temporary folder,
  because Dynare.jl writes files next to the model file. An optional last
  section runs the `method_of_moments` command of Dynare 6 in Octave (as an
  external program, with the model file `RBC_mom.mod`) and compares its
  estimates with those of MSM.jl. It is skipped when Octave is not installed.

### Changed

- `populationSize` is ignored by the NES optimizers (it only applies to
  differential evolution). `MSMOptions` now warns when it is set together with
  one of them. With `verbose = true` and several workers, `msm_optimize!` also
  notes that differential evolution evaluates about one new point at a time,
  so it gets essentially no speed-up from several workers.
- **The default `penaltyValue` is now `Inf`** (it was 999999). With a finite
  penalty, a real distance can exceed it when the moments are badly scaled, and
  failed points then look better than successful ones. Optim and BlackBoxOptim
  handle `Inf` values. The default `thresholdStartingValue` is still
  `penaltyValue/10`, so it is now `Inf` too: any successful point is a valid
  starting value for `msm_multistart!`. Pass `penaltyValue = 999999.0` to
  restore the previous behavior (e.g. to feed failed points to a surrogate).
- **Local minimizations (`msm_refine_globalmin!`, `msm_localmin`,
  `msm_multistart!`) now stop after `maxFuncEvals` evaluations of the objective
  function**, finite-difference gradients included (at the end of the
  iteration during which the budget is reached). `maxFuncEvals` was passed to
  Optim as the maximum number of *iterations*, and with finite-difference
  gradients each iteration costs several evaluations: `maxFuncEvals = 1000`
  could mean tens of thousands of simulations. Optim's `f_calls_limit` could
  not be used, because it does not count the evaluations of finite
  differences. A local minimization stopped by the budget does not count as
  converged.
- `msm_multistart!`: if none of the local minimizations converged, the best
  finite local minimum is now used, with a message saying so. Previously, no
  result was stored.
- MSM.jl now requires **Julia 1.10 or later** (1.10 is the current long-term
  support release). It is tested on Julia 1.13.
- `Project.toml` now declares compatibility bounds for every dependency, so
  that a future breaking release of a dependency cannot be installed with
  MSM.jl. Only the current major versions are allowed, in particular
  DataFrames 1 and BlackBoxOptim 0.6 (with the exceptions below).
- Both **Optim 1 (1.13 or later) and Optim 2** are allowed, so that MSM.jl can
  be installed next to packages that still require Optim 1 (e.g.
  MacroModelling.jl). The test suite passes with Optim 1.13.3 and Optim 2.3.2.
  The results of local minimizations can differ slightly between the two.
  Known issue: with Optim 2 only, `localOptimizer = :AcceleratedGradientDescent`
  can diverge (it does on the Rosenbrock function; it converges with Optim 1).
  Prefer `:LBFGS` (the default).
- Both **OrderedCollections 1 and 2** are allowed, for the same reason:
  MacroModelling.jl's Bayesian estimation requires Turing.jl 0.30 to 0.45,
  whose dependencies require OrderedCollections 1. The test suite passes with
  OrderedCollections 1.8.2 (with Optim 1.13.3) and 2.0.1.
- Both **CSV 0.10 and CSV 1** are allowed, so that MSM.jl can be installed next
  to Dynare.jl (version 0.10.4 requires CSV 0.10). MSM.jl only uses
  `CSV.File`, which is the same in both. The test suite passes with CSV 0.10.17
  and 1.1.0.
- `msm_minimizer`, `msm_minimum`, `msm_local_minimizer` and `msm_local_minimum`
  now throw an informative error when called before the corresponding
  optimization (or after `msm_multistart!` if no local minimization returned a
  finite value).
  They previously returned the result of a dummy optimization of the
  Rosenbrock function. The fields `bbSetup`, `bbResults` and `optimResults` of
  `MSMProblem` are now `nothing` until they are set.
- Loading MSM.jl no longer runs a BlackBoxOptim and an Optim optimization at
  precompile time (they were only used to create those dummy default values).
- `convert_to_optim_algo` and `convert_to_fminbox` no longer use
  `eval(Meta.parse(...))`, and throw an error for names that are not supported
  local optimizers.

**These fixes change numerical results.** If you used `J_test`,
`calculate_pvalue`, `calculate_CI` or `summary_table`, re-run your inference.

- `J_test`: the critical value now comes from the chi-squared distribution
  `Chisq(df)`. It previously came from the chi distribution `Chi(df)`, whose
  quantiles are the square roots of the correct ones, so the test rejected far
  too often.
- `J_test`: the statistic is now `tData/(1 + tau)*g'Wg`, with
  `tau = tData/tSimData`, following Lee and Ingram (1991, pp. 202 and 204). It
  was `tData*(1 + tau)*g'Wg`, as printed in Ruge-Murcia (2012, eq. 13, probably a typo), which
  overstates J by a factor `(1 + tau)^2`: 4 when the simulated and observed
  series have the same length. Combined with the previous point, a correctly
  specified model was rejected 18% to 48% of the time at the 5% level in
  simulations, instead of 5%. The test still requires `W` to converge to
  `inv(Sigma0)`.
- `calculate_pvalue` and the `Pr(>|t|)` column of `summary_table`: the
  two-sided p-value is now `2*ccdf(Normal(), abs(t))`. It was wrong (between 1
  and 2) for negative t-statistics.
- `calculate_CI` and the confidence intervals of `summary_table`: the interval
  now uses the `1 - alpha/2` quantile of the normal distribution. With
  `alpha = 0.05`, it previously reported a 90% confidence interval instead of a
  95% one.

### Fixed

- `msm_optimize!` no longer prints a leftover debug message ("hello") on
  every worker.
- `msm_optimize!(...; verbose = false)` now hides BlackBoxOptim's progress
  trace and MSM's messages. The `verbose` keyword was previously ignored.
  `set_global_optimizer!` and `set_bbSetup!` accept the same keyword.
- `msm_multistart!` with 2 or more workers and without user-provided `x0`:
  candidate starting values were matched with the distances of *other*
  candidates, because results were collected in the order workers finished.
  Invalid points could therefore be used as starting values, and sorting them
  by distance was arbitrary. `search_starting_values` now evaluates each batch
  of candidates with `pmap` (results in input order) and keeps each point's
  distance with it. Each search round now evaluates the whole batch of
  candidates, not only one per worker. The saved `starting_values_*.bson` and
  `starting_distances_*.bson` files now line up. This also removes a
  `BoundsError` when there were more workers than requested starting values.
- `msm_multistart!` with 2 or more workers: the logged "best starting value"
  could belong to another worker. The selected minimizer itself was correct.
- `msm_multistart!`: the logged number of converged local minimizations was
  wrong.
- `msm_multistart!` with `nums < nworkers()`, and `search_starting_values` with
  an invalid `gridType`, threw an `UndefVarError` instead of an informative
  error.
- Objective function: if the user's function did not return one of the
  empirical moments, the optimization crashed with a `KeyError`. It now returns
  the penalty value. NaN or infinite distances are also replaced by the penalty
  value instead of being passed to the optimizers.
- `simulate_empirical_moments_array`, used by `calculate_D` (and so by
  `calculate_Avar!`, the standard errors and `summary_table`), returned the
  moments in the order in which the user's function inserted them in its
  `OrderedDict`, including moments that are not empirical moments. If that
  order differed from the order of the empirical moments, the rows of the
  Jacobian did not match `W` and `Sigma0`, and the standard errors were
  silently wrong. The array now follows the order of the empirical moments.
- `msm_multistart!(...; verbose = false)`: the local minimizations still logged
  their messages. The `verbose` keyword is now passed to them.
- `msm_multistart!`: a local minimization that threw an error was followed by a
  confusing `MethodError` message. It is now reported once ("local
  minimization from starting value ... failed: ..."), and its entry in the
  returned list is `nothing`.
- `msm_slices`: for a parameter equal to 0, the slice had width 0 (it is
  `offset*|θ|`), so all its points were identical. For a parameter that is
  (numerically) zero, the width is now `offset` times the width of its prior.
- `get_now()` (the default `saveName` of `MSMOptions`) read the clock four
  times, so the date, hour, minute and second could come from different
  instants (e.g. across midnight). It now reads the clock once.
- Docstrings that did not match their function: `calculate_D` (a `method`
  keyword that does not exist), `calculate_se`, `calculate_t` and
  `calculate_pvalue` (a `tSimulation` argument that does not exist, and missing
  `theta0` and `i` arguments), `linspace` (arguments in the wrong order),
  `latin_hypercube_sampling(mins, maxs, n)` (it returns an `n`×`dims` matrix,
  not `dims`×`n`) and `create_upper_bound` (described as a lower bound).
- `msm_slices`: passed row views instead of `Vector`s to the objective function.
  A user function requiring `x::Vector{Float64}` failed silently, and the whole
  slice was equal to the penalty value.
- Spelling mistakes in error and log messages: "caclulate" in the errors of
  `calculate_se`, `calculate_t`, `calculate_pvalue`, `calculate_CI` and
  `summary_table`, and "occured" in the messages logged when the simulation or
  the distance fails (they now read "An error occurred ...").

### Removed

- The empty file `src/api.jl`.

- Unused dependencies GLM, PlotlyJS and ParallelDataTransfer. The example
  notebooks still use GLM and ParallelDataTransfer: add them to the
  environment you run the notebooks in.
- Unused dependencies DataStructures, Logging, Pkg and SharedArrays.
  `OrderedDict` still comes from OrderedCollections. If your own code used
  DataStructures through MSM.jl, add it to your environment.
