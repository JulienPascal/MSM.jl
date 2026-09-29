# Changelog

All notable changes to MSM.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- MSM.jl now requires **Julia 1.10 or later** (1.10 is the current long-term
  support release). It is tested on Julia 1.13.
- `Project.toml` now declares compatibility bounds for every dependency, so
  that a future breaking release of a dependency cannot be installed with
  MSM.jl. Only the current major versions are allowed, in particular Optim 2,
  CSV 1, DataFrames 1, BlackBoxOptim 0.6 and OrderedCollections 2.

**These fixes change numerical results.** If you used `J_test`,
`calculate_pvalue`, `calculate_CI` or `summary_table`, re-run your inference.

- `J_test`: the critical value now comes from the chi-squared distribution
  `Chisq(df)`. It previously came from the chi distribution `Chi(df)`, whose
  quantiles are the square roots of the correct ones, so the test rejected far
  too often.
- `J_test`: the statistic is now `tData/(1 + tau)*g'Wg`, with
  `tau = tData/tSimData`, following Lee and Ingram (1991, pp. 202 and 204). It
  was `tData*(1 + tau)*g'Wg`, as printed in Ruge-Murcia (2012, eq. 13), which
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
- `msm_slices`: passed row views instead of `Vector`s to the objective function.
  A user function requiring `x::Vector{Float64}` failed silently, and the whole
  slice was equal to the penalty value.

### Removed

- Unused dependencies GLM, PlotlyJS and ParallelDataTransfer. The example
  notebooks still use GLM and ParallelDataTransfer: add them to the
  environment you run the notebooks in.
- Unused dependencies DataStructures, Logging, Pkg and SharedArrays.
  `OrderedDict` still comes from OrderedCollections. If your own code used
  DataStructures through MSM.jl, add it to your environment.
