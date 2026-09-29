# Changelog

All notable changes to MSM.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

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

### Removed

- Unused dependencies GLM, PlotlyJS and ParallelDataTransfer. The example
  notebooks still use GLM and ParallelDataTransfer: add them to the
  environment you run the notebooks in.
