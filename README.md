# MethodOfSimulatedMoments.jl


| **Documentation**  | **Build Status** | **Coverage** | **Citation** |
|:-:|:-:|:-:|:-:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JulienPascal.github.io/MethodOfSimulatedMoments.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JulienPascal.github.io/MethodOfSimulatedMoments.jl/dev)|[![CI](https://github.com/JulienPascal/MethodOfSimulatedMoments.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JulienPascal/MethodOfSimulatedMoments.jl/actions/workflows/ci.yml?query=branch%3Amain)|[![codecov](https://codecov.io/gh/JulienPascal/MethodOfSimulatedMoments.jl/graph/badge.svg?branch=main)](https://codecov.io/gh/JulienPascal/MethodOfSimulatedMoments.jl)|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23208575.svg)](https://doi.org/10.5281/zenodo.23208575)|


`MethodOfSimulatedMoments.jl` is a package designed to facilitate the estimation of economic models
via the [Method of Simulated Moments](https://en.wikipedia.org/wiki/Method_of_simulated_moments).

## Why

An economic theory can be written as a system of equations that depends on primitive
parameters. The aim of the econometrician is to **recover the unknown parameters**
using **empirical data**. One popular approach is to maximize the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function).
Yet in many instances, the likelihood function is intractable. An alternative approach to estimate the unknown parameters is to minimize a (weighted) distance between
the empirical [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) and their theoretical counterparts.

When the function mapping the set of parameter values to the theoretical moments (the *expected response function*) is known, this method is called
the [Generalized Method of Moments](https://en.wikipedia.org/wiki/Generalized_method_of_moments).
However, in many interesting cases the *expected response function* is unknown. This issue may be circumvented by simulating the expected response function, which is often an easy task. In this case, the method is called the [Method of Simulated Moments](https://en.wikipedia.org/wiki/Method_of_simulated_moments).

## Philosophy

`MethodOfSimulatedMoments.jl` is being developed with the following constraints in mind:
1. Parallelization **within the expected response function** is difficult
to achieve. This is generally the case when working with the simulated method of moments, as the simulated time series are often serially correlated.
2. Thus, the **minimizing algorithm** should be able to run in **parallel**.
3. The minimizing algorithm should search for a **global minimum**, as the
objective function may have multiple local minima.
4. **Do not reinvent the wheel**. Excellent minimization packages already exist in
the Julia ecosystem. This is why `MethodOfSimulatedMoments.jl` relies on [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) and [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to perform the minimization.


## Installation

```julia
pkg> add https://github.com/JulienPascal/MethodOfSimulatedMoments.jl.git
```

Then load it with `using MethodOfSimulatedMoments`. Before version 0.2.0, the package was called `MSM.jl`: the functions and types keep their names (`MSMProblem`, `msm_optimize!`...), only `using MSM` becomes `using MethodOfSimulatedMoments`.

## Usage

See the following notebooks:
* [`notebooks/LinearModel.ipynb`](notebooks/LinearModel.ipynb) for an **introduction** to the package
* [`notebooks/LinearModelCluster.ipynb`](notebooks/LinearModelCluster.ipynb) to see how to use the package on a **cluster**
* [`notebooks/models/RBC.ipynb`](notebooks/models/RBC.ipynb): estimate a simple RBC model using [MacroModelling](https://github.com/thorek1/MacroModelling.jl) to solve and simulate the economic model, while MethodOfSimulatedMoments.jl handles the estimation procedure (function minimization and inference).
* [`notebooks/models/dynare/RBCDynare.ipynb`](notebooks/models/dynare/RBCDynare.ipynb): the same estimation, using [Dynare.jl](https://github.com/DynareJulia/Dynare.jl) to solve and simulate the model.
---

## Experiments

See the following notebooks for experimental features and to see how MethodOfSimulatedMoments.jl can interact with the other estimation packages in the Julia ecosystem:
* [`notebooks/ABC.ipynb`](notebooks/ABC.ipynb): [Approximate Bayesian computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation)
* [`notebooks/Surrogates.ipynb`](notebooks/Surrogates.ipynb): surrogate-based optimization with [Surrogates.jl](https://github.com/SciML/Surrogates.jl)
* [`notebooks/SurrogatesParallel.ipynb`](notebooks/SurrogatesParallel.ipynb): surrogate-based optimization *in parallel* with [Surrogates.jl](https://github.com/SciML/Surrogates.jl)
* [`notebooks/MSM-MCMC.ipynb`](notebooks/MSM-MCMC.ipynb): reframe the MSM problem as a Laplace-type estimator (see [Chernozhukov and Hong, 2003](https://www.sciencedirect.com/science/article/abs/pii/S0304407603001003)) and then use Markov Chain Monte Carlo methods with [AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl).
* [`notebooks/BOSS.ipynb`](notebooks/BOSS.ipynb): [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) with Semiparametric Surrogate using [BOSS.jl](https://github.com/soldasim/BOSS.jl/).

---

## Related Packages

* [SMM.jl](https://github.com/floswald/SMM.jl): a package to do SMM using MCMC algorithms in parallel
