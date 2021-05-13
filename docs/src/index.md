# MSM.jl


| **Documentation**  | **Build Status** | **Coverage** |
|:-:|:-:|:-:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JulienPascal.github.io/MSM.jl/dev)|[![Build Status](https://github.com/JulienPascal/MSM.jl/workflows/CI/badge.svg)](https://github.com/JulienPascal/MSM.jl/actions)|[![codecov.io](https://codecov.io/gh/JulienPascal/MSM.jl/branch/julia_1.5/graphs/badge.svg)](https://codecov.io/gh/JulienPascal/MSM.jl/branch/julia_1.5/)|
||[![Build Status](https://travis-ci.com/JulienPascal/MSM.jl.svg?branch=main)](https://travis-ci.com/JulienPascal/MSM.jl)||



`MSM.jl` is a package designed to facilitate the estimation of economic models
via the [Method of Simulated Moments](https://en.wikipedia.org/wiki/Method_of_simulated_moments).

---

## Why

An economic theory can be written as a system of equations that depends on primitive
parameters. The aim of the econometrician is to **recover the unknown parameters**
using **empirical data**. One popular approach is to maximize the [likelihood funtion](https://en.wikipedia.org/wiki/Likelihood_function).
Yet in many instances, the likelihood function is intractable. An alternative approach to estimate the unknown parameters is to minimize a (weighted) distance between
the empirical [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) and their theoretical counterparts.

When the function mapping the set of parameter values to the theoretical moments (the *expected response function*) is known, this method is called
the [Generalized Method of Moments](https://en.wikipedia.org/wiki/Generalized_method_of_moments).
However, in many interesting cases the *expected response function* is unknown. This issue may be circumvented by simulating the expected response function, which is often an easy task. In this case, the method is called the [Method of Simulated Moments](https://en.wikipedia.org/wiki/Method_of_simulated_moments).

---

## Philosophy

`MSM.jl` is being developed with the following constraints in mind:

1. Parallelization **within the expected response function** is difficult to achieve. This is generally the case when working with the simulated method of moments, as the simulated time series are often serially correlated.
2. Thus, the **minimizing algorithm** should be able to run in **parallel**
3. The minimizing algorithm should search for a **global minimum**, as the objective function may have multiple local minima.
4. **Do not reinvent the wheel**. Excellent minimization packages already exist in the Julia ecosystem. This is why `MSM.jl` relies on [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) and [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to perform the minimization.
