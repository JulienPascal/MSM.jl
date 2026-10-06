# Contributing to MSM.jl

Thank you for your interest in MSM.jl! Contributions are welcome: bug reports, corrections to the
documentation, examples, and improvements to the code.

MSM.jl is maintained in my spare time, so replies to issues and pull requests may take a while.
Thank you for your patience.

## Philosophy: keep MSM.jl small and focused

MSM.jl aims to remain a **minimalist, focused package**. The target audience are economists who want
to estimate structural economic models using the method of simulated moments. Before proposing a new feature, please check that
it fits its design principles:

1. **Parallelism at the optimizer level.** The user's simulation function (the expected response
   function) is assumed to be hard to parallelize, as simulated time series are often serially
   correlated. Instead, MSM.jl evaluates the objective function at several parameter values in
   parallel, on `Distributed` workers.
2. **A global search, then a local refinement.** The objective function may have several local
   minima, so MSM.jl first searches for a global minimum (with BlackBoxOptim.jl), then optionally
   refines it with a local optimizer (from Optim.jl).
3. **Do not reinvent the wheel.** MSM.jl relies on existing, well-maintained packages for
   optimization (BlackBoxOptim.jl, Optim.jl) and numerical derivatives (FiniteDifferences.jl).
   Contributions that reimplement what such a package already does are unlikely to be accepted;
   contributions that connect MSM.jl to such packages are welcome.
4. **Agnostic about how the economic model is solved and simulated.** MSM.jl only needs a
   function that maps parameter values to simulated moments. How the model behind it is solved
   and simulated (value function iteration, linearization, higher-order perturbation, projection
   methods...) is left to the user. The aim is **not** to recreate packages that solve economic
   models, such as Dynare.jl or MacroModelling.jl: these packages can instead be used to write the
   simulation function, as in [`notebooks/models/RBC.ipynb`](notebooks/models/RBC.ipynb)
   (MacroModelling.jl) and [`notebooks/models/dynare/RBCDynare.ipynb`](notebooks/models/dynare/RBCDynare.ipynb)
   (Dynare.jl).

New features are therefore considered conservatively. Small fixes can go directly to a pull
request; for anything larger, please open an issue first (see below).

## Reporting a bug

Please [open an issue](https://github.com/JulienPascal/MSM.jl/issues) with:

* your Julia version (the output of `versioninfo()`) and the version or commit of MSM.jl;
* a minimal example that reproduces the problem: ideally a few lines, with a simple simulation
  function;
* the full error message and stack trace, or what you expected and what you observed.

Tip: when an estimation runs but gives strange results, run `check_problem(myProblem)`. When the
simulation function throws an error, MSM.jl's objective function returns a penalty value instead
of stopping, so a mistake (e.g. a misspelled moment name) may otherwise go unnoticed.

## Proposing a feature

Open an issue first, describing the problem the feature solves and how it fits the philosophy
above. Agreeing on the approach before writing code saves time on both sides.

## Setting up a development environment

1. Fork the repository on GitHub (the `Fork` button in the top-right corner), and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MSM.jl.git
   cd MSM.jl
   ```
2. Create a branch from `main` for your change:
   ```bash
   git checkout -b my-change main
   ```
3. Install the dependencies, and run the tests:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   julia --project -e 'using Pkg; Pkg.test()'
   ```
   The tests start their own worker processes and take a few minutes.
4. If you change the documentation or the docstrings, build the documentation (it runs the
   examples of the documentation; the result is in `docs/build/`):
   ```bash
   julia --project=docs -e 'using Pkg; Pkg.instantiate()'
   julia --project=docs docs/make.jl
   ```

The notebooks in `notebooks/` are examples, not part of the tests. The notebooks in
`notebooks/models/` have their own environment (`notebooks/models/Project.toml`, and
`notebooks/models/dynare/Project.toml` for the Dynare.jl notebook).

## What a pull request should include

Pull requests target the `main` branch. Please keep each pull request focused on one problem, and
include:

* **tests** for the change, in `test/runtests.jl`. All the tests must pass;
* **docstrings** for new public functions, starting with a signature line, and the function
  exported in `src/MSM.jl`. The docstrings of the exported functions are collected
  automatically on the "Functions and Types" page of the documentation;
* an **entry in `CHANGELOG.md`**, under "Unreleased" (the file follows
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/));

Two more points:

* The order of the keys of the priors and of the empirical moments (`OrderedDict`s) defines the
  order of the parameters and of the moments in every vector and matrix (Jacobian, weight
  matrix...). Please preserve it.
* Please avoid adding dependencies. If one is necessary, add a `[compat]` entry for it in
  `Project.toml`.

Continuous integration runs the tests on Julia 1.10 and on the latest release, and builds the
documentation. Please make sure both pass.

## AI-assisted contributions

Contributions written with the help of AI tools are welcome. You remain responsible for them:
please read, check and test every change before submitting it, and say in the pull request that
AI tools were used. Pull requests whose code the contributor cannot explain may be declined.

## Code of conduct

Please follow the [Julia Community Standards](https://julialang.org/community/standards/).

## License

MSM.jl is released under the [MIT License](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms.
