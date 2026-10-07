# Installation

You can install `MethodOfSimulatedMoments.jl` in two steps:

### Step 1
Enter the Pkg REPL by pressing `]` from the Julia REPL (to get back to the Julia REPL, press backspace or ^C. see [Pkg.jl](https://pkgdocs.julialang.org/v1/getting-started/))

### Step 2
To add a package, use `add`:
```julia
pkg> add https://github.com/JulienPascal/MethodOfSimulatedMoments.jl.git
```

Then load it with `using MethodOfSimulatedMoments`. Before version 0.2.0, the package was called `MSM.jl`: the functions and types keep their names (`MSMProblem`, `msm_optimize!`...), only `using MSM` becomes `using MethodOfSimulatedMoments`.
