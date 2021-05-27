# Functions and Types

```@docs
MSMOptions
```

For example, let's say we want to estimate a model with a maximum of 1000 function evaluations
(only the global optimizer will take into consideration this constraint).
We want to use `:dxnes` as our global optimizer algorithm and `:LBFGS` for our local
minimization algorithm. Global optimizers can be chosen from [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl), while local optimizer
are to be chosen from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):
```julia
options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :LBFGS);
```
