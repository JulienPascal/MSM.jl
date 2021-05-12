# Functions

```@docs
MSMOptions
```

Estimate a model with a maximum of 1000 function evaluations (only relevant
for the global optimizer). Set the global optimizer to be `:dxnes` and the local
otimizer to be `:LBFGS`. Global optimizers can be chosen from [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) and local optimizers from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).
```@example
options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :LBFGS);
```
