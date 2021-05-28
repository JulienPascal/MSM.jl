# Getting Started

Our overarching goal is to find the parameter values $\theta_{MSM}$ minimizing
the following function:

$$g(\theta; m*, W) = (m(\theta) - m*)' W (m(\theta) - m*)$$

where $m*$ is a vector of empirical moments, $m(\theta)$ is a vector of moments
calculated using simulated data, and $W$ is carefully chosen weighting matrix. We also
want to build confidence intervals for $\theta_{MSM}$.

While simple in theory (it is just a function minimization, right?), in practice
many bad things can happen. The function $g$ may fail in some areas of the parameter
space; $g$ may be stuck in some local minima; $g$ is really slow and you do not
have a strong prior regarding good starting values. [MSM.jl](https://github.com/JulienPascal/MSM.jl) uses minimization algorithms that are robust to the problems mentioned above. You may choose between two options:
1. Global minimization algorithms from [BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl)
2. A multistart algorithm using several local optimization routines from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)


Let's follow a learning-by-doing approach. As a warm-up, let's first estimate
parameters in serial. In a second step, we use several workers on a cluster.

## Example in serial

In a real-world scenario, you would probably use empirical data. Here, let's
simulate a fake dataset.

```@example
using MSM
using DataStructures
using OrderedCollections
using Random
using Distributions
using Statistics
using LinearAlgebra
using Plots
Random.seed!(1234)  #for replicability reasons
T = 100000          #number of periods
P = 2               #number of dependent variables
beta0 = rand(P)     #choose true coefficients by drawing from a uniform distribution on [0,1]
alpha0 = rand(1)[]  #intercept
theta0 = 0.0 #coefficient to create serial correlation in the error terms

# Generation of error terms
# row = individual dimension
# column = time dimension
U = zeros(T)
d = Normal()
U[1] = rand(d, 1)[] #first error term
for t = 2:T
    U[t] = rand(d, 1)[] + theta0*U[t-1]
end

# Let's simulate the dependent variables x_t
x = zeros(T, P)
d = Uniform(0, 5)
for p = 1:P  
    x[:,p] = rand(d, T)
end

# Let's calculate the resulting y_t
y = zeros(T)
for t=1:T
    y[t] = alpha0 + x[t,1]*beta0[1] + x[t,2]*beta0[2] + U[t]
end

# Visualize data
p1 = scatter(x[1:100,1], y[1:100], xlabel = "x1", ylabel = "y", legend=:none, smooth=true)
p2 = scatter(x[1:100,2], y[1:100], xlabel = "x2", ylabel = "y", legend=:none, smooth=true)
p = plot(p1, p2);
savefig(p, "f-fake-data.svg"); nothing # hide
```

![](f-fake-data.svg)

### Step 1: Initializing a MSMProblem

```@example
# Select a global optimizer (see BlackBoxOptim.jl) and a local minimizer (see Optim.jl):
myProblem = MSMProblem(options = MSMOptions(maxFuncEvals=10000, globalOptimizer = :dxnes, localOptimizer = :LBFGS));
```

### Step 2. Set empirical moments and weight matrix

Choose the set of empirical moments to match and the weight matrix $W$ using the functions `set_empirical_moments!` and `set_weight_matrix!`

```@example
dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
dictEmpiricalMoments["mean"] = [mean(y)] #informative on the intercept
dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2)] #informative on betas

W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
#Special case: diagonal matrix
#Sum of square percentage deviations from empirical moments
#(you may choose something else)
for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
    W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
end

set_empirical_moments!(myProblem, dictEmpiricalMoments)
set_weight_matrix!(myProblem, W)
```

### Step 3. Set priors

Our "prior" belief regarding the parameter values is to be specified using `set_priors!()`.
It is not fully a full-fledged prior probability distribution, but simply an
initial guess for each parameter, as well as lower and upper bounds:

```@example
dictPriors = OrderedDict{String,Array{Float64,1}}()
# Of the form: [initial_guess, lower_bound, upper_bound]
dictPriors["alpha"] = [0.5, 0.001, 1.0]
dictPriors["beta1"] = [0.5, 0.001, 1.0]
dictPriors["beta2"] = [0.5, 0.001, 1.0]
set_priors!(myProblem, dictPriors)
```

### Step 4: Specifying the function generating simulated moments

The objective function must generate an **ordered dictionary** containing the **keys of dictEmpiricalMoments**. Use `set_simulate_empirical_moments!` and `construct_objective_function!`

**Remark:** we "freeze" randomness during the minimization step. One way to do
that is to generate draws from a Uniform([0,1]) outside of the objective function and to use [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to generate draws from a normal distribution. Otherwise the objective function would be "noisy" and the minimization algorithms would have a hard time finding
the global minimum.

```@example
# x[1] corresponds to the intercept; x[2] corresponds to beta1; x[3] corresponds to beta2
function functionLinearModel(x; uniform_draws::Array{Float64,1}, simX::Array{Float64,2}, nbDraws::Int64 = length(uniform_draws), burnInPerc::Int64 = 10)
    T = nbDraws
    P = 2       #number of dependent variables
    alpha = x[1]
    beta = x[2:end]
    theta = 0.0     #coefficient to create serial correlation in the error terms

    # Creation of error terms
    # row = individual dimension
    # column = time dimension
    U = zeros(T)
    d = Normal()
    # Inverse cdf (i.e. quantile)
    gaussian_draws = quantile.(d, uniform_draws)
    U[1] = gaussian_draws[1] #first error term
    for t = 2:T
        U[t] = gaussian_draws[t] + theta*U[t-1]
    end

    # Let's calculate the resulting y_t
    y = zeros(T)
    for t=1:T
        y[t] = alpha + simX[t,1]*beta[1] + simX[t,2]*beta[2] + U[t]
    end

    # Get rid of the burn-in phase:
    #------------------------------
    startT = div(nbDraws, burnInPerc)

    # Moments:
    #---------
    output = OrderedDict{String,Float64}()
    output["mean"] = mean(y[startT:nbDraws])
    output["mean_x1y"] = mean(simX[startT:nbDraws,1] .* y[startT:nbDraws])
    output["mean_x2y"] = mean(simX[startT:nbDraws,2] .* y[startT:nbDraws])
    output["mean_x1y^2"] = mean((simX[startT:nbDraws,1] .* y[startT:nbDraws]).^2)
    output["mean_x2y^2"] = mean((simX[startT:nbDraws,2] .* y[startT:nbDraws]).^2)

    return output
end

# Let's freeze the randomness during the minimization
d_Uni = Uniform(0,1)
nbDraws = 100000 #number of draws in the simulated data
uniform_draws = rand(d_Uni, nbDraws)
simX = zeros(length(uniform_draws), 2)
d = Uniform(0, 5)
for p = 1:2
  simX[:,p] = rand(d, length(uniform_draws))
end

# Attach the function parameters -> simulated moments:
set_simulate_empirical_moments!(myProblem, x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX))

# Construct the objective (m-m*)'W(m-m*):
construct_objective_function!(myProblem)
```

### Step 5. Running the optimization
Use the global optimization algorithm specified in `globalOptimizer`:

```@example
# Global optimization:
msm_optimize!(myProblem, verbose = false)
```

### Step 6. Analysing Results

####  Step 6.A. Point estimates

```@example
minimizer = msm_minimizer(myProblem)
minimum_val = msm_minimum(myProblem)
println("Minimum objective function = $(minimum_val)")
println("Estimated value for alpha = $(minimizer[1]). True value for beta1 = $(alpha0[1])")
println("Estimated value for beta1 = $(minimizer[2]). True value for beta1 = $(beta0[1])")
println("Estimated value for beta2 = $(minimizer[3]). True value for beta2 = $(beta0[2])")
```
```julia
Minimum objective function = 3.364713342376503e-6
Estimated value for alpha = 0.5725856664135125. True value for beta1 = 0.5662374165061859
Estimated value for beta1 = 0.5832878335694766. True value for beta1 = 0.5908446386657102
Estimated value for beta2 = 0.7664889629032697. True value for beta2 = 0.7667970365022592
```

####  Step 6.B. Inference

##### Estimation of the distance matrix $\Sigma_0$

Let's calculate the variance-covariance matrix of the **"distance matrix"** (using the terminolgy of [Duffie and Singleton (1993)](https://www.jstor.org/stable/2951768?seq=1)). Here we know that errors are not correlated (the serial correlation coefficient is set to 0 in the code above). in the presence of serial correlation, an HAC estimation would be needed.

```@example
# Empirical Series
#-----------------
X = zeros(T, 5)
X[:,1] = y
X[:,2] = (x[:,1] .* y)
X[:,3] = (x[:,2] .* y)
X[:,4] = (x[:,1] .* y).^2
X[:,5] = (x[:,2] .* y).^2
Sigma0 = cov(X)
```

##### Asymptotic variance

###### Theory

The asymptotic variance of the MSM estimate is calculated using the usual **GMM sandwich formula**, corrected to take into account simulation noise.

$$AsymptoticVarianceMSM = (1 + \tau)*AsymptoticVarianceGMM$$

Here we are trying to match unconditional moments from time series. In this case, $\tau = \frac{tData}{tSimulation}$, where $tData$ is the number of periods in the empirical data and $tSimulation$ is the number of time periods in the simulated data.

See [Duffie and Singleton (1993)](https://www.jstor.org/stable/2951768?seq=1) and [Gouriéroux and Montfort (1996)](https://www.jstor.org/stable/3533164?seq=1) for details on how to choose $\tau$.

###### Practice

Calculating the asymptotic variance using MSM.jl is done in two steps:
* setting the value of the **"distance matrix"** using the function `set_Sigma0!`
* calculating the asymptotic variance using the function `calculate_Avar!`

```@example
set_Sigma0!(myProblem, Sigma0)
calculate_Avar!(myProblem, minimizer, tau = T/nbDraws) # nbDraws = number of draws in the simulated data
```

#### Step 6.C. Summarizing the results

Once the asymptotic variance has been calculated, a summary table can be obtained using the
function `summary_table`. This function has four inputs:
1. a MSMProblem
2. the minimizer of the objective function
3. the length of the empirical sample
4. the confidence level associated to the test **H0:** $\theta_i = 0$,  **H1:** $\theta_i != 0$

```@example
df = summary_table(myProblem, minimizer, T, 0.05)
```

| Estimate | StdError   | tValue  | pValue  | ConfIntervalLower | ConfIntervalUpper |
|----------|------------|---------|---------|-------------------|-------------------|
| Float64  | Float64    | Float64 | Float64 | Float64           | Float64           |
| 0.572586 | 0.0228065  | 25.1062 | 0.0     | 0.535072          | 0.610099          |
| 0.583288 | 0.00868862 | 67.1324 | 0.0     | 0.568996          | 0.597579          |
| 0.766489 | 0.0081483  | 94.0674 | 0.0     | 0.753086          | 0.779892          |


## Example in parallel

To use the package on a cluster, one must make sure that empirical moments, priors
and the weight matrix are defined for each worker. This can be done using `@everywhere begin end` blocks, or by using [ParallelDataTransfer.jl](https://github.com/ChrisRackauckas/ParallelDataTransfer.jl). The function returning simulated moments must also be
defined `@everywhere`. See the file [LinearModelCluster.jl](https://github.com/JulienPascal/MSM.jl/blob/main/notebooks/LinearModelCluster.jl) for details.


### Option 1: Global parallel optimization

Choose a global optimizer that **supports parallel evaluations** (e.g. xnes or dxnes). See the [documentation](https://github.com/robertfeldt/BlackBoxOptim.jl) for BlackBoxOptim.jl.

```@example
msm_optimize!(myProblem, verbose = false)

# Access the results using best_candidate
minimizer = msm_minimizer(myProblem)
minimum_val = msm_minimum(myProblem)
```

### Option 2: Multistart algorithm

The function `msm_multistart!` searches for starting values for which the model converges. Then, several local optimization algorithms (specified with `localOptimizer`) are started in parallel. The "global" minimum is the minimum of the local minima:

```@example
msm_multistart!(myProblem, nums = nworkers(), verbose = false)
minimizer_multistart = msm_multistart_minimizer(myProblem)
minimum_multistart = msm_multistart_minimum(myProblem)
```
