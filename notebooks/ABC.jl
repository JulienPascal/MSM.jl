using MSM
using ApproxBayes
using DataStructures
using OrderedCollections
using Distributions
using Random
using DataStructures
using Statistics
using LinearAlgebra
using Plots
using DataFrames, GLM

n_threads = Threads.nthreads()
println("Number of threads = $(n_threads)")

Random.seed!(1234)  #for replicability reasons
T = 100000          #number of periods
P = 2               #number of dependent variables
beta0 = rand(P)     #choose true coefficients by drawing from a uniform distribution on [0,1]
alpha0 = rand(1)[]  #intercept
theta0 = 0.0        #coefficient to create serial correlation in the error terms
println("True intercept = $(alpha0)")
println("True coefficient beta0 = $(beta0)")
println("Serial correlation coefficient theta0 = $(theta0)")

# Generation of error terms
# row = individual dimension
# column = time dimension
U = zeros(T)
d = Normal()
U[1] = rand(d, 1)[] #first error term
# loop over time periods
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

optionsSMM = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :NelderMead)
myProblem = MSMProblem(options = optionsSMM);

# Priors
dictPriors = OrderedDict{String,Array{Float64,1}}()
dictPriors["alpha"] = [0.5, 0.001, 1.0]
dictPriors["beta1"] = [0.5, 0.001, 1.0]
dictPriors["beta2"] = [0.5, 0.001, 1.0]
set_priors!(myProblem, dictPriors)

# Empirical moments
dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
dictEmpiricalMoments["mean"] = [mean(y); mean(y)] #informative on the intercept
dictEmpiricalMoments["mean^2"] = [mean(y.^2); mean(y.^2)] #informative on the intercept
dictEmpiricalMoments["mean^3"] = [mean(y.^3); mean(y.^3)] #informative on the intercept
dictEmpiricalMoments["var"] = [mean(y.^2) - mean(y)^2; mean(y.^2) - mean(y)^2]
dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y); mean(x[:,1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y); mean(x[:,2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2); mean((x[:,1] .* y).^2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2); mean((x[:,2] .* y).^2)] #informative on betas
set_empirical_moments!(myProblem, dictEmpiricalMoments)


# x[1] corresponds to the intercept, x[2] corresponds to beta1, x[3] corresponds to beta2
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

    # loop over time periods
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
    output["mean^2"] = mean(y[startT:nbDraws].^2)
    output["mean^3"] = mean(y[startT:nbDraws].^3)
    output["var"] = mean(y[startT:nbDraws].^2) - mean(y[startT:nbDraws])^2
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

set_simulate_empirical_moments!(myProblem, x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX))
construct_objective_function!(myProblem)

targetdata = collect(values(dictEmpiricalMoments))

#Use dictPriors to create a prior for ApproxBayes
priors = ""
for (i, k) in enumerate(keys(myProblem.priors))
  if i==1
      global priors = string("Uniform($(dictPriors[k][2]), $(dictPriors[k][3]))" )
  else
      global priors = string(priors, ", ", "Uniform($(dictPriors[k][2]), $(dictPriors[k][3]))" )
  end
end

priors = eval(Meta.parse(string("Prior([", priors, "])")))

# Approximate Bayesian Computation Sequential Monte Carlo
setup = ABCSMC((params, constants, targetdata) -> (myProblem.objective_function(params), 1), #simulation function
  length(dictPriors), # number of parameters
  1e-4, #target ϵ
  priors; # Prior for each of the parameters
  maxiterations = 10^6, #Maximum number of iterations before the algorithm terminates
  )

# run ABC inference
@time res = runabc(setup, targetdata, verbose=true, progress=true, parallel=true)

#Write accepted draws to disk
#writeoutput(res)

println("True intercept = $(alpha0)")
println("True coefficient beta0 = $(beta0)")

p0=plot(res)
display(p0)

# Compare with OLS
data = DataFrame(x1=x[:,1], x2=x[:,2], y= y[:]);
ols = lm(@formula(y ~ x1 + x2), data)
println(ols)
