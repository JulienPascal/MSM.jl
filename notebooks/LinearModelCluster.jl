using ClusterManagers
using Distributed

OnCluster = false #set to false to run locally
addWorkers = true #set to false to run serially
println("OnCluster = $(OnCluster)")

# Current number of workers
#--------------------------
currentWorkers = nworkers()
println("Initial number of workers = $(currentWorkers)")

# Increase the number of workers available
#-----------------------------------------
maxNumberWorkers = 3
if addWorkers == true
	if OnCluster == true
	  addprocs(SlurmManager(maxNumberWorkers))
	else
	  addprocs(maxNumberWorkers)
	end
end


# Sanity checks
#-------------
hosts = []
pids = []
for i in workers()
	host, pid = fetch(@spawnat i (gethostname(), getpid()))
	println("Hello I am worker $(i), my host is $(host)")
	push!(hosts, host)
	push!(pids, pid)
end

currentWorkers = nworkers()
println("Number of workers = $(currentWorkers)")

using Plots
using ParallelDataTransfer

@everywhere using MSM
@everywhere using DataStructures
@everywhere using OrderedCollections
@everywhere using Distributions
@everywhere using Random
@everywhere using DataStructures
@everywhere using Statistics
@everywhere using LinearAlgebra

# Generate simulated data
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

# Send simulated data to workers
sendto(workers(), y=y)
sendto(workers(), x=x)

# Visualize data
p1 = scatter(x[1:100,1], y[1:100], xlabel = "x1", ylabel = "y", legend=:none, smooth=true)
p2 = scatter(x[1:100,2], y[1:100], xlabel = "x2", ylabel = "y", legend=:none, smooth=true)
plot(p1, p2)


# Define locally
optionsSMM = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :NelderMead)
myProblem = MSMProblem(options = optionsSMM);

# Send to workers
sendto(workers(), optionsSMM=optionsSMM)
sendto(workers(), myProblem=myProblem)

# Priors
dictPriors = OrderedDict{String,Array{Float64,1}}()
dictPriors["alpha"] = [0.5, 0.001, 1.0]
dictPriors["beta1"] = [0.5, 0.001, 1.0]
dictPriors["beta2"] = [0.5, 0.001, 1.0]
sendto(workers(), dictPriors=dictPriors)

# Empirical moments
dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
dictEmpiricalMoments["mean"] = [mean(y)] #informative on the intercept
dictEmpiricalMoments["mean^2"] = [mean(y.^2)] #informative on the intercept
dictEmpiricalMoments["mean^3"] = [mean(y.^3)] #informative on the intercept
dictEmpiricalMoments["var"] = [mean(y.^2) - mean(y)^2]
dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2)] #informative on betas

W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
#Special case: diagonal matrix
#(you may choose something else)
for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
    W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
end

# Send to workers
sendto(workers(), dictPriors=dictPriors)
sendto(workers(), dictEmpiricalMoments=dictEmpiricalMoments)
sendto(workers(), W=W)

@everywhere set_priors!(myProblem, dictPriors)
@everywhere set_empirical_moments!(myProblem, dictEmpiricalMoments)
@everywhere set_weight_matrix!(myProblem, W)

# x[1] corresponds to the intercept, x[2] corresponds to beta1, x[3] corresponds to beta2
@everywhere function functionLinearModel(x; uniform_draws::Array{Float64,1}, simX::Array{Float64,2}, nbDraws::Int64 = length(uniform_draws), burnInPerc::Int64 = 10)
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

# Send to workers
sendto(workers(), simX=simX)
sendto(workers(), uniform_draws=uniform_draws)

# Construct the objective function everywhere
@everywhere set_simulate_empirical_moments!(myProblem, x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX))
@everywhere construct_objective_function!(myProblem)

# Safety check: value on the master node == values on slave nodes?
using Test
val_local = myProblem.objective_function(ones(3)); #local execution
val_workers = [];
for w in workers() #Execution on workers
    push!(val_workers, @fetchfrom w myProblem.objective_function(ones(3)))
end
for (wIndex, w) in enumerate(workers())
    @test abs(val_local - val_workers[wIndex]) < 10e-10
end

println("Global Algorithm")
# Choose a global optimizer that supports parallel evaluations (e.g. xnes or dxnes)
# (see the documentation: https://github.com/robertfeldt/BlackBoxOptim.jl)
msm_optimize!(myProblem, verbose = false)

# Access the results using best_candidate
minimizer = msm_minimizer(myProblem)
minimum_val = msm_minimum(myProblem)

println("Minimum objective function = $(minimum_val)")
println("Estimated value for alpha = $(minimizer[1]). True value for beta1 = $(alpha0[1]) \n")
println("Estimated value for beta1 = $(minimizer[2]). True value for beta1 = $(beta0[1]) \n")
println("Estimated value for beta2 = $(minimizer[3]). True value for beta2 = $(beta0[2]) \n")


println("Multistart Algorithm")
# Start several local optimization algorithms in parallel
# Choose algorithms from the package Optim.jl (https://github.com/JuliaNLSolvers/Optim.jl)
# The "global" mininimum is the minimum of the local minima.
msm_multistart!(myProblem, nums = nworkers(), verbose = false)

minimizer_multistart = msm_multistart_minimizer(myProblem)
minimum_multistart = msm_multistart_minimum(myProblem)

println("Minimum objective function = $(minimum_multistart)")
println("Estimated value for alpha = $(minimizer_multistart[1]). True value for beta1 = $(alpha0[1]) \n")
println("Estimated value for beta1 = $(minimizer_multistart[2]). True value for beta1 = $(beta0[1]) \n")
println("Estimated value for beta2 = $(minimizer_multistart[3]). True value for beta2 = $(beta0[2]) \n")

# Empirical Series
#-----------------
X = zeros(T, 8)

X[:,1] = y
X[:,2] = y.^2
X[:,3] = y.^3
X[:,4] = (y .- mean(y)).^2
X[:,5] = (x[:,1] .* y)
X[:,6] = (x[:,2] .* y)
X[:,7] = (x[:,1] .* y).^2
X[:,8] = (x[:,2] .* y).^2

# "Distance Matrix" (see Duffie and Singleton, 1993)
Sigma0 = cov(X)

# Heatmap to visualize correlation
xs = [string("x", i) for i = 1:8]
ys = [string("x", i) for i = 1:8]
z = cor(X)
hh = heatmap(xs, ys, z, aspect_ratio = 1)

set_Sigma0!(myProblem, Sigma0)
# nbDraws = number of draws in the simulated data
# To decrease standard errors, increase nbDraws
calculate_Avar!(myProblem, minimizer, tau = T/nbDraws)

df = summary_table(myProblem, minimizer, T, 0.05)
println(df)

# Compare results with GLM
using DataFrames, GLM
data = DataFrame(x1=x[:,1], x2=x[:,2], y= y[:]);
ols = lm(@formula(y ~ x1 + x2), data)
println(ols)
