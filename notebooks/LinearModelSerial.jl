using ClusterManagers
using Distributed

OnCluster = false #set to false to run locally
addWorkers = false #set to false to run serially
println("OnCluster = $(OnCluster)")

# Current number of workers
#--------------------------
currentWorkers = nworkers()
println("Initial number of workers = $(currentWorkers)")

# Increase the number of workers available
#-----------------------------------------
maxNumberWorkers = 1
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
using LaTeXStrings
using ParallelDataTransfer

using MSM
using DataStructures
using OrderedCollections
using Distributions
using Random
using DataStructures
using Statistics
using LinearAlgebra

# Generate simulated data
Random.seed!(1234)  #for replicability reasons
T = 10000         #number of periods
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
    U[t] = rand(d, 1)[] + theta0 * U[t-1]
end

# Let's simulate the dependent variables x_t
x = zeros(T, P)

d = Uniform(0, 5)
for p = 1:P
    x[:, p] = rand(d, T)
end

# Let's calculate the resulting y_t
y = zeros(T)

for t = 1:T
    y[t] = alpha0 + x[t, 1] * beta0[1] + x[t, 2] * beta0[2] + U[t]
end


# Visualize data
p1 = scatter(
    x[1:100, 1],
    y[1:100],
    xlabel = "x1",
    ylabel = "y",
    legend = :none,
    smooth = true,
)
p2 = scatter(
    x[1:100, 2],
    y[1:100],
    xlabel = "x2",
    ylabel = "y",
    legend = :none,
    smooth = true,
)
plot(p1, p2)


# Define locally
optionsSMM = MSMOptions(
    maxFuncEvals = 2000,
    globalOptimizer = :dxnes,
    localOptimizer = :NelderMead,
)
myProblem = MSMProblem(options = optionsSMM);

# Priors
dictPriors = OrderedDict{String,Array{Float64,1}}()
dictPriors["alpha"] = [0.5, 0.001, 1.0]
dictPriors["beta1"] = [0.5, 0.001, 1.0]
dictPriors["beta2"] = [0.5, 0.001, 1.0]

# Empirical moments
dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
dictEmpiricalMoments["mean"] = [mean(y)] #informative on the intercept
dictEmpiricalMoments["mean_x1y"] = [mean(x[:, 1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:, 2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:, 1] .* y) .^ 2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:, 2] .* y) .^ 2)] #informative on betas

W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
#Special case: diagonal matrix
#(you may choose something else)
for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
    W[indexMoment, indexMoment] = 1.0 / (dictEmpiricalMoments[k][1])^2
end

set_priors!(myProblem, dictPriors)
set_empirical_moments!(myProblem, dictEmpiricalMoments)
set_weight_matrix!(myProblem, W)

# x[1] corresponds to the intercept, x[2] corresponds to beta1, x[3] corresponds to beta2
function functionLinearModel(
    x;
    uniform_draws::Array{Float64,1},
    simX::Array{Float64,2},
    nbDraws::Int64 = length(uniform_draws),
    burnInPerc::Int64 = 0,
)
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
        U[t] = gaussian_draws[t] + theta * U[t-1]
    end

    # Let's calculate the resulting y_t
    y = zeros(T)

    for t = 1:T
        y[t] = alpha + simX[t, 1] * beta[1] + simX[t, 2] * beta[2] + U[t]
    end

    # Get rid of the burn-in phase:
    #------------------------------
    startT = max(1, Int(nbDraws * (burnInPerc / 100)))

    # Moments:
    #---------
    output = OrderedDict{String,Float64}()
    output["mean"] = mean(y[startT:nbDraws])
    output["mean_x1y"] = mean(simX[startT:nbDraws, 1] .* y[startT:nbDraws])
    output["mean_x2y"] = mean(simX[startT:nbDraws, 2] .* y[startT:nbDraws])
    output["mean_x1y^2"] = mean((simX[startT:nbDraws, 1] .* y[startT:nbDraws]) .^ 2)
    output["mean_x2y^2"] = mean((simX[startT:nbDraws, 2] .* y[startT:nbDraws]) .^ 2)
    return output
end

# Let's freeze the randomness during the minimization
d_Uni = Uniform(0, 1)
nbDraws = T #number of draws in the simulated data
uniform_draws = rand(d_Uni, nbDraws)
simX = zeros(length(uniform_draws), 2)
d = Uniform(0, 5)
for p = 1:2
    simX[:, p] = rand(d, length(uniform_draws))
end

# Construct the objective function everywhere
set_simulate_empirical_moments!(
    myProblem,
    x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX),
)
construct_objective_function!(myProblem)

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
X = zeros(T, length(dictEmpiricalMoments))
X[:, 1] = y
X[:, 2] = (x[:, 1] .* y)
X[:, 3] = (x[:, 2] .* y)
X[:, 4] = (x[:, 1] .* y) .^ 2
X[:, 5] = (x[:, 2] .* y) .^ 2

# "Distance Matrix" (see Duffie and Singleton, 1993)
Sigma0 = cov(X)

# Heatmap to visualize correlation
xs = [string("x", i) for i = 1:length(dictEmpiricalMoments)]
ys = [string("x", i) for i = 1:length(dictEmpiricalMoments)]
z = cor(X)
hh = heatmap(xs, ys, z, aspect_ratio = 1)

set_Sigma0!(myProblem, Sigma0)
# nbDraws = number of draws in the simulated data
# To decrease standard errors, increase nbDraws
calculate_Avar!(myProblem, minimizer_multistart, tau = T / nbDraws)

df = summary_table(myProblem, minimizer_multistart, T, 0.05)
println(df)

# Check the rank condition
# Local identification requires D to be full column rank (in a neighborhood of the solution)
D = calculate_D(myProblem, minimizer_multistart)
println("number of parameters: $(size(D,2))")
println("rank of D is: $(rank(D))")

# Slices
vXGrid, vYGrid = msm_slices(myProblem, minimizer_multistart, nbPoints = 7);
p1 = plot(vXGrid[:, 1],vYGrid[:, 1],title = L"\alpha", label = "",linewidth = 3, xrotation = 45)
plot!(p1, [minimizer_multistart[1]], seriestype = :vline, label = "",linewidth = 1)
p2 = plot(vXGrid[:, 2],vYGrid[:, 2],title = L"\beta_1", label = "",linewidth = 3, xrotation = 45)
plot!(p2, [minimizer_multistart[2]], seriestype = :vline, label = "",linewidth = 1)
p3 = plot(vXGrid[:, 3],vYGrid[:, 3],title = L"\beta_2", label = "",linewidth = 3, xrotation = 45)
plot!(p3, [minimizer_multistart[3]], seriestype = :vline, label = "",linewidth = 1)
plot_combined = plot(p1, p2, p3)
savefig(plot_combined, "slices.png")
display(plot_combined)

# Efficient GMM
#---------------
myProblem.W = inv(Sigma0)
msm_multistart!(myProblem, nums = nworkers(), verbose = false)
minimizer_multistart = msm_multistart_minimizer(myProblem)
minimum_multistart = msm_multistart_minimum(myProblem)

# J-test
J, c = J_test(myProblem, minimizer_multistart, T, nbDraws, 0.05)
println("J-statistic: $(J)")
println("Critical value: $(c)")

# Compare results with GLM
using DataFrames, GLM
data = DataFrame(x1 = x[:, 1], x2 = x[:, 2], y = y[:]);
ols = lm(@formula(y ~ x1 + x2), data)
println(ols)
