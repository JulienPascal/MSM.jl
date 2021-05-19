# Using the Laplace Type Estimator formualation of MSM
# See: https://github.com/JulienPascal/MCMC_Approach_Classical_Estimation/tree/master
#-------------------------------------------------------------------------------
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
maxNumberWorkers = 10
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
using DataFrames
using RobustPmap #Necessary for AffineInvariantMCMC
using CSV
@everywhere using MSM
@everywhere using DataStructures
@everywhere using OrderedCollections
@everywhere using Distributions
@everywhere using Random
@everywhere using DataStructures
@everywhere using Statistics
@everywhere using LinearAlgebra
@everywhere using AffineInvariantMCMC
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
true_vals = [alpha0, beta0[1], beta0[2]]
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
dictEmpiricalMoments["mean"] = [mean(y); mean(y)] #informative on the intercept
dictEmpiricalMoments["mean^2"] = [mean(y.^2); mean(y.^2)] #informative on the intercept
dictEmpiricalMoments["mean^3"] = [mean(y.^3); mean(y.^3)] #informative on the intercept
dictEmpiricalMoments["var"] = [mean(y.^2) - mean(y)^2; mean(y.^2) - mean(y)^2]
dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y); mean(x[:,1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y); mean(x[:,2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2); mean((x[:,1] .* y).^2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2); mean((x[:,2] .* y).^2)] #informative on betas
sendto(workers(), dictEmpiricalMoments=dictEmpiricalMoments)

@everywhere set_priors!(myProblem, dictPriors)
@everywhere set_empirical_moments!(myProblem, dictEmpiricalMoments)

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
nbDraws = 10000 #Number of draws in the simulated data
burnInPerc = 10 #Burn-in phase (10%). Not necessary in the present context.
startT = div(nbDraws, burnInPerc) #First period used to calculate moments on simulated data
NMSM = nbDraws - startT + 1; #Number of Draws used when calculated moments on simulated data
uniform_draws = rand(d_Uni, nbDraws)
simX = zeros(length(uniform_draws), 2)
d = Uniform(0, 5)
for p = 1:2
  simX[:,p] = rand(d, length(uniform_draws))
end

# Send to workers
sendto(workers(), burnInPerc=burnInPerc)
sendto(workers(), simX=simX)
sendto(workers(), uniform_draws=uniform_draws)

# Construct the objective function everywhere
@everywhere set_simulate_empirical_moments!(myProblem, x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX, burnInPerc=burnInPerc))
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

#------------------------------------------------------------------------------
# Formulate the problem as a Laplace Type Estimator and use MCMC to find
# the quasi-posterior median
#------------------------------------------------------------------------------
# For tuning parameters, see: See https://github.com/madsjulia/AffineInvariantMCMC.jl
@everywhere begin
	numdims = 3
	numwalkers = 10
	thinning = 10
	numsamples_perwalker = 10000
	burnin = Int((1/10)*numsamples_perwalker)
	lb = 0 .* ones(numdims) #lower bound
	ub = 1 .* ones(numdims) #upper bound
	# Uniform prior
	# d_prior = Product(Uniform.(lb, ub))
	# Normal
	d_prior = MvNormal(zeros(numdims), 0.1 .* I(numdims))
end


# Pseudo Log-likelihood
@everywhere function Ln_MSM(x, NMSM)
	return -0.5*NMSM*myProblem.objective_function(x)
end

# Pseudo Log quasi-posterior: Pseudo Log(likelihood) + log(prior)
@everywhere function quasi_posterior(x, NMSM, d_prior)
	 return Ln_MSM(x, NMSM) + log(pdf(d_prior, x))
end


# Safety check: value on the master node == values on slave nodes?
using Test
val_local = Ln_MSM(ones(3), NMSM); #local execution
val_workers = [];
for w in workers() #Execution on workers
    push!(val_workers, @fetchfrom w Ln_MSM(ones(3), NMSM))
end
for (wIndex, w) in enumerate(workers())
    @test abs(val_local - val_workers[wIndex]) < 10e-10
end


# Slightly perturb the initial draws for the walkers
x0 = [dictPriors[k][1] for k in keys(dictPriors)]
x0_chains = ones(numdims, numwalkers).*true_vals .+ rand(numdims, numwalkers) .* 1.0
chain, llhoodvals = AffineInvariantMCMC.sample(x -> quasi_posterior(x, nbDraws, d_prior), numwalkers, x0_chains, burnin, 1)
chain, llhoodvals = AffineInvariantMCMC.sample(x -> quasi_posterior(x, nbDraws, d_prior), numwalkers, chain[:, :, end], numsamples_perwalker, thinning)
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain, llhoodvals)

#-------------------------------------------------------------------------------
# Plot Draws
#-------------------------------------------------------------------------------
p1 = plot(flatchain[1,:], ylabel="alpha0", xlabel="T", legend=:none)
p2 = plot(flatchain[2,:], ylabel="beta1", xlabel="T", legend=:none)
p3 = plot(flatchain[3,:], ylabel="beta2", xlabel="T", legend=:none)
p4 = plot(p1, p2, p3)
savefig(p4, joinpath(pwd(),"chains_MSM_MCMC.png"))


hh1 = histogram(flatchain[1,burnin:end], title="alpha0", legend=:none)
vline!(hh1, [alpha0[1]], linewidth = 4)
hh2 = histogram(flatchain[2,burnin:end], title="beta1", legend=:none)
vline!(hh2, [beta0[1]], linewidth = 4)
hh3 = histogram(flatchain[3,burnin:end], title="beta2", legend=:none)
vline!(hh3, [beta0[2]], linewidth = 4)
hh4 = plot(hh1, hh2, hh3)
savefig(hh4, joinpath(pwd(),"histograms_MSM_MCMC.png"))

# Compare results with GLM
using DataFrames, GLM
data = DataFrame(x1=x[:,1], x2=x[:,2], y= y[:]);
ols = lm(@formula(y ~ x1 + x2), data)
coef_ols = coef(ols)
ci_ols = confint(ols)
stderror_ols = stderror(ols)

result_alpha0 = append!(quantile(flatchain[1,burnin:end],[0.05, 0.10, 0.5, 0.90, 0.95]), std(flatchain[1,burnin:end]), NaN, alpha0[1],NaN, coef_ols[1], ci_ols[1,1], ci_ols[1,2], stderror_ols[1])
result_beta1 = append!(quantile(flatchain[2,burnin:end],[0.05, 0.10, 0.5, 0.90, 0.95]), std(flatchain[2,burnin:end]), NaN,beta0[1], NaN,coef_ols[2], ci_ols[2,1], ci_ols[2,2], stderror_ols[2])
result_beta2 = append!(quantile(flatchain[3,burnin:end],[0.05, 0.10, 0.5, 0.90, 0.95]), std(flatchain[3,burnin:end]), NaN,beta0[2], NaN,coef_ols[3], ci_ols[3,1], ci_ols[3,2], stderror_ols[3])
results = DataFrame(variable = ["P5"; "P10"; "Median"; "P90"; "P95"; "std"; "-" ;"True value"; "-" ;"OLS Estimate"; "P5 OLS"; "P95 OLS"; "Std OLS"],
						alpha0 = result_alpha0, beta1 = result_beta1, beta2 = result_beta2)

CSV.write(joinpath(pwd(),"output_table_MSM_MCMC.csv"), results)
