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
maxNumberWorkers = 4
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


using Test
using Plots
using Statistics
using SurrogateModelOptim
using ParallelDataTransfer

@everywhere using MSM
@everywhere using DataStructures
@everywhere using OrderedCollections
@everywhere using Distributions
@everywhere using Random
@everywhere using DataStructures
@everywhere using Statistics
@everywhere using LinearAlgebra

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
optionsSMM = MSMOptions(maxFuncEvals=100, globalOptimizer = :dxnes, localOptimizer = :NelderMead)
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
dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y); mean(x[:,1] .* y)] #informative on betas
dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y); mean(x[:,2] .* y)] #informative on betas
dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2); mean((x[:,1] .* y).^2)] #informative on betas
dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2); mean((x[:,2] .* y).^2)] #informative on betas
sendto(workers(), dictEmpiricalMoments=dictEmpiricalMoments)

W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
#Special case: diagonal matrix
#(you may choose something else)
for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
    W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
end


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

val_local = myProblem.objective_function([dictPriors[k][1] for k in keys(dictPriors)]); #local execution
val_workers = [];
for w in workers() #Execution on workers
    push!(val_workers, @fetchfrom w myProblem.objective_function([dictPriors[k][1] for k in keys(dictPriors)]))
end
for (wIndex, w) in enumerate(workers())
    @test abs(val_local - val_workers[wIndex]) < 10e-10
end


search_range=generate_bbSearchRange(myProblem)

# For tuning, see: https://github.com/MrUrq/SurrogateModelOptim.jl
@time result = smoptimize(x -> myProblem.objective_function(x), search_range;
                    options=SurrogateModelOptim.Options(
                    iterations=50,
                    num_interpolants=1*nworkers(),
                    num_start_samples=5,
                    create_final_surrogate=true,));

show(result)

minimizer = best_candidate(result)
min_function = best_fitness(result)

println("Estimated value for alpha = $(minimizer[1])")
println("True value for alpha = $(alpha0[1]) \n")

println("Estimated value for beta1 = $(minimizer[2])")
println("True value for beta1 = $(beta0[1]) \n")

println("Estimated value for beta2 = $(minimizer[3])")
println("True value for beta2 = $(beta0[2]) \n")

lower_bound = create_lower_bound(myProblem)
upper_bound = create_upper_bound(myProblem)

gr()

p1 = plot(collect(lower_bound[2]:0.1:upper_bound[2]), collect(lower_bound[3]:0.1:upper_bound[3]), (x, y) -> median(result.sm_interpolant([alpha0; x; y])), linetype=:surface)
scatter!([minimizer[2]], [minimizer[3]], [min_function], label="min")

p2 = contour(collect(lower_bound[2]:0.1:upper_bound[2]), collect(lower_bound[3]:0.1:upper_bound[3]), (x, y) -> median(result.sm_interpolant([alpha0; x; y])))
scatter!([minimizer[2]], [minimizer[3]], label="min")


p3 = plot(collect(lower_bound[1]:0.1:upper_bound[1]), collect(lower_bound[2]:0.1:upper_bound[2]), (x, y) -> median(result.sm_interpolant([x; y; beta0[2]])), linetype=:surface)
scatter!([minimizer[1]], [minimizer[2]], [min_function], label="min")

p4 = contour(collect(lower_bound[1]:0.1:upper_bound[1]), collect(lower_bound[2]:0.1:upper_bound[2]), (x, y) -> median(result.sm_interpolant([x; y; beta0[2]])))
scatter!([minimizer[1]], [minimizer[2]], label="min")

p5 = plot(p1, p2, p3, p4, title="Surrogate Model")

versioninfo()
