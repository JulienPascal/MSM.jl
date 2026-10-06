"""
	MSMOptions

MSMOptions is a mutable struct that contains options related to the optimization.
# Examples
```julia-repl
julia> options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :LBFGS)
```
"""
mutable struct MSMOptions
	globalOptimizer::Symbol #algorithm for finding a global maximum
	localOptimizer::Symbol 	#algorithm for finding a local maximum
	maxFuncEvals::Int64			#maximum number of evaluations (global optimization, and each local minimization)
	saveName::String				#name under which the optimization should be saved
	showDistance::Bool			#show the distance, everytime the objective function is calculated?
	minBox::Bool						#When looking for a local maximum, use Fminbox ?
	populationSize::Int64		#When using BlackBoxOptim, set the population size (differential evolution; ignored by NES methods)
	lambda::Int64						#NES global optimizers (dxnes, xnes, separable_nes): points evaluated per generation (0 = automatic)
	penaltyValue::Float64 	#Objective function's value when the model fails (Inf by default)
	gridType::Symbol				#sampling procedure to use (latin hypercube by default)
	saveStartingValues::Bool #whether or not saving the starting values when using local_to_global
	maxTrialsStartingValues::Int64 #maximum number of attempts when searching for valid starting values
	thresholdStartingValue::Float64 #value under which a point is considered as a valid starting value
end

"""
	MSMOptions

Constructor for MSMOptions. MSMOptions is a mutable struct that contains options related to the optimization.

* `maxFuncEvals`: maximum number of evaluations of the objective function, for the global
  optimization and for each local minimization (finite-difference gradients included).
* `penaltyValue`: value of the objective function when the simulation fails (`Inf` by default).
* `thresholdStartingValue`: when searching for starting values, a point is valid if its
  objective function is below this value (default `penaltyValue/10`: with the default
  `penaltyValue = Inf`, any successful point is valid).
* `lambda`: for the natural evolution strategies (`:dxnes`, `:xnes`, `:separable_nes`), number
  of points evaluated per generation. The points of a generation are evaluated in parallel,
  and the next generation starts when all of them are done. With `lambda = 0` (default),
  BlackBoxOptim's default (which depends on the number of parameters) is rounded up to a
  multiple of the number of workers, so that no worker is idle (see `nes_lambda`).
  `:dxnes` requires an even `lambda`. Note that `maxFuncEvals` counts evaluations: a larger
  `lambda` means fewer generations for the same `maxFuncEvals`.
* `populationSize`: population size of the differential evolution optimizers (default 50).
  NES methods ignore it (see `lambda`). Differential evolution evaluates about one new
  point at a time, so it gets essentially no speed-up from several workers: prefer a NES
  method (e.g. `:dxnes`, the default) in parallel.

# Examples
```julia-repl
julia> options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :LBFGS)
```
"""
function MSMOptions( ;
					globalOptimizer::Symbol=:dxnes,
					localOptimizer::Symbol=:LBFGS,
					maxFuncEvals::Int64=1000,
					saveName::String = get_now(),
					showDistance::Bool = false,
					minBox::Bool = false,
					populationSize::Union{Nothing, Int64} = nothing,
					lambda::Int64 = 0,
					penaltyValue::Float64 = Inf,
					gridType::Symbol = :LHC,
					saveStartingValues::Bool = false,
					maxTrialsStartingValues::Int64 = 20,
					thresholdStartingValue::Float64 = penaltyValue/10.0)


	if thresholdStartingValue > penaltyValue
		error("Please set thresholdStartingValue < penaltyValue.")
	end

	#Fminbox does not work with all optimizers in Optim
	if minBox == true
		listValidLocalOptimizers = [:GradientDescent, :BFGS, :LBFGS, :ConjugateGradient]
		if in(localOptimizer, listValidLocalOptimizers) == false
			error("if minBox == true, localOptimizer must be in $(listValidLocalOptimizers)")
		end
	end

	#Check valid grid types
	listValidGridTypes = [:LHC, :Sobol]
	if in(gridType, listValidGridTypes) == false
		error("gridType must be in $(listValidGridTypes)")
	end

	#Number of points per generation of the NES optimizers
	if lambda < 0
		error("lambda must be >= 0 (0 = automatic).")
	end
	if globalOptimizer == :dxnes && isodd(lambda)
		error("globalOptimizer = :dxnes requires an even lambda (or lambda = 0, automatic).")
	end

	#populationSize is ignored by the NES optimizers
	if populationSize !== nothing && is_nes_optimizer(globalOptimizer) == true
		@warn "populationSize is ignored by globalOptimizer = :$(globalOptimizer). The number of points per generation is set by lambda."
	end
	if populationSize === nothing
		populationSize = 50
	end


	MSMOptions(globalOptimizer,
				localOptimizer,
				maxFuncEvals,
				saveName,
				showDistance,
				minBox,
				populationSize,
				lambda,
				penaltyValue,
				gridType,
				saveStartingValues,
				maxTrialsStartingValues,
				thresholdStartingValue)

end

"""
	MSMProblem

MSMProblem is a mutable struct that caries all the information needed to
perform the optimization and display the results.
"""
mutable struct MSMProblem
	priors::OrderedDict{String,Array{Float64,1}}
	W::Matrix{Float64} #Weight matrix in MSM objective function
	empiricalMoments::OrderedDict{String,Array{Float64,1}}
	simulatedMoments::OrderedDict{String, Float64}
	distanceEmpSimMoments::Float64
	simulate_empirical_moments::Function					#returns an ordered dict
	simulate_empirical_moments_array::Function		#returns an Array
	objective_function::Function
	options::MSMOptions
	bbSetup::Union{Nothing, BlackBoxOptim.OptController}					#set up when using BlackBoxOptim (global minimum). nothing until set
	bbResults::Union{Nothing, BlackBoxOptim.OptimizationResults}	#results when using BlackBoxOptim (global minimum). nothing until msm_optimize!
	optimResults::Union{Nothing, Optim.OptimizationResults}				#results when using Optim (local minimum). nothing until a local minimization
	Sigma0::Array{Float64,2}											#distance matrix (in the terminology of Duffie and Singleton (1993))
	Avar::Array{Float64,2}											  #asymptotic variance of the SMM estimate
end

# Constructor for MSMProblem
#------------------------------------------------------------------------------
function MSMProblem(  ; priors::OrderedDict{String,Array{Float64,1}} = OrderedDict{String,Array{Float64,1}}(),
						W::Matrix{Float64}=Matrix(1.0 .* I(1)),
						empiricalMoments::OrderedDict{String,Array{Float64,1}} = OrderedDict{String,Array{Float64,1}}(),
						simulatedMoments::OrderedDict{String, Float64} = OrderedDict{String,Float64}(),
						distanceEmpSimMoments::Float64 = 0.,
						simulate_empirical_moments::Function = default_function,       #returns an ordered dict
						simulate_empirical_moments_array::Function = default_function, #returns an Array
						objective_function::Function = default_function,
						options::MSMOptions = MSMOptions(),
						bbSetup::Union{Nothing, BlackBoxOptim.OptController} = nothing,
						bbResults::Union{Nothing, BlackBoxOptim.OptimizationResults} = nothing,
						optimResults::Union{Nothing, Optim.OptimizationResults} = nothing,
						Sigma0::Array{Float64,2} = Array{Float64}(undef,0,0),
						Avar::Array{Float64,2} = Array{Float64}(undef,0,0))

	MSMProblem(priors,
				W,
				empiricalMoments,
				simulatedMoments,
				distanceEmpSimMoments,
				simulate_empirical_moments,
				simulate_empirical_moments_array,
				objective_function,
				options,
				bbSetup,
				bbResults,
				optimResults,
				Sigma0,
				Avar)

end

"""
	default_function(x)

Function x->x. Used to initialize functions.
"""
function default_function(x)
	x
end


"""
	rosenbrock2d(x)

Rosenbrock function (a standard test function for optimizers). Its minimum is 0, at [1.0, 1.0].
"""
function rosenbrock2d(x)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

"""
	is_global_optimizer(s::Symbol)

function to check that the global optimizer chosen is available.
"""
function is_global_optimizer(s::Symbol)

	# Global Optimizers using BlackBoxOptim
	# source: https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/benchmarking/latest_toplist.csv
	#------------------------------------------------------------------------------
	listValidGlobalOptimizers = [:dxnes, :adaptive_de_rand_1_bin_radiuslimited, :xnes,
								 :de_rand_1_bin_radiuslimited, :adaptive_de_rand_1_bin,
								 :generating_set_search, :de_rand_1_bin,
								 :separable_nes, :resampling_inheritance_memetic_search,
								 :probabilistic_descent, :resampling_memetic_search,
								 :de_rand_2_bin_radiuslimited, :de_rand_2_bin,
								 :random_search, :simultaneous_perturbation_stochastic_approximation]

	in(s, listValidGlobalOptimizers)

end

"""
	is_bb_optimizer(s::Symbol)

function to check whether the global optimizer is using BlackBoxOptim
"""
function is_bb_optimizer(s::Symbol)

	# source: https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/benchmarking/latest_toplist.csv
	listbbOptimizers = [:dxnes, :adaptive_de_rand_1_bin_radiuslimited, :xnes,
					 :de_rand_1_bin_radiuslimited, :adaptive_de_rand_1_bin,
					 :generating_set_search, :de_rand_1_bin,
					 :separable_nes, :resampling_inheritance_memetic_search,
					 :probabilistic_descent, :resampling_memetic_search,
					 :de_rand_2_bin_radiuslimited, :de_rand_2_bin,
					 :random_search, :simultaneous_perturbation_stochastic_approximation]

	in(s, listbbOptimizers)

end

"""
	is_local_optimizer(s::Symbol)

function to check that the local optimizer chosen is available.
"""
function is_local_optimizer(s::Symbol)

	# source: https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/benchmarking/latest_toplist.csv
	listValidLocalOptimizers = [:NelderMead, :SimulatedAnnealing, :ParticleSwarm,
								:BFGS, :LBFGS, :ConjugateGradient, :GradientDescent,
								:MomentumGradientDescent, :AcceleratedGradientDescent]

	in(s, listValidLocalOptimizers)

end

"""
	is_nes_optimizer(s::Symbol)

function to check whether the global optimizer is a natural evolution strategy of
BlackBoxOptim (:dxnes, :xnes or :separable_nes), which evaluates lambda points per generation.
"""
function is_nes_optimizer(s::Symbol)

	in(s, [:dxnes, :xnes, :separable_nes])

end

"""
	is_optim_optimizer(s::Symbol)

function to check whether the local minimizer uses the package Optim.
"""
function is_optim_optimizer(s::Symbol)

	# source: https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/benchmarking/latest_toplist.csv
	listOptimOptimizers = [:NelderMead, :SimulatedAnnealing, :ParticleSwarm,
							:BFGS, :LBFGS, :ConjugateGradient, :GradientDescent,
							:MomentumGradientDescent, :AcceleratedGradientDescent]

	in(s, listOptimOptimizers)

end

"""
	convert_to_optim_algo(s::Symbol)

function to convert local optimizer (of type Symbol) to an Optim algo.
"""
function convert_to_optim_algo(s::Symbol)

	if is_optim_optimizer(s) == false
		error("$(s) is not a supported local optimizer.")
	end

	# e.g. :LBFGS -> Optim.LBFGS()
	getfield(Optim, s)()

end

"""
	convert_to_fminbox(s::Symbol)

function to convert local optimizer (of type Symbol) to a Fminbox usable
by Optim.
"""
function convert_to_fminbox(s::Symbol)

	Fminbox(convert_to_optim_algo(s))

end
