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
	maxFuncEvals::Int64			#maximum number of evaluations
	saveName::String				#name under which the optimization should be saved
	showDistance::Bool			#show the distance, everytime the objective function is calculated?
	minBox::Bool						#When looking for a local maximum, use Fminbox ?
	populationSize::Int64		#When using BlackBoxOptim, set the population size
	penaltyValue::Float64 	#Objective function's value when the model fails
	gridType::Symbol				#sampling procedure to use (latin hypercube by default)
	saveStartingValues::Bool #whether or not saving the starting values when using local_to_global
	maxTrialsStartingValues::Int64 #maximum number of attempts when searching for valid starting values
	thresholdStartingValue::Float64 #value under which a point is considered as a valid starting value
end

"""
	MSMOptions

Constructor for MSMOptions. MSMOptions is a mutable struct that contains options related to the optimization.

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
					populationSize::Int64 = 50,
					penaltyValue::Float64 = 999999.0,
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
	listValidGridTypes = [:LHC]
	if in(gridType, listValidGridTypes) == false
		error("gridType must be in $(listValidGridTypes)")
	end


	MSMOptions(globalOptimizer,
				localOptimizer,
				maxFuncEvals,
				saveName,
				showDistance,
				minBox,
				populationSize,
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
	iter::Int64
	priors::OrderedDict{String,Array{Float64,1}}
	W::Matrix{Float64} #Weight matrix in MSM objective function
	empiricalMoments::OrderedDict{String,Array{Float64,1}}
	simulatedMoments::OrderedDict{String, Float64}
	distanceEmpSimMoments::Float64
	simulate_empirical_moments::Function					#returns an ordered dict
	simulate_empirical_moments_array::Function		#returns an Array
	objective_function::Function
	options::MSMOptions
	bbSetup::BlackBoxOptim.OptController					#set up when using BlackBoxOptim (global minimum)
	bbResults::BlackBoxOptim.OptimizationResults	#results when using BlackBoxOptim (global minimum)
	optimResults::Optim.OptimizationResults				#results when using Optim (local minimum)
	Sigma0::Array{Float64,2}											#distance matrix (in the terminology of Duffie and Singleton (1993))
	Avar::Array{Float64,2}											  #asymptotic variance of the SMM estimate
end

# Constructor for MSMProblem
#------------------------------------------------------------------------------
function MSMProblem(  ;iter::Int64 = 0,
						priors::OrderedDict{String,Array{Float64,1}} = OrderedDict{String,Array{Float64,1}}(),
						W::Matrix{Float64}=Matrix(1.0 .* I(1)),
						empiricalMoments::OrderedDict{String,Array{Float64,1}} = OrderedDict{String,Array{Float64,1}}(),
						simulatedMoments::OrderedDict{String, Float64} = OrderedDict{String,Float64}(),
						distanceEmpSimMoments::Float64 = 0.,
						simulate_empirical_moments::Function = default_function,       #returns an ordered dict
						simulate_empirical_moments_array::Function = default_function, #returns an Array
						objective_function::Function = default_function,
						options::MSMOptions = MSMOptions(),
						bbSetup::BlackBoxOptim.OptController = defaultbbOptimOptController,
						bbResults::BlackBoxOptim.OptimizationResults = defaultbbOptimOptimizationResults,
						optimResults::Optim.OptimizationResults = defaultOptimResults,
						Sigma0::Array{Float64,2} = Array{Float64}(undef,0,0),
						Avar::Array{Float64,2} = Array{Float64}(undef,0,0))

	MSMProblem(iter,
				priors,
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

Rosenbrock function. Used to initialize BlackBoxOptim.OptController().
"""
function rosenbrock2d(x)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# It is quite useful to have "default" BlackBoxOptim.OptController and
# BlackBoxOptim.OptimizationResults objects, since
# BlackBoxOptim.OptController() and BlackBoxOptim.OptimizationResults()
# do not work
#-------------------------------------------------------------------------------
defaultbbOptimOptController = bbsetup(x -> rosenbrock2d(x);
											Method=:dxnes,
											SearchRange = (-5.0, 5.0),
											NumDimensions = 2, MaxFuncEvals = 2,
											TraceMode = :silent);

defaultbbOptimOptimizationResults = bboptimize(defaultbbOptimOptController);

defaultOptimResults = optimize(rosenbrock2d, [0.0, 0.0], LBFGS());

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
	eval(Meta.parse("$(s)()"))
end

"""
	convert_to_fminbox(s::Symbol)

function to convert local optimizer (of type Symbol) to a Fminbox usable
by Optim.
"""
function convert_to_fminbox(s::Symbol)

	# Old API (before v0.15.0)
	# To be changed when switching to Julia v0.7
	eval(Meta.parse("Fminbox($(String(s))())"))

end
