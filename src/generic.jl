"""
  set_simulate_empirical_moments!(sMMProblem::MSMProblem, f::Function)

Function to set the field simulate_empirical_moments for a MSMProblem.
The function simulate_empirical_moments takes parameter values and return
the corresponding simulate moments values.
"""
function set_simulate_empirical_moments!(sMMProblem::MSMProblem, f::Function)

  # set the function that returns an ordered dictionary
  sMMProblem.simulate_empirical_moments = f

  # set the function that returns an array
  # this function is used to calculate the jacobian
  function simulate_empirical_moments_array(x)

    momentsODict = sMMProblem.simulate_empirical_moments(x)

    # Follow the order of the empirical moments (which defines the order of W),
    # not the order in which the user's function inserted the simulated moments
    if isempty(sMMProblem.empiricalMoments) == true
      momentKeys = collect(keys(momentsODict))
    else
      momentKeys = collect(keys(sMMProblem.empiricalMoments))
    end

    momentsArray = Array{Float64}(undef, length(momentKeys))

    for (i, k) in enumerate(momentKeys)
        momentsArray[i] = momentsODict[k]
    end

    return momentsArray

  end

  sMMProblem.simulate_empirical_moments_array = simulate_empirical_moments_array

end

"""
  construct_objective_function!(sMMProblem::MSMProblem)

Function that construct an objective function, using the function
MSMProblem.simulate_empirical_moments.
"""
function construct_objective_function!(sMMProblem::MSMProblem)


    function objective_function_MSM(x)

          # Initialization
          #----------------
          distanceEmpSimMoments = sMMProblem.options.penaltyValue

          #---------------------------------------------------------------------
          # A. Generate simulated moments
          #---------------------------------------------------------------------
          simulatedMoments, convergence = try

          sMMProblem.simulate_empirical_moments(x), 1

      catch errorSimulation

            info("An error occurred with parameter values = $(x)")
            info("$(errorSimulation)")

            OrderedDict{String,Array{Float64,1}}(), 0

      end

      #------------------------------------------------------------------------
      # B. If generating moment was successful, calculate distance between empirical
      # and simulated moments
      # (If no convergence, returns penalty value : sMMProblem.options.penaltyValue)
      #------------------------------------------------------------------------
      if convergence == 1

        # Errors here (e.g. a simulated moment is missing) also return the penalty value
        try

          # to store the distance between empirical and simulated moments
          arrayDistance = zeros(length(keys(sMMProblem.empiricalMoments)))

          for (indexMoment, k) in enumerate(keys(sMMProblem.empiricalMoments))

            # * sMMProblem.empiricalMoments[k][1] is the empirical moments
            #---------------------------------------------------------------------
            arrayDistance[indexMoment] = (sMMProblem.empiricalMoments[k][1] - simulatedMoments[k])

          end

          # formula is (m - m*)'*W*(m - m*)'
          distanceEmpSimMoments = transpose(arrayDistance)*sMMProblem.W*arrayDistance

        catch errorDistance

          info("An error occurred when calculating the distance with parameter values = $(x)")
          info("$(errorDistance)")

          distanceEmpSimMoments = sMMProblem.options.penaltyValue

        end

        # NaN or Inf simulated moments: return the penalty value
        if isfinite(distanceEmpSimMoments) == false

          info("Non-finite distance with parameter values = $(x)")

          distanceEmpSimMoments = sMMProblem.options.penaltyValue

        end

        if sMMProblem.options.showDistance == true
          println("distance = $(distanceEmpSimMoments)")
        end

      end

      return distanceEmpSimMoments

    end


    # Attach the objective function
    sMMProblem.objective_function = objective_function_MSM

end

"""
  check_problem(sMMProblem::MSMProblem)

Check that the MSMProblem is ready for the optimization, and throw an informative
error otherwise. The objective function returns the penalty value on any error, so
without this check a mistake (e.g. a weight matrix of the wrong size, a misspelled
moment name, or a bug in the simulation function) would give a flat objective function
and the optimization would run to completion.

Checks that the priors, the empirical moments, the simulation function and the objective
function are set, that `W` is k×k (k empirical moments), and simulates the moments once,
at the initial values of the priors: the simulation must not throw, must return every
empirical moment, and the simulated moments must be finite.

Called by `msm_optimize!` and `msm_multistart!`, unless `check = false`.
"""
function check_problem(sMMProblem::MSMProblem)

  skipMessage = "To skip this check, use check = false in msm_optimize! or msm_multistart!."

  # A. Fields that must be set
  #---------------------------
  if isempty(sMMProblem.priors) == true
    error("No priors. Please call set_priors! first.")
  end

  if isempty(sMMProblem.empiricalMoments) == true
    error("No empirical moments. Please call set_empirical_moments! first.")
  end

  nbMoments = length(sMMProblem.empiricalMoments)
  if size(sMMProblem.W) != (nbMoments, nbMoments)
    error("The weight matrix W is $(size(sMMProblem.W, 1))×$(size(sMMProblem.W, 2)), but there are $(nbMoments) empirical moments. Please call set_weight_matrix! with a $(nbMoments)×$(nbMoments) matrix.")
  end

  if sMMProblem.simulate_empirical_moments === default_function
    error("No simulation function. Please call set_simulate_empirical_moments! first.")
  end

  if sMMProblem.objective_function === default_function
    error("No objective function. Please call construct_objective_function! first.")
  end

  # B. One simulation at the initial values of the priors
  # (outside of the objective function, which would turn an error into the penalty value)
  #---------------------------------------------------------------------------------------
  x0 = [sMMProblem.priors[k][1] for k in keys(sMMProblem.priors)]

  simulatedMoments = try
    sMMProblem.simulate_empirical_moments(x0)
  catch simulationError
    @error "The simulation of moments failed at the initial values of the priors" x0 exception = (simulationError, catch_backtrace())
    error("The simulation of moments failed at the initial values of the priors, $(x0) (see the error above). Fix the simulation function, or change the initial values with set_priors!. $(skipMessage)")
  end

  missingMoments = [k for k in keys(sMMProblem.empiricalMoments) if haskey(simulatedMoments, k) == false]
  if isempty(missingMoments) == false
    error("The simulation function does not return the empirical moment(s) $(missingMoments). It returns $(collect(keys(simulatedMoments))). $(skipMessage)")
  end

  extraMoments = [k for k in keys(simulatedMoments) if haskey(sMMProblem.empiricalMoments, k) == false]
  if isempty(extraMoments) == false
    @warn "The simulated moment(s) $(extraMoments) are not empirical moments: they are ignored."
  end

  nonFiniteMoments = [k for k in keys(sMMProblem.empiricalMoments) if isfinite(simulatedMoments[k]) == false]
  if isempty(nonFiniteMoments) == false
    error("Non-finite simulated moment(s) $(nonFiniteMoments) at the initial values of the priors, $(x0). Change the initial values with set_priors!. $(skipMessage)")
  end

  # C. With a finite penalty value, successful points must not look worse than failures
  #-------------------------------------------------------------------------------------
  arrayDistance = [sMMProblem.empiricalMoments[k][1] - simulatedMoments[k] for k in keys(sMMProblem.empiricalMoments)]
  distance = transpose(arrayDistance)*sMMProblem.W*arrayDistance
  if isfinite(sMMProblem.options.penaltyValue) == true && distance >= sMMProblem.options.penaltyValue
    @warn "At the initial values of the priors, the distance between empirical and simulated moments ($(distance)) is larger than penaltyValue ($(sMMProblem.options.penaltyValue)): failed simulations look better than this point. Consider rescaling W, or using penaltyValue = Inf."
  end

  return nothing

end

"""
  set_priors!(sMMProblem::MSMProblem, priors::OrderedDict{String,Array{Float64,1}})

Function to change the field sMMProblem.priors
"""
function set_priors!(sMMProblem::MSMProblem, priors::OrderedDict{String,Array{Float64,1}})

  sMMProblem.priors = priors

end

"""
   set_weight_matrix!(sMMProblem::MSMProblem, W::Matrix{Float64})

Function to change the field sMMProblem.empiricalMoments
"""
function set_weight_matrix!(sMMProblem::MSMProblem, W::Matrix{Float64})

  sMMProblem.W = W

end

"""
   set_empirical_moments!(sMMProblem::MSMProblem, empiricalMoments::OrderedDict{String,Array{Float64,1}})

Function to change the field sMMProblem.empiricalMoments
"""
function set_empirical_moments!(sMMProblem::MSMProblem, empiricalMoments::OrderedDict{String,Array{Float64,1}})

  sMMProblem.empiricalMoments = empiricalMoments

end

"""
   set_Sigma0!(sMMProblem::MSMProblem, Sigma0::Array{Float64,2})

Function to change the field sMMProblem.Sigma0, where Sigma0 is the distance matrix,
in the terminology of Duffie and Singleton (1993)
"""
function  set_Sigma0!(sMMProblem::MSMProblem, Sigma0::Array{Float64,2})

  sMMProblem.Sigma0 = Sigma0

end

"""
  set_global_optimizer!(sMMProblem::MSMProblem; verbose::Bool = true)

Function to set the fields corresponding to the global
optimizer problem.
"""
function set_global_optimizer!(sMMProblem::MSMProblem; verbose::Bool = true)

  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true

    set_bbSetup!(sMMProblem, verbose = verbose)

  else

    Base.error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end

end

"""
  set_bbSetup!(sMMProblem::MSMProblem; verbose::Bool = true)

Function to set the field bbSetup for a MSMProblem.
With `verbose = false`, BlackBoxOptim does not display the progress of the optimization.
For the NES optimizers (`:dxnes`, `:xnes`, `:separable_nes`), the number of points per
generation is `nes_lambda(globalOptimizer, options.lambda, number of parameters, nworkers())`.
"""
function set_bbSetup!(sMMProblem::MSMProblem; verbose::Bool = true)

  # A. using sMMProblem.priors, generate searchRange:
  #-------------------------------------------------
  mySearchRange = generate_bbSearchRange(sMMProblem)
  nbDimensions = length(keys(sMMProblem.priors))
  globalOptimizer = sMMProblem.options.globalOptimizer

  traceMode = verbose ? :verbose : :silent

  if verbose == true
    info("$(nworkers()) worker(s) detected")
  end

  # B. NES optimizers: number of points per generation (evaluated in parallel)
  #---------------------------------------------------------------------------
  if is_nes_optimizer(globalOptimizer) == true

    lambda = nes_lambda(globalOptimizer, sMMProblem.options.lambda, nbDimensions, nworkers())
    extraOptions = (lambda = lambda,)

    if verbose == true
      message = "$(globalOptimizer): λ = $(lambda) points per generation ($(nworkers()) worker(s)), maxFuncEvals = $(sMMProblem.options.maxFuncEvals) → ~$(div(sMMProblem.options.maxFuncEvals, lambda)) generations"
      if nworkers() > 1 && mod(lambda, nworkers()) != 0
        message *= " (λ is not a multiple of the number of workers: some workers are idle at the end of each generation)"
      end
      info(message)
    end

  else

    extraOptions = NamedTuple()

    # Differential evolution evaluates about one new point at a time
    if verbose == true && nworkers() > 1 && occursin("de_rand", string(globalOptimizer)) == true
      info("$(globalOptimizer) evaluates about one new point at a time: extra workers give little speed-up. In parallel, prefer :dxnes, :xnes or :separable_nes.")
    end

  end

  if nworkers() == 1
    if verbose == true
      info("Starting optimization in serial")
    end
    sMMProblem.bbSetup = bbsetup(sMMProblem.objective_function;
                              Method = sMMProblem.options.globalOptimizer,
                              SearchRange = mySearchRange,
                              MaxFuncEvals = sMMProblem.options.maxFuncEvals,
                              TraceMode = traceMode,
                              PopulationSize = sMMProblem.options.populationSize,
                              NumDimensions = nbDimensions,
                              extraOptions...)
  else
    if verbose == true
      info("Starting optimization in parallel")
    end
    sMMProblem.bbSetup = bbsetup(sMMProblem.objective_function;
                                Method = sMMProblem.options.globalOptimizer,
                                SearchRange = mySearchRange,
                                MaxFuncEvals = sMMProblem.options.maxFuncEvals,
                                Workers = workers(),
                                PopulationSize = sMMProblem.options.populationSize,
                                TraceMode = traceMode,
                                NumDimensions = nbDimensions,
                                extraOptions...)
  end


end

"""
  nes_lambda(globalOptimizer::Symbol, lambda::Int64, nbDimensions::Int64, nbWorkers::Int64)

Number of points λ evaluated per generation by a natural evolution strategy of BlackBoxOptim
(`:dxnes`, `:xnes`, `:separable_nes`). If `lambda > 0`, returns `lambda`. If `lambda == 0`
(automatic), returns BlackBoxOptim's default for `nbDimensions` parameters, rounded up to a
multiple of `nbWorkers`: the points of a generation are evaluated in parallel, and the next
generation starts when all of them are done, so no worker is idle. `:dxnes` requires an even λ:
when the multiple is odd, λ is decreased by one (one worker is idle at the end of each generation).

# Examples
```julia-repl
julia> nes_lambda(:dxnes, 0, 5, 1)    # BlackBoxOptim's default for 5 parameters
8
julia> nes_lambda(:dxnes, 0, 5, 32)
32
julia> nes_lambda(:dxnes, 0, 5, 11)   # 11 is odd: one worker idle
10
julia> nes_lambda(:dxnes, 0, 10, 4)   # default 10, rounded up to a multiple of 4
12
```
"""
function nes_lambda(globalOptimizer::Symbol, lambda::Int64, nbDimensions::Int64, nbWorkers::Int64)

  if is_nes_optimizer(globalOptimizer) == false
    error("globalOptimizer = $(globalOptimizer) is not a natural evolution strategy (:dxnes, :xnes, :separable_nes).")
  end

  if lambda < 0
    error("lambda must be >= 0 (0 = automatic).")
  end

  if globalOptimizer == :dxnes && isodd(lambda)
    error("globalOptimizer = :dxnes requires an even lambda (or lambda = 0, automatic).")
  end

  # Value set by the user
  if lambda > 0
    return lambda
  end

  # BlackBoxOptim's default (v0.6)
  if globalOptimizer == :dxnes
    lambdaDefault = 4 + 3*floor(Int, log(nbDimensions))
    lambdaDefault += isodd(lambdaDefault) ? 1 : 0
  elseif globalOptimizer == :xnes
    lambdaDefault = 4 + 3*floor(Int, log(nbDimensions))
  else
    lambdaDefault = 4 + ceil(Int, log(3*nbDimensions))
  end

  # Rounded up to a multiple of the number of workers
  lambdaWorkers = nbWorkers*ceil(Int, lambdaDefault/nbWorkers)

  # dxnes: even λ
  if globalOptimizer == :dxnes && isodd(lambdaWorkers)
    lambdaWorkers -= 1
  end

  return lambdaWorkers

end

"""
  generate_bbSearchRange(sMMProblem::MSMProblem)

Function to generate a search range that matches the convention used by
BlackBoxOptim.
"""
function generate_bbSearchRange(sMMProblem::MSMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [(sMMProblem.priors[k][2], sMMProblem.priors[k][3]) for k in keys(sMMProblem.priors)]
end

"""
  create_lower_bound(sMMProblem::MSMProblem)

Function to generate a lower bound used by Optim when minimizing with Fminbox.
The lower bound is of type Array{Float64,1}.
"""
function create_lower_bound(sMMProblem::MSMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [sMMProblem.priors[k][2] for k in keys(sMMProblem.priors)]
end

"""
  create_upper_bound(sMMProblem::MSMProblem)

Function to generate an upper bound used by Optim when minimizing with Fminbox.
The upper bound is of type Array{Float64,1}.
"""
function create_upper_bound(sMMProblem::MSMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [sMMProblem.priors[k][3] for k in keys(sMMProblem.priors)]
end



# Function coded by Robert Feldt. All credits to him. I only changed two lines:
# * cubedim = Vector{T}(n) instead of cubedim = Vector{T}(undef, n)
# * I return return transpose(result) instead of result
# source: https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/src/utilities/latin_hypercube_sampling.jl
"""
    latin_hypercube_sampling(mins, maxs, n)

Randomly sample `n` vectors from the parallelogram defined
by `mins` and `maxs` using the Latin hypercube algorithm.
Returns an `n`×`dims` matrix (one point per row).
"""
function latin_hypercube_sampling(mins::AbstractVector{T},
                                  maxs::AbstractVector{T},
                                  n::Integer) where T<:Number
    length(mins) == length(maxs) ||
        throw(DimensionMismatch("mins and maxs should have the same length"))
    all(xy -> xy[1] <= xy[2], zip(mins, maxs)) ||
        throw(ArgumentError("mins[i] should not exceed maxs[i]"))
    dims = length(mins)
    result = zeros(T, dims, n)
    # Julia 0.7
    #----------
    # cubedim = Vector{T}(undef, n)
    # Julia 0.6.4
    #------------
    cubedim = Vector{T}(undef, n)
    @inbounds for i in 1:dims
        imin = mins[i]
        dimstep = (maxs[i] - imin) / n
        for j in 1:n
            cubedim[j] = imin + dimstep * (j - 1 + rand(T))
        end
        result[i, :] .= shuffle!(cubedim)
    end
    return transpose(result)
end

"""
    latin_hypercube_sampling(mySearchRange::Vector{Tuple{Float64, Float64}}, n::Integer; gens::Integer = 100)

Function to create an optimised Latin Hypercube Sampling Plan
## Input
* mySearchRange: a vector of tuple of the form (xj_min, xj_max)
* n: number of points to draw
* gens: (optional) optimization is run for gens generations. See LatinHypercubeSampling.jl

## Output
* Matrix{Float64}: row = observation; column = dimension
"""
function latin_hypercube_sampling(mySearchRange::Vector{Tuple{Float64, Float64}}, n::Integer; gens::Integer = 100)

        d = length(mySearchRange) #dimension
        # See: https://mrurq.github.io/LatinHypercubeSampling.jl/stable/man/lhcoptim/
        plan, _ = LHCoptim(n,d,gens)
        # Rescale plan:
        scaled_plan = scaleLHC(plan, mySearchRange)

        return scaled_plan
end


function sobol_sampling(lb::Vector{Float64},
                        ub::Vector{Float64},
                        n::Integer)
    s = SobolSeq(lb,ub)
    # "If you know in advance the number n of points that you plan to generate,
    # some authors suggest that better uniformity can be attained by first skipping
    # the initial portion of the LDS.". See: https://github.com/stevengj/Sobol.jl
    skip(s,n)
    d = length(lb) #dimension
    if d == 1
        result = transpose(reduce(hcat, next!(s)[1] for i = 1:n))
    else
        result = transpose(reduce(hcat, next!(s) for i = 1:n))
    end
    return result
end


"""
  get_now()

Returns date and time in a manner that does not clash with Windows, Linux and OSX
(e.g. "2026-09-30--14h-5m-3s")
"""
function get_now()

  # Read the clock once: separate calls could straddle a second, minute or day boundary
  t = Dates.now()
  "$(Dates.Date(t))--$(Dates.hour(t))h-$(Dates.minute(t))m-$(Dates.second(t))s"

end

"""
    info(text)

To display information to user
"""
function info(text)
    @info text
end

"""
  linspace(z_start::Real, z_end::Real, z_n::Int64)

Vector of z_n evenly spaced points from z_start to z_end (similar to Base.linspace on julia v. < 0.7)
"""
function linspace(z_start::Real, z_end::Real, z_n::Int64)
    return collect(range(z_start,stop=z_end,length=z_n))
end
