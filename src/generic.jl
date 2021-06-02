"""
  set_simulate_empirical_moments!(sMMProblem::MSMProblem, f::Function)

Function to set the field simulate_empirical_moments for a MSMProblem.
The function simulate_empirical_moments takes parameter values and return
the corresponding simulate moments values.
"""
function set_simulate_empirical_moments!(sMMProblem::MSMProblem, f::Function)

  # set the function that returns an ordered dictionary
  sMMProblem.simulate_empirical_moments = f

  # set the function that returns an array (respecting the order of the ordered dict)
  # this function is used to calculate the jacobian
  function simulate_empirical_moments_array(x)

    momentsODict = sMMProblem.simulate_empirical_moments(x)
    momentsArray = Array{Float64}(undef, length(momentsODict))

    for (i, k) in enumerate(keys(momentsODict))
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

            info("An error occured with parameter values = $(x)")
            info("$(errorSimulation)")

            OrderedDict{String,Array{Float64,1}}(), 0

      end

      #------------------------------------------------------------------------
      # B. If generating moment was successful, calculate distance between empirical
      # and simulated moments
      # (If no convergence, returns penalty value : sMMProblem.options.penaltyValue)
      #------------------------------------------------------------------------
      if convergence == 1

        # to store the distance between empirical and simulated moments
        arrayDistance = zeros(length(keys(sMMProblem.empiricalMoments)))

        for (indexMoment, k) in enumerate(keys(sMMProblem.empiricalMoments))

          # * sMMProblem.empiricalMoments[k][1] is the empirical moments
          # * sMMProblem.empiricalMoments[k][2] is the weight associated to this moment
          #---------------------------------------------------------------------
          arrayDistance[indexMoment] = (sMMProblem.empiricalMoments[k][1] - simulatedMoments[k])

        end

        # formula is (m - m*)'*W*(m - m*)'
        distanceEmpSimMoments = transpose(arrayDistance)*sMMProblem.W*arrayDistance

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
  set_global_optimizer!(sMMProblem::MSMProblem)

Function to set the fields corresponding to the global
optimizer problem.
"""
function set_global_optimizer!(sMMProblem::MSMProblem)

  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true

    set_bbSetup!(sMMProblem)

  else

    Base.error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end

end

"""
  set_bbSetup!(sMMProblem::MSMProblem)

Function to set the field bbSetup for a MSMProblem.
"""
function set_bbSetup!(sMMProblem::MSMProblem)

  # A. using sMMProblem.priors, generate searchRange:
  #-------------------------------------------------
  mySearchRange = generate_bbSearchRange(sMMProblem)

  info("$(nworkers()) worker(s) detected")
   # Debug:
   #-------
   @sync for (idx, pid) in enumerate(workers())
     @async @spawnat(pid, println("hello"))
   end

  if nworkers() == 1
    info("Starting optimization in serial")
    sMMProblem.bbSetup = bbsetup(sMMProblem.objective_function;
                              Method = sMMProblem.options.globalOptimizer,
                              SearchRange = mySearchRange,
                              MaxFuncEvals = sMMProblem.options.maxFuncEvals,
                              TraceMode = :verbose,
                              PopulationSize = sMMProblem.options.populationSize,
                              NumDimensions = length(keys(sMMProblem.priors)))
  else
    info("Starting optimization in parallel")
    sMMProblem.bbSetup = bbsetup(sMMProblem.objective_function;
                                Method = sMMProblem.options.globalOptimizer,
                                SearchRange = mySearchRange,
                                MaxFuncEvals = sMMProblem.options.maxFuncEvals,
                                Workers = workers(),
                                PopulationSize = sMMProblem.options.populationSize,
                                TraceMode = :verbose,
                                NumDimensions = length(keys(sMMProblem.priors)))
  end


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

Function to generate a lower bound used by Optim when minimizing with Fminbox.
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
Returns `dims`×`n` matrix.
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
"""
function get_now()

  "$(Dates.today())--$(Dates.hour(Dates.now()))h-$(Dates.minute(Dates.now()))m-$(Dates.second(Dates.now()))s"

end

"""
    info(text)

To display information to user
"""
function info(text)
    @info text
end

"""
  linspace(z_n::Int64, z_start::Real, z_end::Real)

Similar behavior of Base.linespace on julia v. < 0.6
"""
function linspace(z_start::Real, z_end::Real, z_n::Int64)
    return collect(range(z_start,stop=z_end,length=z_n))
end
