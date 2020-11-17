"""
  set_simulate_empirical_moments!(sMMProblem::SMMProblem, f::Function)

Function to set the field simulate_empirical_moments for a SMMProblem.
The function simulate_empirical_moments takes parameter values and return
the corresponding simulate moments values.
"""
function set_simulate_empirical_moments!(sMMProblem::SMMProblem, f::Function)

  # set the function that returns an ordered dictionary
  sMMProblem.simulate_empirical_moments = f

  # set the function that returns an array (respecting the order of the ordered dict)
  # this function is used to calculate the jacobian
  function simulate_empirical_moments_array(x)

    momentsODict = sMMProblem.simulate_empirical_moments(x)

    momentsArray = Array{Float64}(length(momentsODict))

    for (i, k) in enumerate(keys(momentsODict))
        momentsArray[i] = momentsODict[k]
    end

    return momentsArray

  end

  sMMProblem.simulate_empirical_moments_array = simulate_empirical_moments_array

end

"""
  construct_objective_function!(sMMProblem::SMMProblem, f::Function)

Function that construct an objective function, using the function
SMMProblem.simulate_empirical_moments. For the moment, the objective function
returns the mean of the percentage deviation of simulated moments from
their true value.
"""
function construct_objective_function!(sMMProblem::SMMProblem, objectiveType::Symbol = :percent)


    function objective_function_percent(x)

          # Initialization
          #----------------
          distanceEmpSimMoments = sMMProblem.options.penaltyValue

          #------------------------------------------------------------------------
          # A.
          # Generate simulated moments
          #------------------------------------------------------------------------
          simulatedMoments, convergence = try

          sMMProblem.simulate_empirical_moments(x), 1

      catch errorSimulation

            info("An error occured with parameter values = $(x)")
            info("$(errorSimulation)")

            OrderedDict{String,Array{Float64,1}}(), 0

      end

      #------------------------------------------------------------------------
      # B.
      # If generating moment was successful, calculate distance between empirical
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
          arrayDistance[indexMoment] = ((sMMProblem.empiricalMoments[k][1] - simulatedMoments[k]) /sMMProblem.empiricalMoments[k][2])^2

        end

        distanceEmpSimMoments = mean(arrayDistance)

        if sMMProblem.options.showDistance == true
          println("distance = $(distanceEmpSimMoments)")
        end

      end

      return distanceEmpSimMoments

    end


  if objectiveType == :percent
    sMMProblem.objective_function = objective_function_percent
  else
    error("Error in construct_objective_function. objectiveType currently only support :percent.")
  end


end

"""
  set_priors!(sMMProblem::SMMProblem, priors::OrderedDict{String,Array{Float64,1}})

Function to change the field sMMProblem.priors
"""
function set_priors!(sMMProblem::SMMProblem, priors::OrderedDict{String,Array{Float64,1}})

  sMMProblem.priors = priors

end

"""
   set_empirical_moments!(sMMProblem::SMMProblem, empiricalMoments::OrderedDict{String,Array{Float64,1}})

Function to change the field sMMProblem.empiricalMoments
"""
function set_empirical_moments!(sMMProblem::SMMProblem, empiricalMoments::OrderedDict{String,Array{Float64,1}})

  sMMProblem.empiricalMoments = empiricalMoments

end

"""
   set_Sigma0!(sMMProblem::SMMProblem, Sigma0::Array{Float64,2})

Function to change the field sMMProblem.Sigma0, where Sigma0 is the distance matrix,
in the terminology of Duffie and Singleton (1993)
"""
function  set_Sigma0!(sMMProblem::SMMProblem, Sigma0::Array{Float64,2})

  sMMProblem.Sigma0 = Sigma0

end

"""
  set_global_optimizer!(sMMProblem::SMMProblem)

Function to set the fields corresponding to the global
optimizer problem.
"""
function set_global_optimizer!(sMMProblem::SMMProblem)

  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true

    set_bbSetup!(sMMProblem)

  else

    Base.error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end

end

"""
  set_bbSetup!(sMMProblem::SMMProblem)

Function to set the field bbSetup for a SMMProblem.
"""
function set_bbSetup!(sMMProblem::SMMProblem)

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
                              MaxFuncEvals = sMMProblem.options.saveSteps,
                              TraceMode = :verbose,
                              PopulationSize = sMMProblem.options.populationSize,
                              NumDimensions = length(keys(sMMProblem.priors)))
  else
    info("Starting optimization in parallel")
    sMMProblem.bbSetup = bbsetup(sMMProblem.objective_function;
                                Method = sMMProblem.options.globalOptimizer,
                                SearchRange = mySearchRange,
                                MaxFuncEvals = sMMProblem.options.saveSteps,
                                Workers = workers(),
                                PopulationSize = sMMProblem.options.populationSize,
                                TraceMode = :verbose,
                                NumDimensions = length(keys(sMMProblem.priors)))
  end


end

"""
  generate_bbSearchRange(sMMProblem::SMMProblem)

Function to generate a search range that matches the convention used by
BlackBoxOptim.
"""
function generate_bbSearchRange(sMMProblem::SMMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [(sMMProblem.priors[k][2], sMMProblem.priors[k][3]) for k in keys(sMMProblem.priors)]
end

"""
  create_lower_bound(sMMProblem::SMMProblem)

Function to generate a lower bound used by Optim when minimizing with Fminbox.
The lower bound is of type Array{Float64,1}.
"""
function create_lower_bound(sMMProblem::SMMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [sMMProblem.priors[k][2] for k in keys(sMMProblem.priors)]
end

"""
  create_upper_bound(sMMProblem::SMMProblem)

Function to generate a lower bound used by Optim when minimizing with Fminbox.
The upper bound is of type Array{Float64,1}.
"""
function create_upper_bound(sMMProblem::SMMProblem)

  # sMMProblem.priors["key"][1] contains the initial guess
  # sMMProblem.priors["key"][2] contains the lower bound
  # sMMProblem.priors["key"][3] contains the upper bound
  #-----------------------------------------------------
  [sMMProblem.priors[k][3] for k in keys(sMMProblem.priors)]
end

"""
  create_grid(a::Array{Float64,1}, b::Array{Float64,1}, nums::Int64)

Function to create a grid. a is a vector of lower
bounds, b a vector of upper bounds and nums is the number of points
along each dimension. The type of grid to use can be specified using gridType.
By default, the functions returns the cartesian product. The output is an
Array{Float64,2}, where each row is a new point and each column is a dimension
of this points.
"""
function create_grid(a::Array{Float64,1}, b::Array{Float64,1}, nums::Int64; gridType::Symbol = :lin)

    #Safety checks
    #-------------
    if nums < 2
        Base.error("The input nums should be >= 2. nums = $(nums).")
    end

    if length(a) != length(b)
        Base.error("length(a) != length(b)")
    end

    if gridType != :cheb && gridType == :spli && gridType == :lin
        Base.error("gridType has to be either :chebn, :spli or :lin")
    end

    Fspace = fundefn(gridType, collect([nums for i =1:length(a)]), a, b)

    Fnodes = funnode(Fspace)[1]

end





# Function coded by Robert Feldt. All credits to him. I only changed two lines:
# * cubedim = Vector{T}(n) instead of cubedim = Vector{T}(undef, n)
# * I return return transpose(result) instead of result, to be consistent with
# the function create_grid
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
  get_now()

Returns date and time in a manner that does not clash with Windows, Linux and OSX
"""
function get_now()

  "$(Dates.today())--$(Dates.hour(Dates.now()))h-$(Dates.minute(Dates.now()))m-$(Dates.second(Dates.now()))s"

end
