"""
  msm_optimize!(sMMProblem::MSMProblem; verbose::Bool = true)

Function to launch an optimization. To be used after the following functions
have been called: (i) set_empirical_moments! (ii) set_priors!
(iii) set_simulate_empirical_moments! (iv) construct_objective_function!
"""
function msm_optimize!(sMMProblem::MSMProblem; verbose::Bool = true)

  # Initialize a BlackBoxOptim problem
  # this modifies sMMProblem.bbSetup
  #-----------------------------------
  set_global_optimizer!(sMMProblem)

  # If the global optimizer is using BlackBoxOptim
  #-----------------------------------------------
  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true


    # Store best fitness and best candidates
    listBestFitness = []
    listBestCandidates = []

    # Run the optimization with BlackBoxOptim
    #----------------------------------------
    sMMProblem.bbResults = bboptimize(sMMProblem.bbSetup)

    push!(listBestFitness, best_fitness(sMMProblem.bbResults))
    push!(listBestCandidates, best_candidate(sMMProblem.bbResults))


  # In the future, we may use other global minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end


  return listBestFitness, listBestCandidates


end

"""
  function msm_minimizer(sMMProblem::MSMProblem)

Function to get the parameter value minimizing the objective function
"""
function msm_minimizer(sMMProblem::MSMProblem)

  # If the global optimizer is using BlackBoxOptim
  #-----------------------------------------------
  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true

    best_candidate(sMMProblem.bbResults)

  # In the future, we may use other global minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end

end


"""
  function msm_minimum(sMMProblem::MSMProblem)

Function to get the minimum of the objective function
"""
function msm_minimum(sMMProblem::MSMProblem)

  # If the global optimizer is using BlackBoxOptim
  #-----------------------------------------------
  if is_bb_optimizer(sMMProblem.options.globalOptimizer) == true

    best_fitness(sMMProblem.bbResults)

  # In the future, we may use other global minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.globalOptimizer = $(sMMProblem.options.globalOptimizer) is not supported.")

  end

end


"""
  msm_refine_globalmin!(sMMProblem::MSMProblem; verbose::Bool = true)

Function to refine the global minimum using a local minimization routine.
To be used after the following functions have been called: (i) set_empirical_moments!
(ii) set_priors! (iii) set_simulate_empirical_moments! (iv) construct_objective_function!
(v) msm_optimize!
"""
function msm_refine_globalmin!(sMMProblem::MSMProblem; verbose::Bool = true)


  x0 = msm_minimizer(sMMProblem)

  # Let's use the result from the global maximizer as the starting value
  #---------------------------------------------------------------------
  if verbose == true
    info("Refining the global maximum using a local algorithm.")
    info("Using Fminbox = $(sMMProblem.options.minBox)")
    info("Starting value = $(x0)")
  end


  if is_local_optimizer(sMMProblem.options.localOptimizer) == true

    # If using Fminbox option is true
    #-------------------------------
    if sMMProblem.options.minBox == true

        lower = create_lower_bound(sMMProblem)
        upper = create_upper_bound(sMMProblem)

        sMMProblem.optimResults = optimize(sMMProblem.objective_function, lower, upper, x0, convert_to_fminbox(sMMProblem.options.localOptimizer), Optim.Options(iterations = sMMProblem.options.maxFuncEvals))

    else
        sMMProblem.optimResults = optimize(sMMProblem.objective_function, x0, convert_to_optim_algo(sMMProblem.options.localOptimizer), Optim.Options(iterations = sMMProblem.options.maxFuncEvals))
    end

  # In the future, we may use other local minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")

  end

  return Optim.minimizer(sMMProblem.optimResults)

end


"""
  function msm_local_minimizer(sMMProblem::MSMProblem)

Function to get the parameter value minimizing the objective function (local)
"""
function msm_local_minimizer(sMMProblem::MSMProblem)

  # If the global optimizer is using BlackBoxOptim
  #-----------------------------------------------
  if is_optim_optimizer(sMMProblem.options.localOptimizer) == true

    Optim.minimizer(sMMProblem.optimResults)

  # In the future, we may use other global minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")

  end

end

"""
  function msm_local_minimum(sMMProblem::MSMProblem)

Function to get the local minimum value of the objetive function
"""
function msm_local_minimum(sMMProblem::MSMProblem)

  # If the global optimizer is using BlackBoxOptim
  #-----------------------------------------------
  if is_optim_optimizer(sMMProblem.options.localOptimizer) == true

    Optim.minimum(sMMProblem.optimResults)

  # In the future, we may use other global minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")

  end

end


"""
  function msm_multistart_minimizer(sMMProblem::MSMProblem)

Function to get the parameter value minimizing the objective function when
using the multistart algorithm
"""
function msm_multistart_minimizer(sMMProblem::MSMProblem)

  # Result given by msm_local_minimizer.
  msm_local_minimizer(sMMProblem::MSMProblem)

end

"""
  function msm_multistart_minimum(sMMProblem::MSMProblem)

Function to get the minimum value of the objetive function when
using the multistart algorithm
"""
function msm_multistart_minimum(sMMProblem::MSMProblem)

  # Result given by msm_local_minimum
  msm_local_minimum(sMMProblem::MSMProblem)

end


"""
  msm_multistart!(sMMProblem::MSMProblem; x0 = Array{Float64}(undef, 0,0), nums::Int64 = nworkers(), verbose::Bool = true)

Function to run several local minimization algorithms in parallel, with different
starting values. The minimum is calculated as the minimum of the local minima.
Changes sMMProblem.optimResults. This function also returns
a list containing Optim results.
"""
function msm_multistart!(sMMProblem::MSMProblem; x0 = Array{Float64}(undef, 0,0), nums::Int64 = nworkers(), verbose::Bool = true)

  # Safety checks
  #--------------
  if nums < nworkers()
    errors("nums < nworkers()")
  elseif nums > nworkers()
    info("nums > nworkers(). Some starting values will be ignored.")
  end

  # To store minization results
  #----------------------------
  results = []

  # Look for valid starting values (for which convergence is reached)
  #-------------------------------------------------------------------
  if x0 == Array{Float64}(undef, 0,0)
    myGrid = search_starting_values(sMMProblem, nums, verbose = verbose)
  # Using starting values provided by the user
  #-------------------------------------------
  else

    # Check that enough starting values were provided
    if size(x0, 1) < nworkers()
      info("$(size(x0, 1)) starting value(s) were provided.")
      error("The minimum number of starting value(s) to provide is $(nworkers()).")
    end

    myGrid = x0

  end


  # If the local optimizer is using Optim
  #--------------------------------------
  if is_optim_optimizer(sMMProblem.options.localOptimizer) == true

      # A. Starting tasks on available workers
      #---------------------------------------
      @sync for (workerIndex, w) in enumerate(workers())

        @async push!(results, @fetchfrom w wrap_msm_localmin(sMMProblem, myGrid[workerIndex,:], verbose = true))

      end


    # B. Looking for the minimum
    #----------------------------
    # Initialization
    minIndex = 0
    minValue = Inf
    minimizerValue = zeros(length(keys(sMMProblem.priors)))
    nbConvergenceReached = 0
    listOptimResults = []

    for (workerIndex, w) in enumerate(workers())

      push!(listOptimResults, results[workerIndex])

      try

        minimumValue = Optim.minimum(results[workerIndex])
        minimizer = Optim.minimizer(results[workerIndex])

        if minimumValue < minValue && Optim.converged(results[workerIndex]) == true

          minIndex = workerIndex
          minValue = minimumValue
          minimizerValue = minimizer
          nbConvergenceReached += 1

        end

      catch myError
        info("$(myError)")
      end

    end


    # D. If none of the optimization converged
    #-----------------------------------------
    if minIndex == 0
      info("None of the local optimizer algorithm converged.")
    else
      info("Convergence reached for $(nbConvergenceReached) worker(s).")
      info("Minimum value found with worker $(minIndex)")
      sMMProblem.optimResults = results[minIndex]
    end

  # In the future, we may use other local minimizer
  # routines. For the moment, let's return an error
  #-------------------------------------------------
  else

    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")

  end

  if verbose == true
    if nbConvergenceReached != 0
      info("Best value found with starting values = $(myGrid[minIndex,:]).")
      info("Best value = $(minValue).")
      info("Minimizer = $(minimizerValue)")
    end
  end

  return listOptimResults

end


"""
  msm_localmin(sMMProblem::MSMProblem, x0::Array{Float64,1}; verbose::Bool = true)

Function find a local minimum using a local minimization routine, with starting value x0.
To be used after the following functions have been called: (i) set_empirical_moments!
(ii) set_priors! (iii) set_simulate_empirical_moments! (iv) construct_objective_function!
"""
function msm_localmin(sMMProblem::MSMProblem, x0::Array{Float64,1}; verbose::Bool = true)

    # Let's use the result from the global maximizer as the starting value
    #---------------------------------------------------------------------
    if verbose == true
        info("Starting value = $(x0)")
        info("Using Fminbox = $(sMMProblem.options.minBox)")
    end


    if is_local_optimizer(sMMProblem.options.localOptimizer) == true

    # If using Fminbox option is true
    #-------------------------------
    if sMMProblem.options.minBox == true

        lower = create_lower_bound(sMMProblem)
        upper = create_upper_bound(sMMProblem)

        optimResults = optimize(sMMProblem.objective_function, lower, upper, x0, convert_to_fminbox(sMMProblem.options.localOptimizer), Optim.Options(iterations = sMMProblem.options.maxFuncEvals))

    else
        optimResults = optimize(sMMProblem.objective_function, x0, convert_to_optim_algo(sMMProblem.options.localOptimizer), Optim.Options(iterations = sMMProblem.options.maxFuncEvals))
    end

    # In the future, we may use other local minimizer
    # routines. For the moment, let's return an error
    #-------------------------------------------------
    else

    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")

    end

    return optimResults

end


"""

"""
function wrap_msm_localmin(sMMProblem::MSMProblem, x0::Array{Float64,1}; verbose::Bool = true)

  try
    msm_localmin(sMMProblem, x0, verbose = verbose)
  catch myError
    info("$(myError)")
    info("Error with msm_localmin")
  end

end


"""
  search_starting_values(sMMProblem::MSMProblem, numPoints::Int64; verbose::Bool = true)

Search for nums valid starting values. To be used after the following functions have been called:
(i) set_empirical_moments! (ii) set_priors! (iii) set_simulate_empirical_moments!
(iv) construct_objective_function!
"""
function search_starting_values(sMMProblem::MSMProblem, numPoints::Int64; verbose::Bool = true)

  # Safety Check
  #-------------
  if is_optim_optimizer(sMMProblem.options.localOptimizer) == false
    error("sMMProblem.options.localOptimizer = $(sMMProblem.options.localOptimizer) is not supported.")
  end

  if verbose == true
    info("Searching for $(numPoints) valid starting value(s)")
  end

  # Generate upper and lower bounds vector
  #--------------------------------------------------------------------------
  lower_bound = zeros(length(keys(sMMProblem.priors)))
  upper_bound = zeros(length(keys(sMMProblem.priors)))

  for (kIndex, k) in enumerate(keys(sMMProblem.priors))
    lower_bound[kIndex] = sMMProblem.priors[k][2]
    upper_bound[kIndex] = sMMProblem.priors[k][3]
  end

  #Each row is a new point and each column is a dimension of this points.
  #---------------------------------------------------------------------
  Validx0 = zeros(numPoints, length(lower_bound))
  distanceValue = zeros(numPoints) #to store the distance associated to each point
  nbValidx0Found = 0

  # Create many grids (stochastic draws) with many potential points
  #----------------------------------------------------------------
  if verbose == true
    info("Creating $(sMMProblem.options.maxTrialsStartingValues) potential starting value(s)")
    info("gridType = $(sMMProblem.options.gridType)")
  end

  # Generate many points for the grid
  #-----------------------------------------------------------------------------
  if sMMProblem.options.gridType == :LHC
    candidates_starting_values = latin_hypercube_sampling(generate_bbSearchRange(sMMProblem), Int(sMMProblem.options.maxTrialsStartingValues*numPoints))
  elseif sMMProblem.options.gridType == :Sobol
    candidates_starting_values = sobol_sampling(lower_bound, upper_bound, Int(sMMProblem.options.maxTrialsStartingValues*numPoints))
  else
    err("sMMProblem.options.gridType = $(sMMProblem.options.gridType) is not a valid sampling procedure.")
  end

  # Split the grid into chunks
  #-----------------------------------------------------------------------------
  listGrids = []
  i = 1;
  j = i + numPoints - 1;
  for k=1:sMMProblem.options.maxTrialsStartingValues
      push!(listGrids, candidates_starting_values[i:j,:])
      i = i + numPoints;
      j = i + numPoints - 1;
  end

  listGridsIndex = 0

  # Looping until numPoints valid points have been found
  #----------------------------------------------------------------------------
  while nbValidx0Found < numPoints

    results = []
    listGridsIndex += 1

    if listGridsIndex > sMMProblem.options.maxTrialsStartingValues
      error("Maximum number of attempts reached without success. maxTrialsStartingValues = $(sMMProblem.options.maxTrialsStartingValues)")
    end

    # Use available workers to simulate moments
    #------------------------------------------
    @sync for (workerIndex, w) in enumerate(workers())

      @async push!(results, @fetchfrom w sMMProblem.objective_function(listGrids[listGridsIndex][workerIndex, :]))

    end

    # Check for convergence
    #----------------------
    for (workerIndex, w) in enumerate(workers())

      # Set penalty value by default
      distanceValue[workerIndex] = sMMProblem.options.penaltyValue

      try

        distanceValue[workerIndex] = results[workerIndex]

        # discard inf distances, values equal to penaltyValue and values above the threshold
        if isinf(distanceValue[workerIndex]) == false && distanceValue[workerIndex] != sMMProblem.options.penaltyValue && distanceValue[workerIndex] < sMMProblem.options.thresholdStartingValue

          nbValidx0Found +=1

          if nbValidx0Found <= numPoints

            Validx0[nbValidx0Found,:] = listGrids[listGridsIndex][workerIndex, :]
            info("Valid starting value = $(Validx0[nbValidx0Found,:]), distance = $(distanceValue[workerIndex])")
          end

        end

      catch myError
        info("$(myError)")
      end

    end

  end

  # sorting starting values according to distance value (in ascending order)
  p = sortperm(distanceValue) #get the ascending order
  Validx0 = Validx0[p,:]  #re-order rows

  if verbose == true
    info("Found $(nbValidx0Found) valid starting value(s)")
  end

  # If requested, save (valid) starting values generated
  if sMMProblem.options.saveStartingValues == true
    if verbose == true
      info("Saving starting values to disk.")
    end

    tempfilename = "starting_values_"* sMMProblem.options.saveName * ".bson"
    bson(tempfilename, Dict(:Validx0=>Validx0))

    tempfilename = "starting_distances_"* sMMProblem.options.saveName * ".bson"
    bson(tempfilename, Dict(:distanceValue=>distanceValue))

  end

  return Validx0

end
