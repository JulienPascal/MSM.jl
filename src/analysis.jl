"""
  msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)

Function to "slice" the objective function. That is, to hold the variables constant,
except for one dimension.
"""
function msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)

    # Create a grid in the neighborhood of the minimizer
    lb_slice = paramValues .- abs.(paramValues) .* offset;
    ub_slice = paramValues .+ abs.(paramValues) .* offset;

    #Checks
    if nbPoints < 3
      error("nbPoints must be >= 3")
    end
    if mod(nbPoints,2) == 0
      error("nbPoints must be odd")
    end

    nbPointsBelow = round(Int, ceil(nbPoints/2))
    nbPointAbove = nbPoints - nbPointsBelow

    vXGrid = zeros(nbPoints, length(keys(sMMProblem.priors)))
    vYGrid = zeros(nbPoints, length(keys(sMMProblem.priors)))


    # Loop over parameter values
    #---------------------------
    for (keyIndex, keyValue) in enumerate(keys(sMMProblem.priors))

        #Create one grid below the minimizer (which also contains the minimizer)
        grid_below = linspace(lb_slice[keyIndex], paramValues[keyIndex], nbPointsBelow);
        #Create one grid above the minimizer (which does not contain the minimizer)
        grid_above = linspace(paramValues[keyIndex], ub_slice[keyIndex], nbPointsBelow);
        grid_above = grid_above[2:end]; #exclude the minimizer from the grid (already in grid_below)
        # store grid points:
        vXGrid[:,keyIndex] = vcat(grid_below, grid_above);
        vYGrid[:,keyIndex] = zeros(nbPoints)

        info("slicing along $(keyValue)")

        # Move along one dimension, keep other values constant
        localParamValues = transpose(repeat(paramValues,outer=[1,nbPoints]))
        localParamValues[:, keyIndex] = vXGrid[:, keyIndex]

        # Use pmap to use several workers in parallel:
        vYGrid[:, keyIndex] = pmap(sMMProblem.objective_function, eachrow(localParamValues))

    end

    return vXGrid, vYGrid

end


#= OLD VERSION THAT USES SharedArrays
"""
  msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)
Function to "slice" the objective function. That is, to hold the variables constant,
except for one dimension.
"""
function msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)

    # Create a grid in the neighborhood of the minimizer
    lb_slice = paramValues .- abs.(paramValues) .* offset;
    ub_slice = paramValues .+ abs.(paramValues) .* offset;

    #Checks
    if nbPoints < 5
      error("nbPoints must be >= 5")
    end
    if mod(nbPoints,2) == 0
      error("nbPoints must be odd")
    end


    vXGrid = SharedArray(zeros(round(Int, 2*floor(nbPoints/2) - 1), length(keys(sMMProblem.priors))))
    vYGrid = SharedArray(zeros(round(Int, 2*floor(nbPoints/2) - 1), length(keys(sMMProblem.priors))))

    # Loop over parameter values
    #---------------------------
    for (keyIndex, keyValue) in enumerate(keys(sMMProblem.priors))


        #Create one grid below the minimizer (which also contains the minimizer)
        grid_below = linspace(lb_slice[keyIndex], paramValues[keyIndex], round(Int, floor(nbPoints/2)));
        #Create one grid abive the minimizer (which does not contain the minimizer)
        grid_above = linspace(paramValues[keyIndex], ub_slice[keyIndex], round(Int, floor(nbPoints/2)));
        grid_above = grid_above[2:end]; #exclude the minimizer from the grid (already in grid_below)
        # store grid points:
        vXGrid[:,keyIndex] = vcat(grid_below, grid_above);

        #vXGrid[:,keyIndex] = collect(linspace(sMMProblem.priors[keyValue][2], sMMProblem.priors[keyValue][3], nbPoints))
        vYGrid[:,keyIndex] = zeros(round(Int, 2*floor(nbPoints/2) - 1))

        info("slicing along $(keyValue)")
        # Options A.
        #-----------
        @sync @distributed for xIndex = 1:1:size(vXGrid,1)

          # Move along one dimension, keeping other values constant
          #--------------------------------------------------------
          localParamValues = copy(paramValues)
          localParamValues[keyIndex] = copy(vXGrid[xIndex, keyIndex])

          vYGrid[xIndex, keyIndex] = sMMProblem.objective_function(localParamValues)

        end

    end

    return vXGrid, vYGrid

end
=#
