"""
  msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)

Function to "slice" the objective function. That is, to hold the variables constant,
except for one dimension. Each parameter θ varies in [θ - offset*|θ|, θ + offset*|θ|],
or in [θ - offset*range, θ + offset*range] when θ is (numerically) zero, where range is
the width of the prior of θ.
"""
function msm_slices(sMMProblem::MSMProblem, paramValues::Vector; nbPoints::Int64 = 5, offset::Float64 = 0.001)

    # Scale of the slice: |θ| (percent deviation), or the width of the prior when θ is (numerically) zero
    rangePrior = create_upper_bound(sMMProblem) .- create_lower_bound(sMMProblem)
    isZero = abs.(paramValues) .<= sqrt(eps()) .* rangePrior
    scale = ifelse.(isZero, rangePrior, abs.(paramValues))

    # Create a grid in the neighborhood of the minimizer
    lb_slice = paramValues .- scale .* offset;
    ub_slice = paramValues .+ scale .* offset;

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

        # Use pmap to use several workers in parallel
        # (pass plain vectors, not row views: the user's function may require a Vector)
        vYGrid[:, keyIndex] = pmap(sMMProblem.objective_function, [localParamValues[row, :] for row = 1:nbPoints])

    end

    return vXGrid, vYGrid

end

