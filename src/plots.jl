"""
  smm_slices(sMMProblem::SMMProblem)

Function to plot slices of the objective function.
"""
function smm_slices(sMMProblem::SMMProblem, paramValues::Vector, nbPoints::Int64; showPlots::Bool = true)

    listPlots = []
    p = Plots.plot()

    # Loop over parameter values
    #---------------------------
    for (keyIndex, keyValue) in enumerate(keys(sMMProblem.priors))

        vXGrid = SharedArray(collect(linspace(sMMProblem.priors[keyValue][2], sMMProblem.priors[keyValue][3], nbPoints)))
        vYGrid = SharedArray(zeros(nbPoints))

        info("slicing along $(keyValue)")
        # Options A.
        #-----------
        @sync @parallel for xIndex = 1:1:length(vXGrid)

          # Move along one dimension, keeping other values constant
          #--------------------------------------------------------
          localParamValues = copy(paramValues)
          localParamValues[keyIndex] = vXGrid[xIndex]

          vYGrid[xIndex] = sMMProblem.objective_function(localParamValues)

        end

        p = Plots.plot(vXGrid, vYGrid, title = "$(keyValue)")
        push!(listPlots, p)

        if showPlots == true
            display(p)
        end

    end

    return listPlots

end
