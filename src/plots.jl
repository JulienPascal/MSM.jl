"""
  smm_slices(sMMProblem::MSMProblem)

Function to plot slices of the objective function.
"""
function smm_slices(sMMProblem::MSMProblem, paramValues::Vector, nbPoints::Int64; showPlot::Bool = false)

    list_plots = []
    p = Plots.plot()

    # Loop over parameter values
    #---------------------------
    for (keyIndex, keyValue) in enumerate(keys(sMMProblem.priors))

        vXGrid = SharedArray(collect(linspace(sMMProblem.priors[keyValue][2], sMMProblem.priors[keyValue][3], nbPoints)))
        vYGrid = SharedArray(zeros(nbPoints))

        info("slicing along $(keyValue)")
        # Options A.
        #-----------
        @sync @distributed for xIndex = 1:1:length(vXGrid)

          # Move along one dimension, keeping other values constant
          #--------------------------------------------------------
          localParamValues = copy(paramValues)
          localParamValues[keyIndex] = vXGrid[xIndex]

          vYGrid[xIndex] = sMMProblem.objective_function(localParamValues)

        end

        p = Plots.plot(vXGrid, vYGrid, title = "$(keyValue)", label = "")
        push!(list_plots, p)

    end

    #Let's combine all the plots in a single plot
    # s0 = ""
    # for i = 1:length(keys(sMMProblem.priors))
    #     if i==1
    #         s0 = string("list_plots[$(i)]" )
    #     else
    #         s0 = string(s0, ", ", "list_plots[$(i)]" )
    #     end
    # end

    # print(string("plot(", s0, ")"))
    # plot_combined = eval(Meta.parse(string("plot(", s0, ")")))
    #
    # if showPlot == true
    #     display(plot_combined)
    # end

    #return plot_combined, list_plots
    return list_plots

end
