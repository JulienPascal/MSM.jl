maxNbWorkers = 1
using Distributed
while nworkers() < maxNbWorkers
  addprocs(maxNbWorkers - nworkers())
end
println(nworkers())

@everywhere using MSM
@everywhere using OrderedCollections
@everywhere using Random
@everywhere using Distributions
@everywhere using Statistics
@everywhere using LinearAlgebra
using Test
using Optim
using BlackBoxOptim
using DataFrames
using StatsBase

# Uncomment to make sure Travis CI works as expected
# @test 1 == 2

# OPTIONS
do_plots = false #To create "visual tests". Set to false when using Travis CI.

@everywhere function eye(n::Int)
    Diagonal(ones(n))
end

# Simulated models used by several testsets
#------------------------------------------
# 1d: the moment is the mean of N(x[1], 1)
@everywhere function functionTest1d(x)
    d = Normal(x[1])
    output = OrderedDict{String,Float64}()
    output["meanU"] = mean(rand(d, 1000000))
    return output
end

# Same, with fixed draws: when using one of the deterministic methods of Optim,
# we can safely "control" for randomness
@everywhere function functionTest1dSeeded(x)
    Random.seed!(1234)
    d = Normal(x[1])
    output = OrderedDict{String,Float64}()
    output["meanU"] = mean(rand(d, 1000000))
    return output
end

# 2d: the moments are the means of N([x[1], x[2]], I)
@everywhere function functionTest2d(x)
    d = MvNormal( [x[1]; x[2]], eye(2))
    output = OrderedDict{String,Float64}()
    draws = rand(d, 1000000)
    output["mean1"] = mean(draws[1,:])
    output["mean2"] = mean(draws[2,:])
    return output
end

# Same, with fixed draws
@everywhere function functionTest2dSeeded(x)
    Random.seed!(1234)
    d = MvNormal( [x[1]; x[2]], eye(2))
    output = OrderedDict{String,Float64}()
    draws = rand(d, 1000000)
    output["mean1"] = mean(draws[1,:])
    output["mean2"] = mean(draws[2,:])
    return output
end

@testset "MSM.jl" begin


    @testset "testing Types" begin

        @testset "testing MSMOptions" begin

            t = MSMOptions()

            # Testing default values
            #-----------------------------------------------------------------------
            @test t.globalOptimizer == :dxnes
            @test t.localOptimizer == :LBFGS
            @test t.maxFuncEvals == 1000
            @test t.showDistance == false
            @test t.minBox == false
            @test t.populationSize == 50
            @test t.lambda == 0
            @test t.penaltyValue == Inf
            @test t.gridType == :LHC
            @test t.saveStartingValues == false
            @test t.maxTrialsStartingValues == 20
            @test t.thresholdStartingValue == t.penaltyValue/10.0

        end


        @testset "testing MSMOptions" begin

            t = MSMProblem()

            # When initialized, iter is equal to 0
            @test typeof(t.priors) == OrderedDict{String,Array{Float64,1}}
            @test typeof(t.empiricalMoments) == OrderedDict{String,Array{Float64,1}}
            @test typeof(t.simulatedMoments) == OrderedDict{String, Float64}
            @test typeof(t.distanceEmpSimMoments) == Float64
            # the functions t.simulate_empirical_moments are initialized with x->x
            @test t.simulate_empirical_moments(1.0) == 1.0
            @test t.objective_function(1.0) == 1.0
            @test typeof(t.options) == MSMOptions

            # No optimization results before optimizing: explicit error
            @test t.bbSetup === nothing
            @test t.bbResults === nothing
            @test t.optimResults === nothing
            @test_throws ErrorException msm_minimizer(t)
            @test_throws ErrorException msm_minimum(t)
            @test_throws ErrorException msm_local_minimizer(t)
            @test_throws ErrorException msm_local_minimum(t)


        end



    end #end "testing Types"

    @testset "Latin hypercube sampling" begin

        # Test the home-made function
        a = zeros(3)
        b = ones(3)
        nums = 100
        nums = 10
        points = latin_hypercube_sampling(a, b, nums)

        @test size(points, 1) == nums
        @test size(points, 2) == length(a)

        for i=1:size(points, 1)
            for j=1:size(points, 2)
                @test points[i,j] >= a[j]
                @test points[i,j] <= b[j]
            end
        end


        # Test the function using the package LatinHypercubeSampling.jl
        myProblem = MSMProblem(options = MSMOptions());
        dictPriors = OrderedDict{String,Array{Float64,1}}()
        # Of the form: [initial_guess, lower_bound, upper_bound]
        dictPriors["alpha"] = [0.5, 0.0, 1.0]
        dictPriors["beta1"] = [0.5, 0.0, 1.0]
        dictPriors["beta2"] = [0.5, 0.0, 1.0]
        set_priors!(myProblem, dictPriors)
        points_2 = latin_hypercube_sampling(generate_bbSearchRange(myProblem), nums)

        @test size(points_2, 1) == nums
        @test size(points_2, 2) == length(a)

        for i=1:size(points_2, 1)
            for j=1:size(points_2, 2)
                @test points_2[i,j] >= a[j]
                @test points_2[i,j] <= b[j]
            end
        end

        if do_plots == true
            using Plots
            gr()
            p1 = scatter(points[:,1], points[:,2], points[:,3], title="Home made")
            p2 = scatter(points_2[:,1], points_2[:,2], points_2[:,3], title="LatinHypercubeSampling.jl")
            plot(p1,p2 )
        end

    end

    @testset "Sobol sampling" begin

        # Test the home-made function
        a = zeros(3)
        b = ones(3)
        nums = 100
        points = sobol_sampling(a, b, nums)

        @test size(points, 1) == nums
        @test size(points, 2) == length(a)

        for i=1:size(points, 1)
            for j=1:size(points, 2)
                @test points[i,j] >= a[j]
                @test points[i,j] <= b[j]
            end
        end

        # For fun, let's compare to random sampling
        points_2 = rand(nums, length(a))

        if do_plots == true
            using Plots
            gr()
            p1 = scatter(points_2[:,1], points_2[:,2], points_2[:,3], title="Random")
            p2 = scatter(points[:,1], points[:,2], points[:,3], title="Sobol")
            plot(p1,p2 )
        end

    end


    @testset "testing checks on global algo" begin

        listValidGlobalOptimizers = [:dxnes, :adaptive_de_rand_1_bin_radiuslimited, :xnes,
                         :de_rand_1_bin_radiuslimited, :adaptive_de_rand_1_bin,
                         :generating_set_search, :de_rand_1_bin,
                         :separable_nes, :resampling_inheritance_memetic_search,
                         :probabilistic_descent, :resampling_memetic_search,
                         :de_rand_2_bin_radiuslimited, :de_rand_2_bin,
                         :random_search, :simultaneous_perturbation_stochastic_approximation]


        for globalOptim in listValidGlobalOptimizers

            @test is_global_optimizer(globalOptim) == true

        end

    end


    @testset "testing checks on local algo" begin

        listValidLocalOptimizers = [:NelderMead, :SimulatedAnnealing, :ParticleSwarm,
                            :BFGS, :LBFGS, :ConjugateGradient, :GradientDescent,
                            :MomentumGradientDescent, :AcceleratedGradientDescent]


        for localOptim in listValidLocalOptimizers

            @test is_local_optimizer(localOptim) == true

            # Symbol -> Optim algorithm
            @test typeof(convert_to_optim_algo(localOptim)) == typeof(getfield(Optim, localOptim)())

        end

        @test convert_to_fminbox(:LBFGS) isa Fminbox
        @test_throws ErrorException convert_to_optim_algo(:NotAnOptimizer)
        @test_throws ErrorException convert_to_fminbox(:NotAnOptimizer)

    end

    @testset "testing Optim" begin

        atolOptim = 0.5
        # Algorthims working with FminBox()
        # * GradientDescent
        # * BFGS
        # * LBFGS
        # * ConjugateGradient
        listValidLocalOptimizers = [:GradientDescent, :NelderMead, :SimulatedAnnealing,
                            :BFGS, :LBFGS, :ConjugateGradient,
                            :MomentumGradientDescent, :AcceleratedGradientDescent]

        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        x0 = [0.0, 0.0]
        lower = [-2.0; -2.0]
        upper = [2.0; 2.0]

        for localOptim in listValidLocalOptimizers


            results = optimize(f, x0, convert_to_optim_algo(localOptim), Optim.Options(iterations = 2000))

            # MSM.jl supports Optim 1 (1.13 or later) and Optim 2, and AcceleratedGradientDescent
            # behaves differently in the two:
            # * Optim 1: it converges on the Rosenbrock function, like the other algorithms
            #   (regular @test, in the else branch below).
            # * Optim 2: known upstream problem, it diverges on the Rosenbrock function (the
            #   objective increases with the number of iterations). The tests are marked as
            #   broken: @test_broken reports an "Unexpected Pass" once Optim fixes it.
            # A plain @test_broken for both versions would fail with Optim 1 ("Unexpected Pass").
            if localOptim == :AcceleratedGradientDescent && pkgversion(Optim) >= v"2"
                @test_broken Optim.minimizer(results)[1] ≈ 1.0 atol = atolOptim
                @test_broken Optim.minimizer(results)[2] ≈ 1.0 atol = atolOptim
            else
                @test Optim.minimizer(results)[1] ≈ 1.0 atol = atolOptim
                @test Optim.minimizer(results)[2] ≈ 1.0 atol = atolOptim
            end

        end


    end

    @testset "testing FminBox, non binding" begin

        atolOptim = 0.5
        # Algorthims working with FminBox()
        # * GradientDescent
        # * BFGS
        # * LBFGS
        # * ConjugateGradient
        listValidLocalOptimizers = [:GradientDescent, :BFGS, :LBFGS, :ConjugateGradient]

        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        x0 = [0.0, 0.0]
        lower = [-2.0; -2.0]
        upper = [2.0; 2.0]

        for localOptim in listValidLocalOptimizers

            results = optimize(f, lower, upper, x0, convert_to_fminbox(localOptim), Optim.Options(iterations = 2000))

            @test Optim.minimizer(results)[1] ≈ 1.0 atol = atolOptim
            @test Optim.minimizer(results)[2] ≈ 1.0 atol = atolOptim

        end

    end


    @testset "testing FminBox binding" begin

        atolOptim = 1e-1

        listValidLocalOptimizers = [:GradientDescent, :BFGS, :LBFGS, :ConjugateGradient]

        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        x0 = [0.0, 0.0]
        lower = [-2.0; -2.0]
        upper = [0.5; 0.5]

        for localOptim in listValidLocalOptimizers


            results = optimize(f, lower, upper, x0, convert_to_fminbox(localOptim), Optim.Options(iterations = 2000))

            @test Optim.minimizer(results)[1] < 0.5
            @test Optim.minimizer(results)[1] > -2.0
            @test Optim.minimizer(results)[2] < 0.5
            @test Optim.minimizer(results)[2] > -2.0

        end

    end



    @testset "testing loading priors and empirical moments" begin


       @testset "testing read_priors" begin

            dictPriors = read_priors(joinpath(@__DIR__, "priorsTest.csv"))

            @test typeof(dictPriors) == OrderedDict{String,Array{Float64,1}}
            # First component stores the value
            @test dictPriors["alpha"][1] == 0.5
            # Second component stores the lower bound:
            @test dictPriors["alpha"][2] == 0.01
            # Third component stores the upper bound:
            @test dictPriors["alpha"][3] == 0.9


       end

       @testset "testing read_empirical_moments" begin

            dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))

            @test typeof(dictEmpiricalMoments) == OrderedDict{String,Array{Float64,1}}

            # First component stores the value
            @test dictEmpiricalMoments["meanU"][1] == 0.05
            # Second component stores the weight associated to
            @test dictEmpiricalMoments["meanU"][2] == 0.05


       end


    end


    @testset "set_priors!, set_empirical_moments!, set_weight_matrix!" begin

        t = MSMProblem();

        @testset "testing set_priors!" begin

            dictPriors = read_priors(joinpath(@__DIR__, "priorsTest.csv"))

            set_priors!(t, dictPriors)

            @test t.priors == dictPriors

        end

        @testset "set_empirical_moments!" begin

            dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))

            set_empirical_moments!(t, dictEmpiricalMoments)

            @test t.empiricalMoments == dictEmpiricalMoments

        end

        @testset "set_weight_matrix!" begin

            N = 5
            W = collect(1:N).*ones(N,N)

            set_weight_matrix!(t, W)

            @test t.W == W

        end
    end


    @testset "testing the construction of the objective function" begin


       @testset "set_simulate_empirical_moments!" begin

            function functionTest(x::Vector)

                output = OrderedDict{String,Float64}()
                output["mom1"] = x[1]
                output["mom2"] = x[2]

                return output
            end

            t = MSMProblem();

            set_simulate_empirical_moments!(t, functionTest)

            x1Value = 1.0
            x2Value = 2.0
            simulatedMoments = t.simulate_empirical_moments([x1Value; x2Value])
            @test simulatedMoments["mom1"]  == x1Value
            @test simulatedMoments["mom2"]  == x2Value

       end

       @testset "Testing construct_objective_function!" begin

            Random.seed!(1234)
            tol1dMean = 0.01

            function functionTest(x::Vector)

                output = OrderedDict{String,Float64}()
                d = Normal(x[1])
                output["meanU"] = mean(rand(d, 100000))

                return output
            end

            t = MSMProblem();

            set_simulate_empirical_moments!(t, functionTest)

            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))
            set_empirical_moments!(t, dictEmpiricalMoments)

            # A. Set the function: parameter -> simulated moments
            set_simulate_empirical_moments!(t, functionTest)

            # A'. Attach weight marix
            W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
            #Special case: diagonal matrix
            #(you may choose something else)
            for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
            end

            set_weight_matrix!(t, W)


            # B. Construct the objective function, using the function: parameter -> simulated moments
            # and moments' weights:
            construct_objective_function!(t)

            # The objective function should be very close to 0 when
            # evaluated at the true value (modulo randomness)
            @test t.objective_function([dictEmpiricalMoments["meanU"][1]]) ≈ 0. atol = tol1dMean


        end


        @testset "Testing generate_bbSearchRange" begin

            # A.
            #----
            t = MSMProblem();

            dictPriors = read_priors(joinpath(@__DIR__, "priorsTest.csv"))

            set_priors!(t, dictPriors)

            testSearchRange = generate_bbSearchRange(t)

            @test testSearchRange[1][1] == 0.01
            @test testSearchRange[1][2] == 0.9
            @test testSearchRange[2][1] == 0.0
            @test testSearchRange[2][2] == 1.0

            # B.
            #---
            t = MSMProblem()

            dictPriors = OrderedDict{String,Array{Float64,1}}()
            dictPriors["mu1"] = [0., -5.0, 5.0]
            dictPriors["mu2"] = [0., -15.0, -10.0]
            dictPriors["mu3"] = [0., -20.0, -15.0]

            set_priors!(t, dictPriors)

            testSearchRange = generate_bbSearchRange(t)

            @test testSearchRange[1][1] == -5.0
            @test testSearchRange[1][2] == 5.0
            @test testSearchRange[2][1] == -15.0
            @test testSearchRange[2][2] == -10.0
            @test testSearchRange[3][1] == -20.0
            @test testSearchRange[3][2] == -15.0


        end


    end


    @testset "Testing msm_optimize!" begin


        # 1d problem
        #-----------
        @testset "Testing msm_optimize! with 1d" begin

            # Rermark:
            # It is important NOT to use Random.seed!()
            # within the function simulate_empirical_moments!
            # Otherwise, BlackBoxOptim does not find the solution
            #----------------------------------------------------
            tol1dMean = 0.2


            t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))

            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))
            set_empirical_moments!(t, dictEmpiricalMoments)


            dictPriors = OrderedDict{String,Array{Float64,1}}()
            dictPriors["mu1"] = [0., -2.0, 2.0]
            set_priors!(t, dictPriors)

            # A. Set the function: parameter -> simulated moments
            #----------------------------------------------------
            set_simulate_empirical_moments!(t, functionTest1d)

            # A'. Attach weight marix
            W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
            #Special case: diagonal matrix
            #(you may choose something else)
            for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
            end

            set_weight_matrix!(t, W)

            # B. Construct the objective function, using the function: parameter -> simulated moments
            # and moments' weights:
            #----------------------------------------------------
            construct_objective_function!(t)

            msm_optimize!(t, verbose = true)

            @test best_candidate(t.bbResults)[1] ≈ 0.05 atol = tol1dMean

            # C. Testing refinement of the global max using a local routine
            #--------------------------------------------------------------
            @test msm_refine_globalmin!(t, verbose = true)[1] ≈ 0.05 atol = tol1dMean

            @test msm_local_minimizer(t)[1] ≈ 0.05 atol = tol1dMean

        end

        @testset "Testing mutlistart 1d" begin


          tol1dMean = 0.1


          t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))

          # For the test to make sense, we need to set the field
          # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
          #------------------------------------------------------
          dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))
          set_empirical_moments!(t, dictEmpiricalMoments)


          dictPriors = OrderedDict{String,Array{Float64,1}}()
          dictPriors["mu1"] = [0., -2.0, 2.0]
          set_priors!(t, dictPriors)

          # A. Set the function: parameter -> simulated moments
          #----------------------------------------------------
          set_simulate_empirical_moments!(t, functionTest1dSeeded)

          # A'. Attach weight marix
          W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
          #Special case: diagonal matrix
          #(you may choose something else)
          for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
              W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
          end

          set_weight_matrix!(t, W)

          # B. Construct the objective function, using the function: parameter -> simulated moments
          # and moments' weights:
          #----------------------------------------------------
          construct_objective_function!(t)

          msm_multistart!(t, nums = maxNbWorkers, verbose = true)

          @test msm_local_minimum(t) ≈ 0.0 atol = tol1dMean

          @test msm_local_minimizer(t)[1] ≈ 0.05 atol = tol1dMean

        end

        @testset "Testing local to global 1d with FminBox" begin


          tol1dMean = 0.1

          # Let's try with the optim minBox on
          #-----------------------------------
          t = MSMProblem(options = MSMOptions(maxFuncEvals=1000, localOptimizer = :LBFGS, minBox = true))

          # For the test to make sense, we need to set the field
          # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
          #------------------------------------------------------
          dictEmpiricalMoments = read_empirical_moments(joinpath(@__DIR__, "empiricalMomentsTest.csv"))
          set_empirical_moments!(t, dictEmpiricalMoments)


          dictPriors = OrderedDict{String,Array{Float64,1}}()
          dictPriors["mu1"] = [0., -2.0, 2.0]
          set_priors!(t, dictPriors)

          # A. Set the function: parameter -> simulated moments
          #----------------------------------------------------
          set_simulate_empirical_moments!(t, functionTest1dSeeded)

          # A'. Attach weight marix
          W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
          #Special case: diagonal matrix
          #(you may choose something else)
          for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
              W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
          end

          set_weight_matrix!(t, W)

          # B. Construct the objective function, using the function: parameter -> simulated moments
          # and moments' weights:
          #----------------------------------------------------
          construct_objective_function!(t)

          msm_multistart!(t, nums = maxNbWorkers, verbose = true)

          @test msm_local_minimum(t) ≈ 0.0 atol = tol1dMean

          @test msm_local_minimizer(t)[1] ≈ 0.05 atol = tol1dMean

        end


        # 2d problem
        #-----------
        @testset "Testing smmoptimize with 2d and same magnitude" begin

            # Rermark:
            # It is important NOT to use Random.seed!()
            # within the function simulate_empirical_moments!
            # Otherwise, BlackBoxOptim does not find the solution
            #----------------------------------------------------
            tol2dMean = 0.2


            t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))


            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
            dictEmpiricalMoments["mean1"] = [1.0; 1.0]
            dictEmpiricalMoments["mean2"] = [-1.0; -1.0]
            set_empirical_moments!(t, dictEmpiricalMoments)


            dictPriors = OrderedDict{String,Array{Float64,1}}()
            dictPriors["mu1"] = [0., -5.0, 5.0]
            dictPriors["mu2"] = [0., -5.0, 5.0]
            set_priors!(t, dictPriors)

            # A. Set the function: parameter -> simulated moments
            set_simulate_empirical_moments!(t, functionTest2d)

            # A'. Attach weight marix
            W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
            #Special case: diagonal matrix
            #(you may choose something else)
            for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
            end

            set_weight_matrix!(t, W)

            # B. Construct the objective function, using the function: parameter -> simulated moments
            # and moments' weights:
            construct_objective_function!(t)

            # C. Run the optimization
            # This function first modifies t.bbSetup
            # and then modifies t.bbResults
            msm_optimize!(t, verbose = true)

            @test best_candidate(t.bbResults)[1] ≈ 1.0 atol = tol2dMean
            @test best_candidate(t.bbResults)[2] ≈ -1.0 atol = tol2dMean



        end


        # 2d problem
        #-----------
        @testset "Testing msm_multistart! with 2d and same magnitude" begin

            # Rermark:
            # It is important NOT to use Random.seed!()
            # within the function simulate_empirical_moments!
            # Otherwise, BlackBoxOptim does not find the solution
            #----------------------------------------------------
            tol2dMean = 0.2


            t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))


            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
            dictEmpiricalMoments["mean1"] = [1.0; 1.0]
            dictEmpiricalMoments["mean2"] = [-1.0; -1.0]
            set_empirical_moments!(t, dictEmpiricalMoments)


            dictPriors = OrderedDict{String,Array{Float64,1}}()
            dictPriors["mu1"] = [0., -5.0, 5.0]
            dictPriors["mu2"] = [0., -5.0, 5.0]
            set_priors!(t, dictPriors)

            # A. Set the function: parameter -> simulated moments
            set_simulate_empirical_moments!(t, functionTest2dSeeded)

            # A'. Attach weight marix
            W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
            #Special case: diagonal matrix
            #(you may choose something else)
            for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
            end

            set_weight_matrix!(t, W)

            # B. Construct the objective function, using the function: parameter -> simulated moments
            # and moments' weights:
            construct_objective_function!(t)

            msm_multistart!(t, nums = nworkers(), verbose = true)

            @test msm_local_minimum(t) ≈ 0.0 atol = tol2dMean

            @test msm_local_minimizer(t)[1] ≈ 1.0 atol = tol2dMean
            @test msm_local_minimizer(t)[2] ≈ - 1.0 atol = tol2dMean

          end

          @testset "Testing msm_multistart! with 2d, same magnitude and minBox = true" begin

              # Rermark:
              # It is important NOT to use Random.seed!()
              # within the function simulate_empirical_moments!
              # Otherwise, BlackBoxOptim does not find the solution
              #----------------------------------------------------
              tol2dMean = 0.2


              t = MSMProblem(options = MSMOptions(maxFuncEvals=1000, localOptimizer = :GradientDescent, minBox = true))


              # For the test to make sense, we need to set the field
              # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
              #------------------------------------------------------
              dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
              dictEmpiricalMoments["mean1"] = [1.0; 1.0]
              dictEmpiricalMoments["mean2"] = [-1.0; -1.0]
              set_empirical_moments!(t, dictEmpiricalMoments)


              dictPriors = OrderedDict{String,Array{Float64,1}}()
              dictPriors["mu1"] = [0., -5.0, 5.0]
              dictPriors["mu2"] = [0., -5.0, 5.0]
              set_priors!(t, dictPriors)

              # A. Set the function: parameter -> simulated moments
              set_simulate_empirical_moments!(t, functionTest2dSeeded)

              # A'. Attach weight marix
              W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
              #Special case: diagonal matrix
              #(you may choose something else)
              for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                  W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
              end

              set_weight_matrix!(t, W)

              # B. Construct the objective function, using the function: parameter -> simulated moments
              # and moments' weights:
              construct_objective_function!(t)

              msm_multistart!(t, nums = nworkers(), verbose = true)

              @test msm_local_minimum(t) ≈ 0.0 atol = tol2dMean

              @test msm_local_minimizer(t)[1] ≈ 1.0 atol = tol2dMean
              @test msm_local_minimizer(t)[2] ≈ - 1.0 atol = tol2dMean

            end

        # 2d problem
        #-----------
        @testset "Testing smmoptimize with 2d with a 1-order magnitude difference" begin

            # Rermark:
            # It is important NOT to use Random.seed!()
            # within the function simulate_empirical_moments!
            # Otherwise, BlackBoxOptim does not find the solution
            #----------------------------------------------------
            # The difference of magniture make it more difficult to find the minimum
            tol2dMean = 2.0

            function functionTest2d(x)

                d = MvNormal( [x[1]; x[2]], eye(2))
                output = OrderedDict{String,Float64}()

                draws = rand(d, 1000000)
                output["mean1"] = mean(draws[1,:])
                output["mean2"] = mean(draws[2,:])

                return output
            end


            t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))


            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
            dictEmpiricalMoments["mean1"] = [ 1.0; 1.0]
            dictEmpiricalMoments["mean2"] = [-12.0; 12.0]
            set_empirical_moments!(t, dictEmpiricalMoments)


            dictPriors = OrderedDict{String,Array{Float64,1}}()
            dictPriors["mu1"] = [0., -5.0, 5.0]
            dictPriors["mu2"] = [0., -15.0, -10.0]
            set_priors!(t, dictPriors)

            # A. Set the function: parameter -> simulated moments
            set_simulate_empirical_moments!(t, functionTest2d)

            # A'. Attach weight marix
            W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
            #Special case: diagonal matrix
            #(you may choose something else)
            for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
            end

            set_weight_matrix!(t, W)

            # B. Construct the objective function, using the function: parameter -> simulated moments
            # and moments' weights:
            construct_objective_function!(t)

            # C. Run the optimization
            # This function first modifies t.bbSetup
            # and then modifies t.bbResults
            msm_optimize!(t, verbose = true)

            @test best_candidate(t.bbResults)[1] ≈  1.0 atol = tol2dMean
            @test best_candidate(t.bbResults)[2] ≈ -12.0 atol = tol2dMean

        end


    end


    @testset "Testing minimizing a function that may fail" begin

              #---------------------------------------------------
              tol2dMean = 0.5
              @everywhere d_Uni = Uniform(0,1)
              @everywhere uniform_draws = rand(d_Uni, 10000)

              @everywhere function functionTest2d(x, uniform_draws::Array{Float64,1})

                  # function that fails when the first input
                  # is smaller than minus 1:
                  #------------------------------------------
                  if x[1] < -1.0
                    error("I failed")
                  end

                  d = Normal(x[1], 1.0)
                  # Inverse cdf (i.e. quantile)
                  draws = quantile.(d, uniform_draws)
                  output = OrderedDict{String,Float64}()
                  output["mean1"] = mean(draws)

                  return output
              end


              t = MSMProblem(options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes,
                            localOptimizer = :NelderMead, penaltyValue = 100.0, showDistance=true))

              #---------------------------------------------------------------------
              # Using multistart algo
              #---------------------------------------------------------------------
              # For the test to make sense, we need to set the field
              # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
              #------------------------------------------------------
              dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
              dictEmpiricalMoments["mean1"] = [1.0; 1.0]
              set_empirical_moments!(t, dictEmpiricalMoments)


              dictPriors = OrderedDict{String,Array{Float64,1}}()
              dictPriors["mu1"] = [0., -5.0, 5.0]
              set_priors!(t, dictPriors)

              # A. Set the function: parameter -> simulated moments
              set_simulate_empirical_moments!(t, x -> functionTest2d(x,uniform_draws))

              # A'. Attach weight marix
              W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
              #Special case: diagonal matrix
              #(you may choose something else)
              for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                  W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
              end

              set_weight_matrix!(t, W)

              # B. Construct the objective function, using the function: parameter -> simulated moments
              # and moments' weights:
              construct_objective_function!(t)

              msm_multistart!(t, nums = nworkers(), verbose = true)
              @test msm_local_minimum(t) ≈ 0.0 atol = tol2dMean
              @test msm_local_minimizer(t)[1] ≈ 1.0 atol = tol2dMean

              #---------------------------------------------------------------------
              # Using BlackBoxOptim
              #---------------------------------------------------------------------
              t = MSMProblem(options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes))

              dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
              dictEmpiricalMoments["mean1"] = [1.0; 1.0]
              set_empirical_moments!(t, dictEmpiricalMoments)


              dictPriors = OrderedDict{String,Array{Float64,1}}()
              dictPriors["mu1"] = [0., -5.0, 5.0]
              set_priors!(t, dictPriors)

              # A. Set the function: parameter -> simulated moments
              set_simulate_empirical_moments!(t, x -> functionTest2d(x, uniform_draws))

              # A'. Attach weight marix
              W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
              #Special case: diagonal matrix
              #(you may choose something else)
              for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
                  W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
              end

              set_weight_matrix!(t, W)

              # B. Construct the objective function, using the function: parameter -> simulated moments
              # and moments' weights:
              construct_objective_function!(t)

              msm_optimize!(t, verbose = true)

              @test best_candidate(t.bbResults)[1] ≈  1.0 atol = tol2dMean


    end

    @testset "Testing Var-Covariance estimation" begin

        d = Normal(0, 0.2)
        T=10000
        y1 = rand(d, T);
        y2 = rand(d, T);
        y3 = rand(d, T);
        data = hcat(y1, y2, y3);

        #When l=0, cov_NW(data) should be cov(data)
        @test maximum(abs.(cov_NW(data, l=0) .- cov(data))) < 1.0e-10

        # Even when including lags, with no serial correlation the difference should
        # be small
        @test maximum(abs.(cov_NW(data, l=1) .- cov(data))) < 1.0e-2


    end

    @testset "Testing J_test" begin

        # Correctly specified model: two independent N(theta,1) series, one parameter.
        # k = 2 moments, l = 1 parameter, so J should be Chi²(1) under the null.
        # Sigma0 = I, so W = I is the efficient weighting matrix, and the SMM
        # estimate has a closed form (average of the two moment gaps).
        tData = 200
        tSimData = 200 #tau = 1: the old formula T*(1+tau)*g'Wg rejected ~33% of the time
        alpha = 0.05
        nbReps = 4000
        rng = MersenneTwister(1234)

        myProblem = MSMProblem()
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(2)))

        nbRejections = 0
        for r = 1:nbReps

            x = randn(rng, tData)
            y = randn(rng, tData)
            e1 = randn(rng, tSimData) #simulation draws held fixed
            e2 = randn(rng, tSimData)

            dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
            dictEmpiricalMoments["mean_x"] = [mean(x), 1.0]
            dictEmpiricalMoments["mean_y"] = [mean(y), 1.0]
            set_empirical_moments!(myProblem, dictEmpiricalMoments)

            set_simulate_empirical_moments!(myProblem, theta -> OrderedDict{String,Float64}("mean_x" => theta[1] + mean(e1), "mean_y" => theta[1] + mean(e2)))

            thetaHat = [((mean(x) - mean(e1)) + (mean(y) - mean(e2)))/2]

            J, c = J_test(myProblem, thetaHat, tData, tSimData, alpha)

            # Value of the statistic and critical value
            if r == 1
                g = [mean(x) - mean(e1) - thetaHat[1], mean(y) - mean(e2) - thetaHat[1]]
                @test J ≈ tData/(1.0 + tData/tSimData)*sum(g.^2)
                @test c ≈ 3.841458820694124 #95% quantile of Chi²(1)
            end

            nbRejections += (J > c)

        end

        # Empirical size close to the nominal size (standard error ≈ 0.0034)
        @test 0.03 < nbRejections/nbReps < 0.07

    end

    @testset "Testing p-values and confidence intervals" begin

        # Known asymptotic variance: se = sqrt(Avar[i,i]/tData) = [0.2, 0.3]
        # theta0 = [0.2, -0.3] gives t-statistics of +1 and -1
        myProblem = MSMProblem()
        myProblem.Avar = [4.0 0.0; 0.0 9.0]
        theta0 = [0.2, -0.3]
        tData = 100
        alpha = 0.05

        @test calculate_se(myProblem, tData, 1) ≈ 0.2
        @test calculate_se(myProblem, tData, 2) ≈ 0.3
        @test calculate_t(myProblem, theta0, tData, 1) ≈ 1.0
        @test calculate_t(myProblem, theta0, tData, 2) ≈ -1.0

        # Two-sided p-value for |t| = 1 is 0.3173..., whatever the sign of t
        @test calculate_pvalue(myProblem, theta0, tData, 1) ≈ 0.31731050786291415
        @test calculate_pvalue(myProblem, theta0, tData, 2) ≈ 0.31731050786291415

        # 95% confidence interval uses the 97.5% quantile of N(0,1): 1.959963...
        z = 1.959963984540054
        CI_lower, CI_upper = calculate_CI(myProblem, theta0, tData, 1, alpha)
        @test CI_lower ≈ 0.2 - 0.2*z
        @test CI_upper ≈ 0.2 + 0.2*z
        CI_lower, CI_upper = calculate_CI(myProblem, theta0, tData, 2, alpha)
        @test CI_lower ≈ -0.3 - 0.3*z
        @test CI_upper ≈ -0.3 + 0.3*z

        # summary_table reports the same values
        df = DataFrame(summary_table(myProblem, theta0, tData, alpha))
        @test df[:, "Pr(>|t|)"] ≈ [0.31731050786291415, 0.31731050786291415]
        @test df[:, "CI Lower"] ≈ theta0 .- [0.2, 0.3] .* z
        @test df[:, "CI Upper"] ≈ theta0 .+ [0.2, 0.3] .* z

    end

    @testset "Testing starting values and multistart with several workers" begin

        # Regression test: results must be matched to the right points even when
        # workers finish in a different order than the one in which they were started.
        # The first worker is made slow, so that it finishes last.
        newWorkers = addprocs(2)
        @everywhere newWorkers begin
            using Distributed
            using MSM
            using OrderedCollections
        end
        slowWorker = first(workers())

        myProblem = MSMProblem(options = MSMOptions(localOptimizer = :LBFGS, thresholdStartingValue = 0.25))
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m" => [0.0, 1.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(1)))
        # distance = x^2
        set_simulate_empirical_moments!(myProblem, x -> (myid() == slowWorker && sleep(0.05); OrderedDict{String,Float64}("m" => x[1])))
        construct_objective_function!(myProblem)

        # Starting values: only x < 0.5 is valid (x^2 < 0.25), sorted by distance
        Random.seed!(1234)
        Validx0 = MSM.search_starting_values(myProblem, 4, verbose = false)
        @test size(Validx0) == (4, 1)
        @test all(Validx0[:, 1] .< 0.5)
        @test issorted(Validx0[:, 1])

        # Multistart: listOptimResults[i] must start from x0[i,:]
        x0 = reshape(collect(range(0.9, 0.1, length = nworkers())), :, 1)
        # All local minimizations converge, and they must all be counted
        listOptimResults = @test_logs (:info, "Convergence reached for $(nworkers()) worker(s).") match_mode=:any msm_multistart!(myProblem, x0 = x0, nums = nworkers(), verbose = false)
        for workerIndex = 1:nworkers()
            @test listOptimResults[workerIndex].initial_x == x0[workerIndex, :]
        end
        @test msm_multistart_minimizer(myProblem)[1] ≈ 0.0 atol = 1e-4

        # Fewer starting values than workers: explicit error
        @test_throws ErrorException msm_multistart!(myProblem, x0 = x0, nums = nworkers() - 1, verbose = false)

        rmprocs(newWorkers)

    end

    @testset "Testing robustness of the objective function" begin

        myProblem = MSMProblem()
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m" => [0.0, 1.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(1)))
        penaltyValue = myProblem.options.penaltyValue

        # A simulated moment is missing: penalty value (instead of a KeyError)
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("wrong_name" => x[1]))
        construct_objective_function!(myProblem)
        @test myProblem.objective_function([0.3]) == penaltyValue

        # NaN or Inf simulated moments: penalty value
        for badValue in (NaN, Inf, -Inf)
            set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m" => badValue))
            construct_objective_function!(myProblem)
            @test myProblem.objective_function([0.3]) == penaltyValue
        end

        # msm_slices must work with a function that requires a Vector{Float64}
        set_simulate_empirical_moments!(myProblem, (x::Vector{Float64}) -> OrderedDict{String,Float64}("m" => x[1]))
        construct_objective_function!(myProblem)
        vXGrid, vYGrid = msm_slices(myProblem, [0.3], nbPoints = 5)
        @test vYGrid ≈ vXGrid.^2

        # Invalid gridType: explicit error
        myProblem.options.gridType = :notAGrid
        @test_throws ErrorException MSM.search_starting_values(myProblem, 1, verbose = false)

    end

    @testset "Testing lambda (NES optimizers) and populationSize" begin

        # Options: lambda >= 0, even for dxnes
        @test_throws "lambda must be >= 0" MSMOptions(lambda = -2)
        @test_throws "even lambda" MSMOptions(lambda = 7)
        @test MSMOptions(globalOptimizer = :xnes, lambda = 7).lambda == 7

        # populationSize: warning with NES optimizers only, when set by the user
        @test_logs MSMOptions()
        @test_logs (:warn, r"populationSize is ignored") MSMOptions(populationSize = 20)
        @test_logs MSMOptions(populationSize = 20, globalOptimizer = :adaptive_de_rand_1_bin_radiuslimited)
        @test MSMOptions(populationSize = 20, globalOptimizer = :adaptive_de_rand_1_bin_radiuslimited).populationSize == 20

        # Automatic lambda: BlackBoxOptim's default rounded up to a multiple of the number of workers
        # (dxnes: even, so one worker idle when the number of workers is odd)
        @test nes_lambda(:dxnes, 0, 5, 1) == 8
        @test nes_lambda(:dxnes, 0, 5, 10) == 10
        @test nes_lambda(:dxnes, 0, 5, 32) == 32
        @test nes_lambda(:dxnes, 0, 5, 11) == 10
        @test nes_lambda(:dxnes, 0, 5, 3) == 8
        @test nes_lambda(:dxnes, 0, 10, 4) == 12
        @test nes_lambda(:xnes, 0, 5, 11) == 11
        @test nes_lambda(:dxnes, 12, 5, 32) == 12
        @test_throws "even lambda" nes_lambda(:dxnes, 7, 5, 1)
        @test_throws "not a natural evolution strategy" nes_lambda(:adaptive_de_rand_1_bin_radiuslimited, 0, 5, 1)

        # With one worker, the automatic lambda is BlackBoxOptim's own default
        for method in (:dxnes, :xnes, :separable_nes), d in (1, 2, 5, 10, 30)
            bbDefault = bbsetup(x -> sum(abs2, x); Method = method, SearchRange = (-1.0, 1.0), NumDimensions = d, TraceMode = :silent)
            @test nes_lambda(method, 0, d, 1) == bbDefault.optimizer.lambda
        end

        # lambda is passed to BlackBoxOptim, and logged with verbose = true
        myProblem = MSMProblem(options = MSMOptions(maxFuncEvals = 40))
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x$(i)" => [0.5, 0.0, 1.0] for i in 1:5))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m$(i)" => [0.0] for i in 1:5))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(5)))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m$(i)" => x[i] for i in 1:5))
        construct_objective_function!(myProblem)
        expectedLambda = nes_lambda(:dxnes, 0, 5, nworkers())
        @test_logs (:info, r"dxnes: λ = \d+ points per generation") match_mode=:any set_bbSetup!(myProblem, verbose = true)
        @test myProblem.bbSetup.optimizer.lambda == expectedLambda
        myProblem.options.lambda = 12
        msm_optimize!(myProblem, verbose = false)
        @test myProblem.bbSetup.optimizer.lambda == 12

    end

    @testset "Testing small utilities" begin

        # get_now: "yyyy-mm-dd--HHh-MMm-SSs", a valid date and time, close to now
        stamp = get_now()
        m = match(r"^(\d{4}-\d{2}-\d{2})--(\d{1,2})h-(\d{1,2})m-(\d{1,2})s$", stamp)
        @test m !== nothing
        t = MSM.Dates.DateTime(MSM.Dates.Date(m[1]), MSM.Dates.Time(parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4])))
        @test abs(MSM.Dates.now() - t) < MSM.Dates.Second(5)

        # linspace(z_start, z_end, z_n)
        @test linspace(0.0, 1.0, 5) == [0.0, 0.25, 0.5, 0.75, 1.0]

        # latin_hypercube_sampling(mins, maxs, n): n×dims matrix
        @test size(latin_hypercube_sampling([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 5)) == (5, 3)

    end

    @testset "Testing check_problem" begin

        # A problem set up correctly: no error, no log message
        myProblem = MSMProblem(options = MSMOptions(maxFuncEvals = 20))
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.5, 0.0, 1.0], "y" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m1" => [0.0], "m2" => [0.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(2)))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => x[1], "m2" => x[2]))
        construct_objective_function!(myProblem)
        @test (@test_logs check_problem(myProblem)) === nothing

        # Fields that are not set
        @test_throws "set_priors!" check_problem(MSMProblem())
        emptyProblem = MSMProblem(priors = myProblem.priors)
        @test_throws "set_empirical_moments!" check_problem(emptyProblem)
        set_empirical_moments!(emptyProblem, myProblem.empiricalMoments)
        set_weight_matrix!(emptyProblem, myProblem.W)
        @test_throws "set_simulate_empirical_moments!" check_problem(emptyProblem)
        set_simulate_empirical_moments!(emptyProblem, myProblem.simulate_empirical_moments)
        @test_throws "construct_objective_function!" check_problem(emptyProblem)

        # Default 1×1 weight matrix with 2 moments: error, also in msm_optimize! and msm_multistart!
        myProblem.W = Matrix(1.0 .* I(1))
        @test_throws "W is 1×1, but there are 2 empirical moments" check_problem(myProblem)
        @test_throws "W is 1×1" msm_optimize!(myProblem, verbose = false)
        @test_throws "W is 1×1" msm_multistart!(myProblem, verbose = false)
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(2)))

        # A missing moment: error
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => x[1], "wrong_name" => x[2]))
        construct_objective_function!(myProblem)
        @test_throws "does not return the empirical moment(s) [\"m2\"]" check_problem(myProblem)

        # An extra moment: warning only
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => x[1], "extra" => 1.0, "m2" => x[2]))
        construct_objective_function!(myProblem)
        @test_logs (:warn, r"extra") check_problem(myProblem)

        # Non-finite simulated moments at the initial values: error
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => NaN, "m2" => x[2]))
        construct_objective_function!(myProblem)
        @test_throws "Non-finite simulated moment(s) [\"m1\"]" check_problem(myProblem)

        # The simulation throws at the initial values: error (with the original error logged),
        # which says how to skip the check. With check = false, the optimization runs
        set_simulate_empirical_moments!(myProblem, x -> error("typo in the simulation function"))
        construct_objective_function!(myProblem)
        @test_logs (:error, "The simulation of moments failed at the initial values of the priors") @test_throws "check = false" check_problem(myProblem)
        @test_logs (:error, r"failed at the initial values") @test_throws "check = false" msm_optimize!(myProblem, verbose = false)
        @test_logs (:info, r"An error occurred") match_mode=:any msm_optimize!(myProblem, verbose = false, check = false)
        @test msm_minimum(myProblem) == Inf

        # With a finite penalty value, a distance larger than the penalty value: warning
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => x[1], "m2" => x[2]))
        construct_objective_function!(myProblem)
        myProblem.options.penaltyValue = 0.1
        @test_logs (:warn, r"larger than penaltyValue") check_problem(myProblem)

    end

    @testset "Testing the order of simulate_empirical_moments_array" begin

        # The array follows the order of the empirical moments (the order of W),
        # whatever the order of the moments returned by the simulation function
        myProblem = MSMProblem()
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m1" => [0.0], "m2" => [0.0]))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m2" => 2.0*x[1], "extra" => 99.0, "m1" => x[1]))
        @test myProblem.simulate_empirical_moments_array([1.0]) == [1.0, 2.0]
        # Jacobian: one row per empirical moment, in the same order (d m1/dx = 1, d m2/dx = 2)
        @test calculate_D(myProblem, [1.0]) ≈ [1.0; 2.0;;]

        # Without empirical moments: the order of the simulated moments
        otherProblem = MSMProblem()
        set_simulate_empirical_moments!(otherProblem, myProblem.simulate_empirical_moments)
        @test otherProblem.simulate_empirical_moments_array([1.0]) == [2.0, 99.0, 1.0]

    end

    @testset "Testing the evaluation budget of local minimizations" begin

        # Rosenbrock function (distance = (1 - x)^2 + 100 (y - x^2)^2), started far from its minimum [1, 1]
        nbCalls = Ref(0)
        myProblem = MSMProblem(options = MSMOptions(maxFuncEvals = 50))
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [-1.2, -5.0, 5.0], "y" => [1.0, -5.0, 5.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m1" => [0.0], "m2" => [0.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(2)))
        set_simulate_empirical_moments!(myProblem, x -> (nbCalls[] += 1; OrderedDict{String,Float64}("m1" => 1.0 - x[1], "m2" => 10.0*(x[2] - x[1]^2))))
        construct_objective_function!(myProblem)

        # maxFuncEvals counts the evaluations of finite-difference gradients too:
        # stop at the end of the iteration during which the budget is reached (an LBFGS iteration
        # with a long line search can take ~25 evaluations here; before the fix, maxFuncEvals was the
        # number of iterations, i.e. several hundred evaluations)
        for (localOptimizer, minBox) in [(:LBFGS, false), (:NelderMead, false), (:LBFGS, true)]
            myProblem.options.localOptimizer = localOptimizer
            myProblem.options.minBox = minBox
            nbCalls[] = 0
            optimResults = msm_localmin(myProblem, [-1.2, 1.0], verbose = false)
            @test 50 <= nbCalls[] <= 100
            @test Optim.converged(optimResults) == false
        end

        # A large enough budget: convergence
        myProblem.options.localOptimizer = :LBFGS
        myProblem.options.minBox = false
        myProblem.options.maxFuncEvals = 10000
        optimResults = msm_localmin(myProblem, [-1.2, 1.0], verbose = false)
        @test Optim.converged(optimResults) == true
        @test Optim.minimizer(optimResults) ≈ [1.0, 1.0] atol = 1e-3

        # msm_multistart!: if no local minimization converged, the best finite local minimum is used
        myProblem.options.maxFuncEvals = 50
        x0 = repeat([-1.2 1.0], nworkers())
        @test_logs (:info, r"None of the local minimizations converged") match_mode=:any msm_multistart!(myProblem, x0 = x0, verbose = false)
        @test msm_multistart_minimum(myProblem) < myProblem.objective_function([-1.2, 1.0])
        @test Optim.converged(myProblem.optimResults) == false

    end

    @testset "Testing msm_multistart! with failed local minimizations" begin

        myProblem = MSMProblem()
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m" => [0.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(1)))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m" => x[1]))
        construct_objective_function!(myProblem)
        x0 = fill(0.5, nworkers(), 1)

        # verbose is passed to the local minimizations
        logs, _ = Test.collect_test_logs(() -> msm_multistart!(myProblem, x0 = x0, verbose = false))
        @test any(occursin("Starting value", string(log.message)) for log in logs) == false
        logs, _ = Test.collect_test_logs(() -> msm_multistart!(myProblem, x0 = x0, verbose = true))
        @test any(occursin("Starting value", string(log.message)) for log in logs) == true

        # A local minimization that throws: one clear message, result nothing, no MethodError
        myProblem.objective_function = x -> error("boom")
        logs, listOptimResults = Test.collect_test_logs(() -> msm_multistart!(myProblem, x0 = x0, verbose = false, check = false))
        messages = [string(log.message) for log in logs]
        @test all(listOptimResults .=== nothing)
        @test any(occursin("failed: ", m) for m in messages)
        @test any(occursin("MethodError", m) for m in messages) == false
        @test "None of the local minimizations returned a finite value." in messages

    end

    @testset "Testing msm_slices at a parameter equal to 0" begin

        myProblem = MSMProblem()
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.0, -1.0, 1.0], "y" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m1" => [0.0], "m2" => [0.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(2)))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m1" => x[1], "m2" => x[2]))
        construct_objective_function!(myProblem)

        vXGrid, vYGrid = msm_slices(myProblem, [0.0, 0.5], nbPoints = 5, offset = 0.001)
        # x = 0: the slice uses the width of the prior (2), y = 0.5: percent deviation
        @test vXGrid[:, 1] ≈ collect(range(-0.002, 0.002, length = 5))
        @test vXGrid[:, 2] ≈ collect(range(0.4995, 0.5005, length = 5))
        @test vYGrid[:, 1] ≈ vXGrid[:, 1].^2 .+ 0.25

    end

    @testset "Testing the verbose option of msm_optimize!" begin

        myProblem = MSMProblem(options = MSMOptions(maxFuncEvals = 50))
        set_priors!(myProblem, OrderedDict{String,Array{Float64,1}}("x" => [0.5, 0.0, 1.0]))
        set_empirical_moments!(myProblem, OrderedDict{String,Array{Float64,1}}("m" => [0.0, 1.0]))
        set_weight_matrix!(myProblem, Matrix(1.0 .* I(1)))
        set_simulate_empirical_moments!(myProblem, x -> OrderedDict{String,Float64}("m" => x[1]))
        construct_objective_function!(myProblem)

        # Returns what f() prints to stdout
        function captured_stdout(f)
            mktemp() do path, io
                redirect_stdout(f, io)
                flush(io)
                read(path, String)
            end
        end

        # verbose = true: BlackBoxOptim displays its progress
        output = captured_stdout(() -> msm_optimize!(myProblem, verbose = true))
        @test occursin("Starting optimization", output)

        # verbose = false: nothing printed, no log message
        output = captured_stdout(() -> (@test_logs msm_optimize!(myProblem, verbose = false)))
        @test output == ""
        @test myProblem.bbResults !== nothing

    end

    @testset "Testing Inference" begin

      # Inference in the linear model
      #------------------------------
      Random.seed!(1234)         #for replicability reasons
      T = 100000          #number of periods
      P = 2               #number of dependent variables
      beta0 = [2.0; 3.0]     #choose true coefficients by drawing from a uniform distribution on [0,1]
      alpha0 = 1.0  #intercept
      theta0 = 0.0        #coefficient to create serial correlation in the error terms
      println("True intercept = $(alpha0)")
      println("True coefficient beta0 = $(beta0)")
      println("Serial correlation coefficient theta0 = $(theta0)")

      # Simulation of a sample:
      # Generation of error terms
      #--------------------------
      # row = individual dimension
      # column = time dimension
      U = zeros(T)
      d = Normal()
      U[1] = rand(d, 1)[] #first error term
      # loop over time periods
      for t = 2:T
          U[t] = rand(d, 1)[] + theta0*U[t-1]
      end
      # Let's simulate x_t
      #-------------------
      x = zeros(T, P)

      d = Uniform(0, 5)
      for p = 1:P
              x[:,p] = rand(d, T)
      end

      # Let's calculate the resulting y_t
      y = zeros(T)

      for t=1:T
          y[t] = alpha0 + x[t,1]*beta0[1] + x[t,2]*beta0[2] + U[t]
      end

      myProblem = MSMProblem(options = MSMOptions(maxFuncEvals=1000, globalOptimizer = :dxnes, localOptimizer = :LBFGS));

      # Empirical moments
      #------------------
      dictEmpiricalMoments = OrderedDict{String,Array{Float64,1}}()
      dictEmpiricalMoments["mean"] = [mean(y); mean(y)] #informative on the intercept
      dictEmpiricalMoments["mean_x1y"] = [mean(x[:,1] .* y); mean(x[:,1] .* y)] #informative on betas
      dictEmpiricalMoments["mean_x2y"] = [mean(x[:,2] .* y); mean(x[:,2] .* y)] #informative on betas
      dictEmpiricalMoments["mean_x1y^2"] = [mean((x[:,1] .* y).^2); mean((x[:,1] .* y).^2)] #informative on betas
      dictEmpiricalMoments["mean_x2y^2"] = [mean((x[:,2] .* y).^2); mean((x[:,2] .* y).^2)] #informative on betas

      set_empirical_moments!(myProblem, dictEmpiricalMoments)

      dictPriors = OrderedDict{String,Array{Float64,1}}()
      dictPriors["alpha"] = [0.5, 0.001, 1.0]
      dictPriors["beta1"] = [0.5, 0.001, 1.0]
      dictPriors["beta2"] = [0.5, 0.001, 1.0]

      set_priors!(myProblem, dictPriors)

      # A'. Attach weight marix
      W = Matrix(1.0 .* I(length(dictEmpiricalMoments)))#initialization
      #Special case: diagonal matrix
      #(you may choose something else)
      for (indexMoment, k) in enumerate(keys(dictEmpiricalMoments))
          W[indexMoment,indexMoment] = 1.0/(dictEmpiricalMoments[k][1])^2
      end

      set_weight_matrix!(myProblem, W)


      # x[1] corresponds to the intercept
      # x[1] corresponds to beta1
      # x[3] corresponds to beta2
      @everywhere function functionLinearModel(x; uniform_draws::Array{Float64,1}, simX::Array{Float64,2}, nbDraws::Int64 = length(uniform_draws), burnInPerc::Int64 = 10)


          T = nbDraws
          P = 2       #number of dependent variables

          alpha = x[1]
          beta = x[2:end]
          theta = 0.0     #coefficient to create serial correlation in the error terms


          # Creation of error terms
          # row = individual dimension
          # column = time dimension
          U = zeros(T)
          d = Normal()
          # Inverse cdf (i.e. quantile)
          gaussian_draws = quantile.(d, uniform_draws)
          U[1] = gaussian_draws[1] #first error term

          # loop over time periods
          for t = 2:T
              U[t] = gaussian_draws[t] + theta*U[t-1]
          end

          # Let's calculate the resulting y_t
          y = zeros(T)

          for t=1:T
              y[t] = alpha + simX[t,1]*beta[1] + simX[t,2]*beta[2] + U[t]
          end

          # Get rid of the burn-in phase:
          #------------------------------
          startT = div(nbDraws, burnInPerc)

          # Moments:
          #---------
          output = OrderedDict{String,Float64}()
          output["mean"] = mean(y[startT:nbDraws])
          output["mean_x1y"] = mean(simX[startT:nbDraws,1] .* y[startT:nbDraws])
          output["mean_x2y"] = mean(simX[startT:nbDraws,2] .* y[startT:nbDraws])
          output["mean_x1y^2"] = mean((simX[startT:nbDraws,1] .* y[startT:nbDraws]).^2)
          output["mean_x2y^2"] = mean((simX[startT:nbDraws,2] .* y[startT:nbDraws]).^2)

          return output
      end


      # Let's freeze the randomness during the minimization
      @everywhere d_Uni = Uniform(0,1)
      @everywhere nbDraws = 100000 #number of draws in the simulated data
      @everywhere uniform_draws = rand(d_Uni, nbDraws)
      simX = zeros(length(uniform_draws), 2)
      d = Uniform(0, 5)
      for p = 1:2
          simX[:,p] = rand(d, length(uniform_draws))
      end

      set_simulate_empirical_moments!(myProblem, x -> functionLinearModel(x, uniform_draws = uniform_draws, simX = simX))

      # Construct the objective function using:
      #* the function: parameter -> simulated moments
      #* emprical moments values
      #* emprical moments weights
      construct_objective_function!(myProblem)

      # Run the optimization in parallel using n different starting values
      # where n is equal to the number of available workers
      #--------------------------------------------------------------------
      listOptimResults = msm_multistart!(myProblem, verbose = true)

      # Remark: it would not be appropriate to use BlackBoxOptim because the
      # No big deal here, because we use Optim
      minimizer = msm_local_minimizer(myProblem)

      # Empirical Distance matrix
      #--------------------------
      X = zeros(T, 5)

      X[:,1] = y
      X[:,2] = (x[:,1] .* y)
      X[:,3] = (x[:,2] .* y)
      X[:,4] = (x[:,1] .* y).^2
      X[:,5] = (x[:,2] .* y).^2

      Sigma0 = cov(X)

      set_Sigma0!(myProblem, Sigma0)

      @test myProblem.Sigma0 == Sigma0

      calculate_Avar!(myProblem, minimizer, tau = T/nbDraws)

      # The asymptotic variance should be
      # * symmetric
      # * positive semi-definite
      #-----------------------------------
      @test issymmetric(myProblem.Avar) == true

      # test for positive semi-definiteness
      # all eigenvalues should be non-negative
      eigv = eigvals(myProblem.Avar)

      counterNegativeEigVals = 0
      for i=1:length(eigv)
        if eigv[i] < 0
          counterNegativeEigVals += 1
        end
      end

      @test counterNegativeEigVals == 0

      # summary table:
      #---------------
      df = summary_table(myProblem, minimizer, T, 0.05)
      @test typeof(df) == CoefTable

      # Convert coeftable to a DataFrame
      df = DataFrame(df)

      # first column : point estimates
      @test df[:, "Coef."] == minimizer

      # 2nd column : std error
      for i =1:size(df,1)
        @test df[i, "Std. Error"] > 0.
      end

      # The minimizer should not be too far from the true values:
      # within 4 standard errors (a fixed tolerance can be smaller than
      # the sampling noise, and then depends on the random sample)
      #---------------------------------------------------------
      trueValues = [alpha0; beta0]
      for i = 1:length(trueValues)
        @test abs(minimizer[i] - trueValues[i]) < 4*calculate_se(myProblem, T, i)
      end

      # confidence interval
      for i =1:size(df,1)
        @test df[i, "CI Lower"] <= df[i, "CI Upper"]
      end

      D = calculate_D(myProblem, minimizer)
      @test rank(D) == size(df,1)

      # Slice the objective functions
      nb_p = 7
      vXGrid, vYGrid = msm_slices(myProblem, minimizer, nbPoints = nb_p)

      @test size(vXGrid,1) == nb_p
      @test size(vXGrid,2) == size(df,1)

      @test size(vYGrid,1) == nb_p
      @test size(vYGrid,2) == size(df,1)

      # Plot the results
      if do_plots == true

          gr()
          list_plots = []
          for (keyIndex, keyValue) in enumerate(keys(myProblem.priors))

              p = plot(vXGrid[:,keyIndex], vYGrid[:,keyIndex], title = "$(keyValue)", label = "")
              plot!([minimizer[keyIndex]], seriestype = :vline, label = "")
              push!(list_plots, p)

          end

          #Let's combine all the plots in a single plot
          s0 = ""
          for i = 1:length(keys(myProblem.priors))
              if i==1
                  s0 = string("list_plots[$(i)]" )
              else
                  s0 = string(s0, ", ", "list_plots[$(i)]" )
              end
          end

          plot_combined = eval(Meta.parse(string("plot(", s0, ")")))
          display(plot_combined)

      end


    end



end
