maxNbWorkers = 1
using Distributed
while nworkers() < maxNbWorkers
  addprocs(maxNbWorkers - nworkers())
end
println(nworkers())

@everywhere using MSM
@everywhere using DataStructures
@everywhere using OrderedCollections
@everywhere using Random
@everywhere using Distributions
@everywhere using Statistics
@everywhere using LinearAlgebra
using Test
using Pkg
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
            @test t.penaltyValue == 999999.0
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

        end

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

            @test Optim.minimizer(results)[1] ≈ 1.0 atol = atolOptim
            @test Optim.minimizer(results)[2] ≈ 1.0 atol = atolOptim

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

            dictPriors = read_priors(joinpath(Pkg.dir("MSM"), "test/priorsTest.csv"))

            @test typeof(dictPriors) == OrderedDict{String,Array{Float64,1}}
            # First component stores the value
            @test dictPriors["alpha"][1] == 0.5
            # Second component stores the lower bound:
            @test dictPriors["alpha"][2] == 0.01
            # Third component stores the upper bound:
            @test dictPriors["alpha"][3] == 0.9


       end

       @testset "testing read_empirical_moments" begin

            dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))

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

            dictPriors = read_priors(joinpath(Pkg.dir("MSM"), "test/priorsTest.csv"))

            set_priors!(t, dictPriors)

            @test t.priors == dictPriors

        end

        @testset "set_empirical_moments!" begin

            dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))

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
            dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))
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

            dictPriors = read_priors(joinpath(Pkg.dir("MSM"), "test/priorsTest.csv"))

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

            @everywhere function functionTest1d(x)

                d = Normal(x[1])
                output = OrderedDict{String,Float64}()

                output["meanU"] = mean(rand(d, 1000000))

                return output
            end


            t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))

            # For the test to make sense, we need to set the field
            # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
            #------------------------------------------------------
            dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))
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

          @everywhere function functionTest1d(x)

              # When using one of the deterministic methods of Optim,
              # we can safely "control" for randomness
              #-----------------------------------------------------
              Random.seed!(1234)
              d = Normal(x[1])
              output = OrderedDict{String,Float64}()

              output["meanU"] = mean(rand(d, 1000000))

              return output
          end


          t = MSMProblem(options = MSMOptions(maxFuncEvals=1000))

          # For the test to make sense, we need to set the field
          # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
          #------------------------------------------------------
          dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))
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

          msm_multistart!(t, nums = maxNbWorkers, verbose = true)

          @test msm_local_minimum(t) ≈ 0.0 atol = tol1dMean

          @test msm_local_minimizer(t)[1] ≈ 0.05 atol = tol1dMean

        end

        @testset "Testing local to global 1d with FminBox" begin


          tol1dMean = 0.1

          @everywhere function functionTest1d(x)

              # When using one of the deterministic methods of Optim,
              # we can safely "control" for randomness
              #-----------------------------------------------------
              Random.seed!(1234)
              d = Normal(x[1])
              output = OrderedDict{String,Float64}()

              output["meanU"] = mean(rand(d, 1000000))

              return output
          end

          # Let's try with the optim minBox on
          #-----------------------------------
          t = MSMProblem(options = MSMOptions(maxFuncEvals=1000, localOptimizer = :LBFGS, minBox = true))

          # For the test to make sense, we need to set the field
          # t.empiricalMoments::OrderedDict{String,Array{Float64,1}}
          #------------------------------------------------------
          dictEmpiricalMoments = read_empirical_moments(joinpath(Pkg.dir("MSM"), "test/empiricalMomentsTest.csv"))
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

            @everywhere function functionTest2d(x)

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

            @everywhere function functionTest2d(x)

                d = MvNormal( [x[1]; x[2]], eye(2))
                output = OrderedDict{String,Float64}()

                # When using one of the deterministic methods of Optim,
                # we can safely "control" for randomness
                #-----------------------------------------------------
                Random.seed!(1234)
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

              @everywhere function functionTest2d(x)

                  d = MvNormal( [x[1]; x[2]], eye(2))
                  output = OrderedDict{String,Float64}()

                  # When using one of the deterministic methods of Optim,
                  # we can safely "control" for randomness
                  #-----------------------------------------------------
                  Random.seed!(1234)
                  draws = rand(d, 1000000)
                  output["mean1"] = mean(draws[1,:])
                  output["mean2"] = mean(draws[2,:])

                  return output
              end


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

    @testset "Testing Inference" begin

      tolLinear = 0.05

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

      # The minimizer should not be too far from the true values
      #---------------------------------------------------------
      @test minimizer[1] ≈  alpha0[1] atol = tolLinear
      @test minimizer[2] ≈  beta0[1] atol = tolLinear
      @test minimizer[3] ≈  beta0[2] atol = tolLinear

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
