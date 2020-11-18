"""
  function calculate_D(sMMProblem::MSMProblem, theta0::Array{Float64,1}; method::Symbol = :central)

Function to calculate the jacobian of the simulated moments.
If the simulation is long enough, this provideds a good approximation for
the expected value of the jacobian. The output is the "D" matrix in the terminology
of Gouriéroux and Monfort (1996).
"""
function calculate_D(sMMProblem::MSMProblem, theta0::Array{Float64,1}; method::Symbol = :central)

  Calculus.jacobian(x -> sMMProblem.simulate_empirical_moments_array(x), theta0, method)

end

"""
  calculate_Avar!(sMMProblem::MSMProblem, theta0::Array{Float64,1}; method::Symbol = :central)

Calculate the asymptotic variance of the SMM estimator using the sandwich formula.
This function assume that your model is simulating time series.
As a result, the asymptotic variance of the SMM estimator is (1+tau) times
the asymptotic variance of the GMM one, where tau is the ratio of the sample size
to the size of the simulated sample. See Duffie and Singleton (1993) and Gouriéroux and Monfort (1996).
"""
function calculate_Avar!(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, tSimulation::Int64; method::Symbol = :central)

  # Safety Checks
  if sMMProblem.Sigma0 == Array{Float64}(undef, 0,0)
    error("Please initialize sMMProblem.Sigma0 using the function set_Sigma0!.")
  end

  tau = tData/tSimulation

  # Jacobian matrix
  D = calculate_D(sMMProblem, theta0, method = method)

  # Weigthing matrix (only diagonal matrix for the moment)
  W = diagm([1/sMMProblem.empiricalMoments[k][2] for k in keys(sMMProblem.empiricalMoments)])

  Sigma1 = transpose(D)*W*D

  Sigma2 = transpose(D)*W*sMMProblem.Sigma0*W*D

  inv_Sigma1 = inv(Sigma1)

  # sandwich formula, where tau captures the extra noise introduced by simulation,
  # compared to the usual GMM estimator.
  # I use the function Symmetric, because otherwise a test issymmetric()
  # on the asymptotic variance will return "false" because of numerical approximations
  sMMProblem.Avar = Symmetric((1+tau)*inv_Sigma1*Sigma2*inv_Sigma1)

end

"""
  calculate_se(sMMProblem::MSMProblem, tData::Int64, tSimulation::Int64)

Function to calculate the standard error associated to the the ith parameter,
respecting the ordering given by the ordered dictionary sMMProblem.priors
"""
function calculate_se(sMMProblem::MSMProblem, tData::Int64, i::Int64)

  # Safety Checks
  if sMMProblem.Avar == Array{Float64}(undef, 0,0)
    error("Please caclulate the asymptotic variance using the function calculate_Avar!.")
  end

  sqrt((1/tData)*sMMProblem.Avar[i,i])

end


"""
  calculate_t(sMMProblem::MSMProblem, tData::Int64, tSimulation::Int64)

Function to calculate the t-statistic associated to following test:
H0: theta_i = 0
H1: theta_i != 0
The ordering of parameters is the one given by the ordered dictionary sMMProblem.priors
"""
function calculate_t(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, i::Int64)

  # Safety Checks
  if sMMProblem.Avar == Array{Float64}(undef, 0,0)
    error("Please caclulate the asymptotic variance using the function calculate_Avar!.")
  end

  theta0[i]/calculate_se(sMMProblem, tData, i)

end


"""
  calculate_pvalue(sMMProblem::MSMProblem, tData::Int64, tSimulation::Int64)

Function to calculate the p-value associated to following test:
H0: theta_i = 0
H1: theta_i != 0
The ordering of parameters is the one given by the ordered dictionary sMMProblem.priors
"""
function calculate_pvalue(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, i::Int64)

  # Safety Checks
  if sMMProblem.Avar == Array{Float64}(undef, 0,0)
    error("Please caclulate the asymptotic variance using the function calculate_Avar!.")
  end

  t =  calculate_t(sMMProblem, theta0, tData, i)

  # asymptotically normally distributed N(0,1)
  d = Normal(0,1)

  # p-value  = 2 times area to the right of t
  pvalue = 2*(1.0 - cdf(d, t))

  return pvalue

end


"""
  calculate_CI(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, i::Int64)

Function to calculate an alpha confidence interval for the ith parameter.
"""
function calculate_CI(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, i::Int64, alpha::Float64)

  # Safety Checks
  if sMMProblem.Avar == Array{Float64}(undef, 0,0)
    error("Please caclulate the asymptotic variance using the function calculate_Avar!.")
  end

  # standard error
  se = calculate_se(sMMProblem, tData, i)

  # asymptotically normally distributed N(0,1)
  d = Normal(0,1)

  critical_value = - quantile(d, alpha)

  #return lower and upper bound of the confidence interval
  return theta0[i] - se*critical_value, theta0[i] + se*critical_value

end


function summary_table(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, alpha::Float64)

  # Safety Checks
  if sMMProblem.Avar == Array{Float64}(undef, 0,0)
    error("Please caclulate the asymptotic variance using the function calculate_Avar!.")
  end

  df = DataFrame(Estimate = Float64[], StdError = Float64[], tValue = Float64[], pValue = Float64[], ConfIntervalLower = Float64[], ConfIntervalUpper = Float64[])

  for i = 1:length(theta0)

    se = calculate_se(sMMProblem, tData, i)
    t = calculate_t(sMMProblem, theta0, tData, i)
    p = calculate_pvalue(sMMProblem, theta0, tData, i)
    CI_lower, CI_upper = calculate_CI(sMMProblem, theta0, tData, i, alpha)

    push!(df, [theta0[i], se, t, p, CI_lower, CI_upper])

  end

  return df

end
