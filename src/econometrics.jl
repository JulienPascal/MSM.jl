"""
  function calculate_D(sMMProblem::MSMProblem, theta0::Array{Float64,1}; method::Symbol = :central)

Function to calculate the jacobian of the simulated moments.
If the simulation is long enough, this provideds a good approximation for
the expected value of the jacobian. The output is the "D" matrix in the terminology
of Gouriéroux and Monfort (1996).
"""
function calculate_D(sMMProblem::MSMProblem, theta0::Array{Float64,1})

  #Calculus.jacobian(x -> sMMProblem.simulate_empirical_moments_array(x), theta0, method)
  jacobian(central_fdm(5, 1), x -> sMMProblem.simulate_empirical_moments_array(x), theta0)[1]

end

"""
  calculate_Avar!(sMMProblem::MSMProblem, theta0::Array{Float64,1}; tau::Float64 = 1.0)

Calculate the asymptotic variance of the SMM estimator using the sandwich formula.
This function assume that your model is simulating time series.
As a result, the asymptotic variance of the SMM estimator is (1+tau) times
the asymptotic variance for the GMM estimator, where tau is the ratio of the sample size
to the size of the simulated sample. See Duffie and Singleton (1993) and Gouriéroux and Monfort (1996).
"""
function calculate_Avar!(sMMProblem::MSMProblem, theta0::Array{Float64,1}; tau::Float64 = 1.0)

  # Safety Checks
  if isempty(sMMProblem.Sigma0) == true
    error("Please initialize sMMProblem.Sigma0 using the function set_Sigma0!.")
  end

  # Jacobian matrix
  D = calculate_D(sMMProblem, theta0)

  Sigma1 = transpose(D)*sMMProblem.W*D

  Sigma2 = transpose(D)*sMMProblem.W*sMMProblem.Sigma0*sMMProblem.W*D

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
  if isempty(sMMProblem.Avar) == true
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
  if isempty(sMMProblem.Avar) == true
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
  if isempty(sMMProblem.Avar) == true
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
  if isempty(sMMProblem.Avar) == true
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
  if isempty(sMMProblem.Avar) == true
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

"""
  cov_NW(data::Matrix; l::Int64 = -1)

Function to calculate Newey–West (1987) variance-covariance matrix. `data` is the
data matrix with each row representing a time period and each column representing
a variable. `l` is the nummber of lags to include.
"""
function cov_NW(data::Matrix; l::Int64 = -1)
    # See: Newey, Whitney K; West, Kenneth D (1987)
    # data: rows = time; columns = dimension
    # l: number of lags to include
    #number of periods
    T=size(data,1);
    nlag=l;
    #Default value if user does not provide l
    if nlag == -1
        nlag=Int(min(floor(1.2*T^(1/3)),T));
    end

    #Demean each column
    data = data .- repeat(mean(data, dims=1), outer = [T, 1]);

    # Newey West weights:
    w=(nlag+1 .- collect(0:nlag))./(nlag+1);

    # Variance-covariance matrix when no serial correlation
    V=transpose(data)*data./(T-1);

    # Additional terms (non-zeros if serial correlation)
    for i=1:nlag
        Gammai=(transpose(data[(i+1):T,:])*data[1:T-i,:])./(T-1);
        V = V + w[i+1].*(Gammai + transpose(Gammai));
    end

    return V
end


"""
  J_test(sMMProblem::MSMProblem, theta0::Array{Float64,1})

Run a J-test, also called a test for over-identifying restrictions. The null hypothesis that the model is “valid”.
The alternative hypothesis that model is “invalid”. For the test to be valid,
the weigth matrix must converge in probability to the efficient weighting matrix
(Sigma0^-1).

#Ouput:
* J: value of the J-statistic
* c: critical value associated to the J-test
"""
function J_test(sMMProblem::MSMProblem, theta0::Array{Float64,1}, tData::Int64, alpha::Float64)

  df=length(keys(sMMProblem.empiricalMoments)) - length(theta0)
  #Chi² distribution with k-l degrees of freedom, where
  d_chi=Chi(df)
  #Calculate J
  simulatedMoments=sMMProblem.simulate_empirical_moments(theta0)
  # to store the distance between empirical and simulated moments
  arrayDistance = zeros(length(keys(sMMProblem.empiricalMoments)))
  for (indexMoment, k) in enumerate(keys(sMMProblem.empiricalMoments))
    arrayDistance[indexMoment] = (sMMProblem.empiricalMoments[k][1] - simulatedMoments[k])
  end

  J = tData*transpose(arrayDistance)*sMMProblem.W*arrayDistance
  #Critical value above which the null hypothesis is rejected
  #Look at the 1.0 - alpha percentile of Chi²(df)
  c = quantile(d_chi, 1.0 - alpha)

  return J, c

end
