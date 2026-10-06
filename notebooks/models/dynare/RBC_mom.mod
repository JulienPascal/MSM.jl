// Real business cycle model of RBC.mod, estimated with Dynare's own simulated method of moments
// (Dynare 6 for MATLAB or Octave; see the last section of RBCDynare.ipynb).
//
// The observed series are 100*log(c) and 100*log(y), minus constants shift_c and shift_y (the means
// of the data). Dynare's matched moments are uncentered (E[x], E[x*x], ...): with the shift, the
// second moments are close to variances and covariances, which keeps them well scaled. The shifts
// are given on Dynare's command line: dynare RBC_mom -DSHIFT_C=... -DSHIFT_Y=...

var c k y z log_c log_y;
varexo eps_z;
parameters alpha rho log_std_z delta beta shift_c shift_y;

alpha = 0.33;
rho = 0.9;
log_std_z = log(0.7);
delta = 0.025;
beta = 0.99;
shift_c = @{SHIFT_C};
shift_y = @{SHIFT_Y};

model;
1 / c = (beta / c(+1)) * (alpha * exp(z(+1)) * k^(alpha - 1) + (1 - delta));
c + k = (1 - delta) * k(-1) + y;
y = exp(z) * k(-1)^alpha;
z = rho * z(-1) + exp(log_std_z) / 100 * eps_z;
log_c = 100 * log(c) - shift_c;
log_y = 100 * log(y) - shift_y;
end;

// Analytical non-stochastic steady state
steady_state_model;
k = ((1 / beta - 1 + delta) / alpha)^(1 / (alpha - 1));
y = k^alpha;
c = y - delta * k;
z = 0;
log_c = 100 * log(c) - shift_c;
log_y = 100 * log(y) - shift_y;
end;

shocks;
var eps_z; stderr 1;
end;

steady;

varobs log_c log_y;

// Starting value, lower bound and upper bound (the MSM priors of RBCDynare.ipynb)
estimated_params;
alpha, 0.3, 0.1, 0.6;
rho, 0.8, 0.0, 0.99;
log_std_z, 0, log(0.1), log(3);
end;

// The 7 moments of RBCDynare.ipynb: means, second moments, cross moment and first-order autocovariances
matched_moments;
log_c;
log_y;
log_c * log_c;
log_y * log_y;
log_c * log_y;
log_c * log_c(-1);
log_y * log_y(-1);
end;

// Simulated sample of 10 times the length of the data, after a burn-in of 200 periods (as in the notebook)
method_of_moments(datafile = 'observed_shifted.csv', mom_method = SMM, order = 1, simulation_multiple = 10,
                  burnin = 200, mode_compute = 4, nograph);
