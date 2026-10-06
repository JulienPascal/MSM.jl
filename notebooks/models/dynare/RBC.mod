// Real business cycle model of notebooks/models/RBC.ipynb, written for Dynare.jl (see RBCDynare.ipynb).
// The variables are in levels; the standard deviation of the technology shock is parameterized by
// its logarithm, log_std_z, with std_z in percent (as in RBC.ipynb).

var c k y z;
varexo eps_z;
parameters alpha rho log_std_z delta beta;

alpha = 0.33;
rho = 0.9;
log_std_z = log(0.7);
delta = 0.025;
beta = 0.99;

model;
1 / c = (beta / c(+1)) * (alpha * exp(z(+1)) * k^(alpha - 1) + (1 - delta));
c + k = (1 - delta) * k(-1) + y;
y = exp(z) * k(-1)^alpha;
z = rho * z(-1) + exp(log_std_z) / 100 * eps_z;
end;

// Analytical non-stochastic steady state
steady_state_model;
k = ((1 / beta - 1 + delta) / alpha)^(1 / (alpha - 1));
y = k^alpha;
c = y - delta * k;
z = 0;
end;

shocks;
var eps_z; stderr 1;
end;

steady;

// First-order solution
stoch_simul(order = 1, irf = 0, noprint);
