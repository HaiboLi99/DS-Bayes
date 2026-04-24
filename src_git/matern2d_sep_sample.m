function [y, K] = matern2d_sep_sample(x_low, x_high, N, nu, rho, sigma, seed)
% Sample true image from separable Matérn prior
%
% Inputs:
%   [x_low, x_high]: the 1D interval for discretization
%   N: number of uniform grids in one dimension
%   nu, rho, sigma: hyperparameters of the 1D Matern kernel
%   seed: random seed
%
% Output:
%   y  : a sampled vector from the prior
%   K  : matrix of the 1d Matern kernel
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

h = (x_high - x_low) / N;
x = x_low + (0.5:N-0.5) * h;
K1 = matern1d_cov(x, nu, rho, sigma);
K1 = 0.5 * (K1 + K1');
K = K1 + 1e-10 * eye(N);  
L1 = chol(K, 'lower');

% If Z has iid N(0,1), then X = L1 * Z * L1' has vec(X) ~ N(0, K1 \kron K1)
rng(seed);
Z = randn(N, N);
Ytrue = L1 * Z * L1';
y = Ytrue(:);

end


%%------------------------------------------------------------------
function K = matern1d_cov(x, nu, rho, sigma)
% 1D Matérn covariance matrix on points x.
x = x(:);
D = abs(x - x.');
K = zeros(size(D));

if abs(nu - 0.5) < 1e-14
    K = sigma^2 * exp(-D / rho);
    return;
end

R = sqrt(2 * nu) * D / rho;

% diagonal
K(D == 0) = sigma^2;
% off-diagonal
idx = (D > 0);
c = sigma^2 * (2^(1 - nu)) / gamma(nu);
K(idx) = c * (R(idx).^nu) .* besselk(nu, R(idx));

end