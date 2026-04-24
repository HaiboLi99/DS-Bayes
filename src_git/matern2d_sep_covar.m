
function y = matern2d_sep_covar(v, x_low, x_high, N, nu, rho, sigma)
% Apply separable 2D Matérn covariance:
%   y = (K1 ⊗ K1) v
%
% Inputs:
%   v: the vector
%   [x_low, x_high]: the 1D interval for discretization
%   N: number of uniform grids in one dimension
%   nu, rho, sigma: hyperparameters of the 1D Matern kernel
%
% Output:
%   y  : (K1 ⊗ K1) v
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

h = (x_high - x_low) / N;
x = x_low + (0.5:N-0.5) * h;
K1 = matern1d_cov(x, nu, rho, sigma);
K1 = 0.5 * (K1 + K1');
K1 = K1 + 1e-10 * eye(N);   % numerical safeguard

X = reshape(v, N, N);
Y = K1 * X * K1';
y = Y(:);
end


%%----------------------------------------------
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