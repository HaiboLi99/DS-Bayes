function [m_post, diagC_post, VarField, aux] = exact_post_mean_var(b, lambda, sigma_eps, ProbInfo)
% Compute the exact posterior mean and the diagonal of the posterior
% covariance for the separable 2D Gaussian deblurring model with Matern kernel.
%
% Model: b = A x + eps,   eps ~ N(0, sigma_eps^2 I)
%   x ~ N(0, lambda^{-1} (K1 \kron K1))
%
% Inputs:
%   b         : M^2-by-1 observation vector
%   lambda    : prior precision scaling
%   sigma_eps : noise std
%   ProbInfo  : struct containing
%               .G1  : M-by-N 1D Gaussian blur matrix
%               .K1  : N-by-N 1D Matérn covariance matrix
%               .h   : fine-grid spacing = 1/N
%               .xSize = [N,N]
%               .bSize = [M,M]
%
% Outputs:
%   m_post     : N^2-by-1 exact posterior mean
%   diagC_post : N^2-by-1 exact posterior covariance diagonal
%   VarField   : N-by-N variance field
%   aux        : struct with intermediate quantities
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

G = ProbInfo.G1;
K = ProbInfo.K1;
h = ProbInfo.h;

% N = ProbInfo.xSize(1);
M = ProbInfo.bSize(1);

% ---------- factorize prior ----------
K = 0.5 * (K + K');
S = chol(K, 'lower');

% ---------- eigendecomposition of prior-whitened 1D Hessian ----------
B = S' * (G' * G) * S;
B = 0.5 * (B + B');

[U, Dmu] = eig(B);
mu = real(diag(Dmu));

% sort descending for consistency
[mu, idx] = sort(mu, 'descend');
U = U(:, idx);
W = S * U;

% ---------- exact posterior mean ----------
Y = reshape(b, M, M);

R = (h^2 / sigma_eps^2) * (G' * Y * G);
T = S' * R * S;

beta = h^4 / (lambda * sigma_eps^2);

Den = 1 + beta * (mu * mu.');         % N-by-N
Ccoef = (U' * T * U) ./ Den;          % elementwise division

Mpost = (1 / lambda) * W * Ccoef * W';
m_post = Mpost(:);

% ---------- exact posterior covariance diagonal ----------
P = W.^2;                             % elementwise square
Dvar = 1 ./ Den;                      % N-by-N

VarField = (1 / lambda) * P * Dvar * P';
diagC_post = VarField(:);

% ---------- auxiliary info ----------
aux = struct();
aux.mu = mu;
aux.U = U;
aux.W = W;
aux.beta = beta;

end