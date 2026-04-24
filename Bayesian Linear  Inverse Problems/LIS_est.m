function [x, postVar, mstd] = LIS_est(A, b, M, N, lambda, k)
% Approximate posterior mean and diagonal variance using a rank-k LIS update.
% Model:
%   b = A x + e,   e ~ N(0, M)
%   x ~ N(0, lambda^{-1} N).
% The exact posterior is
%   x | b ~ N(x_post, C_post),
%   C_post = (A' M^{-1} A + lambda N^{-1})^{-1}.
%
% This routine computes a rank-k LIS approximation:
%   C_k = lambda^{-1} N - W D W',
% where columns of W are the first k generalized eigenvectors of
%   (H, N^{-1}),   H = A' M^{-1} A,
% normalized so that W' (N \ W) = I,
% and D = diag(mu ./ (lambda * (lambda + mu))).
%
% Inputs:
%   A:  full or sparse mxn matrix;
%   b: right-hand side vector
%   M: covaraince matrix of noise e, diagonal
%   N: covariance matrix
%   lambda： hyperparameter
%   k: dimension of LIS
% 
% Outputs:
%   x       : approximate posterior mean
%   postVar : diagonal of approximate posterior covariance
%   mstd    : sqrt(mean(postVar))
%
% Haibo Li
% 16 April 2026

% h = waitbar(0, 'Beginning LIS computation: please wait ...');
n = size(A, 2);
Minv = @(y) y ./ diag(M);

% waitbar(0.1, h, 'Computing Cholesky factor...');
S = chol(N, 'lower');

% waitbar(0.3, h, 'Building operator...');
Kfun = @(z) S' * (A' * Minv(A * (S * z)));

% waitbar(0.5, h, 'Computing eigenpairs (eigs)...');
opts.isreal = true;
opts.issym  = true;
[Z, Dmu] = eigs(Kfun, n, k, 'largestreal', opts);

% waitbar(0.7, h, 'Constructing LIS basis...');
mu = real(diag(Dmu));
mu = max(mu, 0);
[mu, idx] = sort(mu, 'descend');
Z = Z(:, idx);
W = S * Z;

% waitbar(0.9, h, 'Computing posterior mean/variance...');
g = A' * Minv(b);
coeff = mu ./ (lambda * (lambda + mu));
Ng = N * g;
Wtg = W' * g;
x = (1 / lambda) * Ng - W * (coeff .* Wtg);

postVar = (1 / lambda) * diag(N) - (W.^2) * coeff;
postVar = max(postVar, 0);
mstd = sqrt(mean(postVar));

% waitbar(1, h, 'LIS Done!');
% pause(0.2); 
% close(h);

end