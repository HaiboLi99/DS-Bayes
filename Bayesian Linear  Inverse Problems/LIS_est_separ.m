function [X, postVar, mstd] = LIS_est_separ(A, b, M, Lam, k, ProbInfo)
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
%   x       : the first k approximate posterior mean
%   postVar : the first k diagonal of approximate posterior covariance
%   mstd    : the first k sqrt(mean(postVar))
%
% Haibo Li
% 16 April 2026

h = waitbar(0, 'Beginning LIS computation: please wait ...');
n = size(A, 2); 
K = ProbInfo.K1;

waitbar(0.1, h, 'Computing Cholesky factor...');
L1 = chol(K, 'lower'); 
NN = size(L1, 1); 
Minv = @(y) y ./ diag(M);

waitbar(0.3, h, 'Building operator...');
% Apply S = L1 \kron L1
apply_S  = @(v) reshape(L1  * reshape(v,NN,NN) * L1',  [],1);
apply_ST = @(v) reshape(L1' * reshape(v,NN,NN) * L1,   [],1);

% Operator K = S' A' M^{-1} A S
Kfun = @(z) apply_ST( A' * Minv( A * apply_S(z) ) );

waitbar(0.5, h, 'Computing eigenpairs (eigs)...');
opts.isreal = true;
opts.issym  = true;
[Z, D] = eigs(Kfun, n, k, 'largestreal', opts);

waitbar(0.7, h, 'Constructing LIS basis...');
mu = real(diag(D));
mu = max(mu, 0);
[mu, idx] = sort(mu, 'descend');
Z = Z(:,idx);

% generalized eigenvectors: w_i = S z_i
W = zeros(n,k);
for i = 1:k
    W(:,i) = apply_S(Z(:,i));
end

waitbar(0.9, h, 'Computing posterior mean/variance...');
X = zeros(n, k);
postVar = zeros(n,k);
mstd = zeros(k, 1);

apply_N = @(v) reshape(K * reshape(v,NN,NN) * K, [], 1);

g  = A' * Minv(b);
Ng = apply_N(g);
diagN = kron(diag(K), diag(K));

for i = 1:k 
    mu_i = mu(1:i);
    W_i = W(:,1:i);
    lambda = Lam(i);
    coeff_i = mu_i ./ (lambda * (lambda + mu_i));
    Wtg = W_i' * g;
    x = (1/lambda) * Ng - W_i * (coeff_i .* Wtg);
    X(:,i) = x;
    postVar_i = (1/lambda) * diagN - (W_i.^2) * coeff_i;
    postVar(:,i) = max(postVar_i, 0);
    mstd_i = sqrt(mean(postVar(:,i)));
    mstd(i) = mstd_i;
end 

waitbar(1, h, 'LIS Done!');
pause(0.2); 
close(h);

end
