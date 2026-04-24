function [mu, W] = eigs_MGamma(Mfun, Gamma, k)
% Compute the first k largest nonzero generalized eigenpairs of
%   M w = mu * Gamma * w,
%   where W is M-column orthornormal
% 
%  The computation is based on the symmetric standard eigenproblem
%          T u = mu u,   T = Gamma^{-1/2} M Gamma^{-1/2}.
%   2. If w_i = Gamma^{-1/2} u_i, then w_i are Gamma-orthonormal.
%   3. For nonzero mu_i, scaling by 1/sqrt(mu_i) yields M-orthonormal vectors.
% 
% Inputs:
%   M: symmetric positive semidefinite and accessed only via Mfun(z)=Mz
%   Gamma： diagonal and positive definite
%
% Outputs:
%   mu: k-by-1 vector of the largest nonzero generalized eigenvalues
%   W: matrix whose columns are the corresponding M-orthonormal
%        generalized eigenvectors, i.e., W'* M*W = I
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
% 

g = diag(Gamma);
m = size(g, 1);
ginvsqrt = 1 ./ sqrt(g);

% Symmetric operator T = Gamma^{-1/2} M Gamma^{-1/2}
Tfun = @(x) ginvsqrt .* Mfun(ginvsqrt .* x);

% need to ask eigs for a few more eigenpairs, in case some are zero
nev = min(max(2*k, k+5), m);

opts.isreal = true;
opts.issym  = true;
opts.tol    = 1e-10;
opts.maxit  = 500;
opts.disp   = 0;

% Compute largest eigenvalues of T
[U, D] = eigs(Tfun, m, nev, 'largestreal', opts);

mu_all = real(diag(D));
U = real(U);

% Sort in descending order
[mu_all, idx] = sort(mu_all, 'descend');
U = U(:, idx);

% Remove tiny negative values caused by roundoff
mu_all(mu_all < 0 & abs(mu_all) < 1e-12) = 0;

% Keep only nonzero eigenvalues
tol_zero = 1e-10;
pos = find(mu_all > tol_zero);

if isempty(pos)
    mu = [];
    W = [];
    warning('No nonzero generalized eigenvalues found.');
    return;
end

if length(pos) < k
    warning('Only %d nonzero generalized eigenvalues found (requested %d).', length(pos), k);
    k_eff = length(pos);
else
    k_eff = k;
end

mu = mu_all(pos(1:k_eff));
U = U(:, pos(1:k_eff));

% Recover Gamma-orthonormal generalized eigenvectors:
% w_gamma = Gamma^{-1/2} u
Wg = ginvsqrt .* U;

% Convert to M-orthonormal generalized eigenvectors:
% if M w = mu Gamma w and w'Gamma w = 1, then w'Mw = mu
% so divide by sqrt(mu)
W = Wg ./ sqrt(mu(:))';

end