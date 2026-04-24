function [coeff, mu, W] = proj_coeff(G, y, Gamma, Sigma, k, plotON)
% Given G, Gamma, Sigma and y, first form the operator
%  M = G * Sigma * G'
% then compute the first k largest nonzero generalized eigenpairs
% of (M, Gamma) using eigs_MGamma, and finally compute the
% projection coefficients c_i in
%   P_{w_i}(Gamma^{-1} y) = c_i * w_i,
% where w_i are M-orthonormal generalized eigenvectors:
%  M w_i = mu_i * Gamma * w_i,    w_i' * M * w_i = 1.
%
% Formula: coeff(i) = mu(i) * (y' * w_i)
%
% Also plots:
%   (1) mu_i vs i
%   (2) projection magnitude vs i
% 
% Inputs:
%   G      : matrix or function handle for the forward operator
%   y      : data vector in R^m
%   Gamma  : diagonal positive definite matrix in R^(m x m)
%   Sigma  : matrix or function handle for the prior covariance
%   k      : number of largest nonzero generalized eigenpairs requested
%   plotON : optional, 1 = plot, 0 = no plot (default = 0)
%
% Outputs:
%   coeff  : projection coefficients c_i
%   mu     : generalized eigenvalues
%   W      : M-orthonormal generalized eigenvectors
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
% 

if nargin < 6
    plotON = 0;
end

% dimension check
m = length(y);
if size(Gamma,1) ~= m || size(Gamma,2) ~= m
    error('Dimension mismatch: Gamma must be m-by-m, where m = length(y).');
end

% Define Mfun(z) = G * Sigma * G' * z
Mfun = @(z) Gfun(z, G, Sigma);

% Compute first k largest nonzero generalized eigenpairs of (M, Gamma)
[mu, W] = eigs_MGamma(Mfun, Gamma, k);

% Compute projection coefficients:
%   coeff(i) = mu(i) * (y' * w_i)
mu = mu(:);
coeff = mu .* (W' * y);

% ---------- Plot ----------
if plotON == 1
    fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
    t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
    kk = length(mu);
    semilogy(1:kk, mu, 'o--', ...
    'LineWidth',1.5, ...
    'MarkerSize',5);
    hold on;
    semilogy(1:kk, abs(coeff), 'v-', ...
    'LineWidth',1.5, ...
    'MarkerSize',5);
    legend('$\hat{\mu}_i$', '$|P_{G_{i}}\Gamma^{-1}y|$', 'Location', 'northeast','interpreter','latex','fontsize',20);
    set(gca, 'FontSize', 12);
    xlabel('$i$','interpreter','latex','fontsize',20);
    ylabel('Gen-eig / Proj','interpreter','latex','fontsize',16);
    grid on;
    grid minor;
end

end


%%----------------------------------------------------------
function [mu, W] = eigs_MGamma(Mfun, Gamma, k)
% Compute the first k largest nonzero generalized eigenpairs of
%   M w = mu * Gamma * w,
% where W is M-column orthornormal
% 
% The computation is based on the symmetric standard eigenproblem
%   1. T u = mu u,   T = Gamma^{-1/2} M Gamma^{-1/2}.
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
%      generalized eigenvectors, i.e., W'* M*W = I
% 

g = diag(Gamma);
m = length(g);
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
[U, D] = eigs(Tfun, m, nev, 'largestabs', opts);

mu_all = real(diag(D));
U = real(U);

% Sort in descending order
[mu_all, idx] = sort(mu_all, 'descend');
U = U(:, idx);

% Remove tiny negative values caused by roundoff
mu_all(mu_all < 0 & abs(mu_all) < 1e-20) = 0;

% Keep only nonzero eigenvalues
tol_zero = 1e-20;
pos = find(mu_all > tol_zero);
% pos = find(mu_all>0);

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


