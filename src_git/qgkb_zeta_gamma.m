function [zeta, gamma] = qgkb_zeta_gamma(G, Gamma, Sigma, Bk)
% Compute zeta_k and gamma_k for k = 0,1,...,K,
% where
%   zeta_k  = Tr(A - Ahat_k),
%   gamma_k = ||A - Ahat_k||_F,
% used to bound the distance between the exact and approximate posterior.
%
% Inputs:
%   G      : matrix or function handle for forward operator
%   Gamma  : diagonal SPD matrix in data space
%   Sigma  : matrix or function handle for prior covariance
%   Bk     : final bidiagonal matrix from Q-GKB, size (K+1)-by-K
%
% Outputs:
%   zeta   : (K+1)-by-1 vector, zeta(1)=zeta_0
%   gamma  : (K+1)-by-1 vector, gamma(1)=gamma_0
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.

K = size(Bk, 2);
m = size(Gamma, 1);

% Build T = Gamma^{-1/2} M Gamma^{-1/2} explicitly in data space,
% where M = G Sigma G^T; T has the same nonzero eigenvalues as A.
g = diag(Gamma);
ginvsqrt = 1 ./ sqrt(g);
Mfun = @(z) Gfun(z, G, Sigma);

T = zeros(m, m);
for j = 1:m
    ej = zeros(m,1);
    ej(j) = 1;
    T(:,j) = ginvsqrt .* Mfun(ginvsqrt .* ej);
end
T = 0.5 * (T + T');   % symmetrize

% Initial values: zeta_0 = Tr(T), gamma_0^2 = ||T||_F^2
zeta = zeros(K+1, 1);
gamma2 = zeros(K+1, 1);

zeta(1) = trace(T);
gamma2(1) = norm(T, 'fro')^2;

alpha = diag(Bk);          % alpha_1,...,alpha_K
beta_sub = diag(Bk, -1);   % beta_2,...,beta_{K+1}

% Recurrence:
% zeta_k = zeta_{k-1} - d_k, where d_k = alpha_k^2 + beta_{k+1}^2
% gamma_1^2 = gamma_0^2 - d_1^2
% gamma_k^2 = gamma_{k-1}^2 - d_k^2 - 2(alpha_k beta_k)^2,  k>=2
for k = 1:K
    ak = alpha(k);
    bkp1 = beta_sub(k);         % beta_{k+1}
    dk = ak^2 + bkp1^2;

    zeta(k+1) = zeta(k) - dk;

    if k == 1
        gamma2(k+1) = gamma2(k) - dk^2;
    else
        bk = beta_sub(k-1);     % beta_k
        gamma2(k+1) = gamma2(k) - dk^2 - 2 * (ak * bk)^2;
    end

    % numerical safeguard
    if gamma2(k+1) < 0 && abs(gamma2(k+1)) < 1e-12
        gamma2(k+1) = 0;
    end
end

gamma = sqrt(max(gamma2, 0));

end