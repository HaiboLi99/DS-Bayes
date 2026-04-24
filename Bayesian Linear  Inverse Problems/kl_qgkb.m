function [DKL_vals, bnd_vals, zeta, gamma] = kl_qgkb(G, Sigma, Gamma, y, Vmax, Bmax, Lam)
% Compute D_KL( \hat pi_k || pi ) and the upper bound by QGKB
%
% Inputs:
%   G, Sigma, Gamma, y : as before
%   Vmax  : final V matrix from Q-GKB, size m-by-K
%   Bmax  : final bidiagonal matrix from Q-GKB, size (K+1)-by-K
%   Lam   : K-by-1 vector of lambda_k
%
% Outputs:
%   DKL_vals  : K-by-1 actual Forstner distances
%   bnd_vals  : K-by-1 bounds gamma_k / lambda_k
%   zeta      : zeta_k with index from 0 to K
%   gamma     : gamma_k with index from 0 to K
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

K = length(Lam);

% [zeta0, gamma0] = qgkb_zeta_gamma(G, Gamma, Sigma, Bmax);
[zeta, gamma] = zeta_gamma(G, Gamma, Sigma, Vmax, Bmax);
% zeta = max(zeta0, zeta1);
% gamma = max(gamma0, gamma1);

alpha1 = Bmax(1,1);
g = diag(Gamma);
beta1 = sqrt(sum((y.^2) ./ g));

DKL_vals = zeros(K,1);
bnd_vals = zeros(K,1);

for k = 1:K
    lambda = Lam(k);
    Vk = Vmax(:,1:k);
    Bk = Bmax(1:k+1,1:k);

    [m_exact, C_exact] = exact_post(lambda, Sigma, G, Gamma, y);
    [m_approx, C_approx] = approx_post(lambda, Sigma, G, Gamma, y, Bk, Vk);
    DKL_vals(k) = kl_gauss(m_approx, C_approx, m_exact, C_exact);
    DKL_vals(k) = abs(DKL_vals(k));

    % DKL_vals(k) = kl_post(lambda, Sigma, G, Gamma, y, Vk, Bk);
    bnd_vals(k) = (1/(2*lambda)) * ...
            (zeta(k+1) + (alpha1^2 * beta1^2 * gamma(k+1)^2) / (lambda * (lambda + gamma(k+1))));
end

end





