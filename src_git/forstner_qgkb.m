function [dF_vals, bnd_vals, zeta, gamma] = forstner_qgkb(G, Sigma, Gamma, y, Vmax, Bmax, Lam)
% Compute d_F(C_{lambda_k}, Chat_{lambda_k}^{(k)}) and the bound gamma_k/lambda_k
%
% Inputs:
%   G, Sigma, Gamma, y : as before
%   Vmax  : final V matrix from Q-GKB, size m-by-K
%   Bmax  : final bidiagonal matrix from Q-GKB, size (K+1)-by-K
%   Lam   : K-by-1 vector of lambda_k
%
% Outputs:
%   dF_vals  : K-by-1 actual Forstner distances
%   bnd_vals : K-by-1 bounds gamma_k / lambda_k
%   zeta     : zeta_k with index from 0 to K
%   gamma    : gamma_k with index from 0 to K
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

K = length(Lam);

% [zeta0, gamma0] = qgkb_zeta_gamma(G, Gamma, Sigma, Bmax);
[zeta, gamma] = zeta_gamma(G, Gamma, Sigma, Vmax, Bmax);
% zeta = max(zeta0, zeta1);
% gamma = max(gamma0, gamma1);

dF_vals = zeros(K,1);
bnd_vals = zeros(K,1);

for k = 1:K
    lambda = Lam(k);
    Vk = Vmax(:,1:k);
    Bk = Bmax(1:k+1,1:k);

    % [~, C] = exact_post(lambda, Sigma, G, Gamma, y);
    % [~, Chat] = approx_post(lambda, Sigma, G, Gamma, y, Bk, Vk);
    % dF_vals(k) = forstner_dist(C, Chat);

    dF_vals(k) = forstner_post(lambda, Sigma, G, Gamma, Vk, Bk);
    bnd_vals(k) = gamma(k+1) / lambda;
end

end

