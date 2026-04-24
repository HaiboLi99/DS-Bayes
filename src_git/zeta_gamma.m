function [zeta, gamma] = zeta_gamma(G, Gamma, Sigma, Vmax, Bmax)
% compute zeta_gamma_direct
% Compute zeta_k and gamma_k directly from the definitions:
%   zeta_k  = Tr(A - Ahat_k),
%   gamma_k = ||A - Ahat_k||_F,
% where
%   A     = Sigma^{1/2} H Sigma^{1/2},   H = G' Gamma^{-1} G,
%   Ahat_k = Sigma^{1/2} Hhat_k Sigma^{1/2},
%   Hhat_k = G' V_k (B_k' B_k) V_k' G.
%
% This routine is intended mainly for validation of the recurrence-based code.
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 11, April, 2026.
%

if ~isa(G, 'function_handle')
    [m, n] = size(G);
else
    [m, n] = sizemm(G);
end    

K = size(Bmax, 2);
    
% materialize Sigma and G
Sig = materialize_sigma(Sigma, n);
Gmat = materialize_G(G);

% build H
H = Gmat' * (Gamma \ Gmat);

% symmetric square root of Sigma
Sig = 0.5 * (Sig + Sig');
Rsig = chol(Sig, 'lower');   % Sig = Rsig * Rsig'

% A = Sigma^{1/2} H Sigma^{1/2}
A = Rsig' * H * Rsig;
A = 0.5 * (A + A');

zeta = zeros(K+1,1);
gamma = zeros(K+1,1);

% k = 0
zeta(1) = trace(A);
gamma(1) = norm(A, 'fro');

for k = 1:K
    Vk = Vmax(:,1:k);
    Bk = Bmax(1:k+1,1:k);

    Hhat = Gmat' * Vk * (Bk' * Bk) * Vk' * Gmat;
    Hhat = 0.5 * (Hhat + Hhat');

    Ahat = Rsig' * Hhat * Rsig;
    Ahat = 0.5 * (Ahat + Ahat');

    Dk = A - Ahat;
    Dk = 0.5 * (Dk + Dk');

    zeta(k+1) = max(trace(Dk),100*eps);  % may be negative
    gamma(k+1) = norm(Dk, 'fro');
end

end