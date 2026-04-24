function [m, C] = exact_post(lambda, Sigma, G, Gamma, y)
% Compute exact posterior covarianc:
%   C = (G' Gamma^{-1} G + lambda Sigma^{-1})^{-1}
% exact posterior mean: 
%   x_lambda = C_lambda G' Gamma^{-1} y
%
% Inputs:
%   lambda : positive scalar
%   Sigma  : matrix or function handle
%   G      : matrix or function handle
%   Gamma  : diagonal SPD matrix in data space
%   y      : data vector, used to compute beta_1
%   n      : parameter dimension (needed if G or Sigma is a handle)
%
% Output:
%   m   : n-by-1 exact posterior mean
%   C   : n-by-n exaxt posterior covariance matrix
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

if ~isa(G, 'function_handle')
    [~, n] = size(G);
else
    [~, n] = sizemm(G);
end

Sig = materialize_sigma(Sigma, n);
Gmat = materialize_G(G);

H = Gmat' * (Gamma \ Gmat);
Sinv = Sig \ eye(n);

C = (H + lambda * Sinv) \ eye(n);
C = 0.5 * (C + C');
m = C * (Gmat' * (Gamma \ y));

end
