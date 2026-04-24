function [mhat, Chat] = approx_post(lambda, Sigma, G, Gamma, y, Bk, Vk)
% Compute the approximate posterior covariance：
%  Chat = lambda^{-1} Sigma - lambda^{-1} Sigma G' Vk (lambda I + Bk'Bk)^{-1} Bk'Bk Vk' G Sigma
% approximate posterior mean:
%   xhat = Sigma G' Vk (Bk'Bk + lambda I)^{-1} Bk' beta1 e1
%
% Inputs:
%   lambda : positive scalar
%   Sigma  : matrix or function handle
%   G      : matrix or function handle
%   Gamma  : diagonal SPD matrix in data space
%   y      : data vector, used to compute beta_1
%   Bk     : (k+1)-by-k bidiagonal matrix
%   Vk     : m-by-k basis matrix
%
% Output:
%   mhat   : n-by-1 approximate posterior mean
%   Chat   : n-by-n approximate posterior covariance matrix
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
GtVk = apply_Gt_cols(G, Vk);   % n-by-k
U = Sig * GtVk;                   % n-by-k

Kmat = (lambda * eye(size(Bk,2)) + Bk' * Bk) \ (Bk' * Bk);
Chat = (1/lambda) * Sig - (1/lambda) * (U * Kmat * U');

Chat = 0.5 * (Chat + Chat');      % symmetrize
% Chat = Chat + 1e-12*eye(size(Chat,1));
    

Sig = materialize_sigma(Sigma, n);
GtVk = apply_Gt_cols(G, Vk);

g = diag(Gamma);
beta1 = sqrt(sum((y.^2) ./ g));

rhs = Bk' * [beta1; zeros(size(Bk,1)-1,1)];
mhat = Sig * GtVk * ((Bk' * Bk + lambda * eye(size(Bk,2))) \ rhs);

end


%------------------------------------------------
function GtV = apply_Gt_cols(G, V)
% Compute G' * V columnwise, output is n-by-k
%

if ~isa(G, 'function_handle')
    [m, n] = size(G);
else
    [m, n] = sizemm(G);
end

if size(G,2) ~= n || size(V,1) ~= m
    error('Dimensions not consistent')
end

k = size(V,2);
GtV = zeros(n, k);

if ~isa(G, 'function_handle')
    GtV = G' * V;
    return;
end

for j = 1:k
    GtV(:,j) = G(V(:,j), 'transp');
end

end