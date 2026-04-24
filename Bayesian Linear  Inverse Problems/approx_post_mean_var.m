function [mhat, diagChat] = approx_post_mean_var(lambda, G, Gamma, y, Bk, Vk, ProbInfo)
% Compute the approximate posterior mean and the diagonal of the 
% posterior covariance for the separable 2D Gaussian deblurring 
% model with Matern kernel.
% 
% Inputs:
%   lambda    : prior precision scaling
%   G         : forward matrix
%   Gamma  : diagonal SPD matrix in data space
%   y      : data vector, used to compute beta_1
%   Bk     : (k+1)-by-k bidiagonal matrix
%   Vk     : m-by-k basis matrix
%   ProbInfo  : struct containing
%               .K1  : N-by-N 1D Matérn covariance matrix
%
% Outputs:
%   m_hat     : N^2-by-1 approxiamte posterior mean
%   diagC_post : N^2-by-1 approximate posterior covariance diagonal
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

K1 = ProbInfo.K1;

% ---------- U = Sigma G' Vk ----------
GtVk = apply_Gt_cols(G, Vk);   % n×k
U = Sigma_apply(GtVk, K1);    

% ---------- low-rank covariance correction ----------
Jk = Bk' * Bk;
Kmat = (lambda * eye(size(Jk)) + Jk) \ Jk;
Kmat = 0.5 * (Kmat + Kmat');     % numerical symmetrization

UK = U * Kmat;
diag_lowrank = sum(U .* UK, 2);

% ---------- diag(Sigma) ----------
d1 = diag(K1);
diagSigma = kron(d1, d1);

% ---------- diagonal of approximate posterior covariance ----------
diagChat = (1/lambda) * (diagSigma - diag_lowrank);
diagChat = abs(diagChat);   % numerical safeguard

% ---------- approximate posterior mean ----------
g = diag(Gamma);
beta1 = sqrt(sum((y.^2) ./ g));

rhs = Bk' * [beta1; zeros(size(Bk,1)-1,1)];
coeff = (Jk + lambda * eye(size(Jk))) \ rhs;

mhat = Sigma_apply(GtVk * coeff, K1);

end


%%-------------------------------------------
function Y = Sigma_apply(X, K1)
% X: n×k → reshape each column

N = size(K1,1);
k = size(X,2);
Y = zeros(size(X));

for j = 1:k
    Xt = reshape(X(:,j), N, N);
    Yt = K1 * Xt * K1';
    Y(:,j) = Yt(:);
end

end