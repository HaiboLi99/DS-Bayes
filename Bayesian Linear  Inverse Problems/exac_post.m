function [m_post, diagC_post] = exac_post(A, b, M, N, lambda)
% Compute the exact posterior mean and the diagonal of the posterior
% covariance for a general linear Gaussian system.
%
% Model: 
%   b = A x + e,   e ~ N(0, M)
%   x ~ N(0, lambda^{-1} N).
% The exact posterior is
%   x | b ~ N(x_post, C_post),
%   x_post = C_post * A'*M^{-1}b
%   C_post = (A' M^{-1} A + lambda N^{-1})^{-1}.
%
% Inputs:
%   A:  full or sparse mxn matrix;
%   b: right-hand side vector
%   M: covaraince matrix of noise e, diagonal
%   N: covariance matrix
%   lambda： hyperparameter
%
% Outputs:
%   m_post     : n-by-1 exact posterior mean
%   diagC_post : n-by-1 exact posterior covariance diagonal
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 28, Sept, 2026.


[m, n] = sizemm(A);

% ---------- input checks ----------
assert(numel(b) == m, 'b must have length m.');
assert(isequal(size(M), [m, m]), 'M must be m-by-m.');
assert(isequal(size(N), [n, n]), 'N must be n-by-n.');
assert(lambda > 0, 'lambda must be positive.');

% ---------- H = A' M^{-1} A,  rhs = A' M^{-1} b ----------
if isdiag(M)
    Minv = 1 ./ diag(M);
    H    = A' * (Minv .* A);
    rhs  = A' * (Minv .* b);
else
    Lm   = chol(M, 'lower');
    MA   = Lm \ A;
    H    = MA' * MA;
    rhs  = A' * (Lm' \ (Lm \ b));
end
H = 0.5 * (H + H');      % enforce symmetry

% ---------- P = H + lambda * N^{-1} ----------
Ln   = chol(N, 'lower');       % N = Ln * Ln'
Ninv = Ln' \ (Ln \ eye(n));    % N^{-1} = Ln^{-T} * Ln^{-1}
P    = H + lambda * Ninv;
P    = 0.5 * (P + P');         % enforce symmetry

% ----------  Cholesky factorization of P ----------
Lp = chol(P, 'lower');         % P = Lp * Lp'

% posterior mean: m_post = P^{-1} * rhs
m_post = Lp' \ (Lp \ rhs);

% posterior covariance diagonal: diag(P^{-1})
% P^{-1} = Lp^{-T} * Lp^{-1},  Linv = Lp^{-1}
% diag(P^{-1})_i = sum_j (Linv)_{j,i}^2
Linv       = inv(Lp);                  % triangular inverse
diagC_post = sum(Linv.^2, 1)';         % n-by-1
diagC_post = max(diagC_post, 0);       % numerical safeguard

end