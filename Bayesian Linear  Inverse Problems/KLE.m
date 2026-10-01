function [X, V, s] = KLE(A, b, M, N, kk, lambda)
% Karhunen-Loeve expansion method for Bayesian linear inverse problems. The truncated KL representation is
%       x = V_i * diag(sqrt(s(1:i))) * z,
% where z ~ N(0, lambda(i)^(-1) I).
%
% Inputs:
%   A: forward matrix, m-by-n
%   b: observation vector, m-by-1
%   M: noise covariance matrix
%   N: prior covariance matrix/function
%   kk: maximum truncation rank
%   lambda: kk-by-1 vector, where lambda(i) in the prior N(0, lambda(i)^(-1) N)
%
% Outputs:
%   X: n-by-kk matrix. The i-th column is the posterior mean
%      obtained by truncating the KL expansion to the first i modes.
%   V: n-by-kk matrix containing the first kk leading eigenvectors of N
%   s: kk-by-1 vector containing the corresponding leading eigenvalues
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 26, Oct, 2026.
% 

[m, n] = size(A);
if length(b) ~= m
    error('The dimension of b is inconsistent with A.')
end
if size(M,1) ~= m || size(M,2) ~= m
    error('The dimension of M is inconsistent with A.')
end

if kk > n
    error('kk must not exceed the parameter dimension n.')
end

lambda = lambda(:);

if length(lambda) < kk
    error('lambda must contain at least kk components.')
end

% Compute the first kk leading eigenpairs of N
opts.issym  = true;
opts.isreal = true;
opts.tol    = 1e-12;
opts.maxit  = 5000;

if isa(N, 'function_handle')
    [V, D] = eigs(@(v) N(v), n, kk, 'largestreal', opts);
else
    [V, D] = eigs(N, kk, 'largestreal', opts);
end

s = real(diag(D));

% Sort the eigenvalues in decreasing order
[s, idx] = sort(s, 'descend');
V = real(V(:,idx));

% Remove tiny negative eigenvalues caused by roundoff
s(s < 0) = 0;

% Form the KL basis L = V * diag(sqrt(s))
L = V .* sqrt(s)';

AL = A * L;

% Apply Gamma^{-1}; do not explicitly form inv(M)
Minv_AL = M \ AL;
Minv_b  = M \ b;

% Reduced likelihood Hessian and right-hand side
H = AL' * Minv_AL;
g = AL' * Minv_b;

% Symmetrize to remove small roundoff errors
H = (H + H') / 2;

% Compute the posterior mean for each truncation rank
X = zeros(n, kk);

for i = 1:kk
    Hi = H(1:i, 1:i);
    gi = g(1:i);
    zi = (Hi + lambda(i) * eye(i)) \ gi;
    X(:,i) = L(:,1:i) * zi;
end





J = zeros(kk,1);
condP = zeros(kk,1);

for i = 1:kk

    Hi = H(1:i,1:i);
    gi = g(1:i);

    Pi = Hi + lambda(i)*eye(i);
    Pi = 0.5*(Pi + Pi');

    zi = Pi \ gi;

    X(:,i) = L(:,1:i)*zi;

    % Condition number of reduced posterior precision
    condP(i) = cond(Pi);

    % Bayesian/Tikhonov objective
    ri = A*X(:,i) - b;

    J(i) = ri'*(M\ri) + lambda(i)*(zi'*zi);

end

figure;
semilogy(1:kk,J,'o-');
xlabel('KLE rank');
ylabel('Objective function');
grid on;
grid minor;

figure;
semilogy(1:kk,condP,'o-');
xlabel('KLE rank');
ylabel('cond(H_k+\lambda I)');
grid on;
grid minor;


end


