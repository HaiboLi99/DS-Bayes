function [X, Vk, Bk, Lam, L_vals] = QGKB_HB(A, b, M, N, k)
% Hybrid Hierarchical Bayesian method based on Q-GKB.
% At each iteration k, lambda_k is defined by
%     lambda_k = argmin_{lambda>0} L^{(k)}(lambda),
% where L^{(k)} is the approximate negative log-marginal likelihood
% derived from the reduced matrix B_k.
%
% Inputs:
%   A: either (a) a full or sparse mxn matrix;
%             (b) a matrix function handle
%   b: right-hand side vector
%   M: covaraince matrix of noise e, e~N(0,M), symmetric positive definite and sparse
%   N: Gaussian prior of x, a matrix or a function handle
%   k: the maximum number of iterations 
% 
% Outputs:
%   X      : regularized solutions x_{lambda_k}^{(k)}
%   Vk     : basis vectors v_1,...,v_k in the quotient-space realization
%   Bk     : bidiagonal matrix at final step
%   Lam    : lambda_k at each iteration
%   L_vals : minimum value of reduced marginal likelihood at each iteration
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 09, April, 2026.
%

% M is diagonal, form M^{-1} directly
M_inv = sparse(diag(1./diag(M)));

X = [];
U = [];
V = [];
B = [];
Lam = zeros(k,1);
L_vals = zeros(k,1);
reorth = 2;

G = @(v) Gfun(v, A, N);

% fprintf('Start the QGKB-HB iteration ==============================\n');
h = waitbar(0, 'Beginning QGKB-HB iterations: please wait ...');

% ---- Initialization ----
beta1 = sqrt(b' * M_inv * b);
u = b / beta1;
U(:,1) = u;

s = M_inv * u;
alpha = sqrt(s' * G(s));
B(1,1) = alpha;

v = s / alpha;
V(:,1) = v;

for i = 1:k
    % -------- Q-GKB iteration -------------
    r = G(v) - alpha * u;

    if reorth == 1
        rr = M_inv * r;
        r = r - U(:,1:i) * (U(:,1:i)' * rr);
    elseif reorth == 2
        rr = M_inv * r;
        r = r - U(:,1:i) * (U(:,1:i)' * rr);
        rr = M_inv * r;
        r = r - U(:,1:i) * (U(:,1:i)' * rr);
    end

    beta = sqrt(r' * M_inv * r);
    if beta < 1e-10
        fprintf('[Breakdown], beta = %e, at step %d\n', beta, i);
        i = i-1;
        X = X(:,1:i);
        Vk = V(:,1:i);
        if i >= 1
            Bk = B(1:i+1,1:i);
            Lam = Lam(1:i);
            L_vals = L_vals(1:i);
        else
            Bk = [];
            Lam = [];
            L_vals = [];
        end
        break;
    end

    u = r / beta;
    U(:,i+1) = u;
    B(i+1,i) = beta;

    s = M_inv * u - beta * v;
    if reorth == 1
        ss = G(s);
        s = s - V(:,1:i) * (V(:,1:i)' * ss);
    elseif reorth == 2
        ss = G(s);
        s = s - V(:,1:i) * (V(:,1:i)' * ss);
        ss = G(s);
        s = s - V(:,1:i) * (V(:,1:i)' * ss);
    end

    alpha = sqrt(s' * G(s));
    if alpha < 1e-10
        fprintf('[Breakdown], alpha = %e, at step %d\n', alpha, i);
        i = i-1;
        X = X(:,1:i);
        Vk = V(:,1:i);
        if i >= 1
            Bk = B(1:i+1,1:i);
            Lam = Lam(1:i);
            L_vals = L_vals(1:i);
        else
            Bk = [];
            Lam = [];
            L_vals = [];
        end
        break;
    end

    v = s / alpha;
    V(:,i+1) = v;
    B(i+1,i+1) = alpha;

    % -------- Reduced HB step -------------
    Vk = V(:,1:i);
    Bk = B(1:i+1,1:i);
    b_proj = [beta1; zeros(i,1)];

    % Economy SVD of Bk: Bk = P*S*Q'
    [P, S, Q] = svd(Bk, 0);
    s_diag = diag(S);           % singular values s_j
    p1 = P(1,1:i)';             % first row of P, as column vector

    % Exact minimization of reduced marginal likelihood
    obj_fun = @(lam) HB_reduced_obj(lam, s_diag, p1, beta1);

    % Search interval for lambda > 0
    lam_low = 1e-6;
    lam_high = max(1e3, 10 * max(s_diag.^2 + 1e-10));

    [lambda_k, min_val] = fminbnd(obj_fun, lam_low, lam_high);

    Lam(i) = lambda_k;
    L_vals(i) = min_val;

    % Solve projected Tikhonov problem:
    % f = Q * diag(s_j/(s_j^2 + lambda_k)) * P' * b_proj
    b_hat = P' * b_proj;
    f = Q * ((s_diag .* b_hat) ./ (s_diag.^2 + lambda_k));

    z = Vk * f;
    X(:,i) = iso_embed(z, A, N);

    waitbar(i/k, h);
end

close(h);

end


%---------------------------------------------------------
function val = HB_reduced_obj(lambda, s_diag, p1, beta1)
% Reduced negative log-marginal likelihood (up to an additive constant):
%
%   L^{(k)}(lambda) = sum_j log(1+s_j^2/lambda)
%     + beta1^2*(1-sum_j[s_j^2/(s_j^2+lambda)*p1_j^2])
%
% lambda must be positive.

if lambda <= 0
    val = inf;
    return;
end

term1 = sum(log(1 + (s_diag.^2) / lambda));
term2 = beta1^2 * (1 - sum((s_diag.^2 ./ (s_diag.^2 + lambda)) .* (p1.^2)));

val = term1 + term2;
end