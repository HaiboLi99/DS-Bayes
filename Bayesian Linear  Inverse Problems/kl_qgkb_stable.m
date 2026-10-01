function [DKL_vals, bnd_vals, zeta, gamma] = ...
    kl_qgkb_stable(ref, Xmax, Vmax, Bmax, Lam)
% Compute D_KL( pi_hat_k || pi_lambda_k ) for the Q-GKB posterior
% approximation in prior-whitened coordinates.
%
% The full posterior and the Q-GKB posterior use the SAME lambda_k
% at every iteration.
%
% Inputs:
%   ref   : reference structure from kl_reference
%   Xmax  : n-by-K matrix of posterior means computed by QGKB_HB;
%           Xmax(:,k) = x_hat_{lambda_k}^{(k)}
%   Vmax  : final Q-GKB data-space basis, m-by-K
%   Bmax  : final bidiagonal matrix, (K+1)-by-K
%   Lam   : K-by-1 vector containing lambda_k
%
% Outputs:
%   DKL_vals : actual KL divergence
%   bnd_vals : theoretical/reference-space upper bound
%   zeta     : zeta_k, k=0,...,K
%   gamma    : gamma_k, k=0,...,K
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 26, Oct, 2026.
%

K = length(Lam);

if size(Xmax,2) < K
    error('Xmax has fewer than K columns.')
end

if size(Vmax,2) < K
    error('Vmax has fewer than K columns.')
end

if size(Bmax,2) < K
    error('Bmax has fewer than K columns.')
end

r = ref.r;
H = ref.H;
F = ref.F;
f = ref.f;

sref = ref.s(:);

if any(sref <= 0)
    error('Reference eigenvalues must be strictly positive.')
end

beta1 = norm(ref.yw);

DKL_vals = zeros(K,1);
bnd_vals = zeros(K,1);


%%--------------------------------------------------------------
% Compute zeta_k and gamma_k
%
% zeta(1) = zeta_0
% gamma(1) = gamma_0
%
% The initial values are evaluated in the resolved reference space.

zeta  = zeros(K+1,1);
gamma2 = zeros(K+1,1);

zeta(1)  = trace(H);
gamma2(1) = norm(H,'fro')^2;

alpha = diag(Bmax);
beta_sub = diag(Bmax,-1);
% beta_sub(i) = beta_{i+1}


for k = 1:K

    ak = alpha(k);
    bkp1 = beta_sub(k);

    dk = ak^2 + bkp1^2;

    % zeta_k = zeta_{k-1} - (alpha_k^2 + beta_{k+1}^2)
    zeta(k+1) = zeta(k) - dk;

    if k == 1

        % gamma_1^2
        gamma2(k+1) = gamma2(k) - dk^2;

    else

        % beta_sub(k-1) = beta_k
        bk = beta_sub(k-1);

        gamma2(k+1) = gamma2(k) ...
            - 2*(ak*bk)^2 ...
            - dk^2;

    end

    % Safeguard only roundoff-size negative values
    tol = 1e3 * eps * max(gamma2(1),1);

    if gamma2(k+1) < 0 && abs(gamma2(k+1)) <= tol

        gamma2(k+1) = 0;

    elseif gamma2(k+1) < -tol

        warning('gamma_k^2 became significantly negative at k = %d.',k);

    end

end

gamma = sqrt(max(gamma2,0));


%%--------------------------------------------------------------
% KL divergence

alpha1 = Bmax(1,1);

for k = 1:K

    lambda = Lam(k);

    if lambda <= 0
        error('All lambda values must be positive.')
    end

    Vk = Vmax(:,1:k);
    Bk = Bmax(1:k+1,1:k);

    Jk = Bk' * Bk;
    Jk = 0.5 * (Jk + Jk');


    %%----------------------------------------------------------
    % Coordinates of Sigma^{1/2} G' V_k
    %
    % F = Gamma^{-1/2} G L,
    %
    % where L = V_ref diag(sqrt(s_ref)).
    %
    % Therefore
    %
    % U_k = L' G' V_k.

    if ref.gamma_is_diag

        Wk = ref.sqrtg .* Vk;

    else

        Wk = ref.Rgamma' * Vk;

    end

    Uk = F' * Wk;       % r-by-k


    %%----------------------------------------------------------
    % Exact posterior in prior-whitened coordinates

    P = H + lambda * eye(r);
    P = 0.5 * (P + P');

    m_exact = P \ f;

    logdetP = logdet_spd(P);
    logdetC_exact = -logdetP;


    %%----------------------------------------------------------
    % Q-GKB posterior mean
    %
    % IMPORTANT:
    % Use the posterior mean actually computed by QGKB_HB,
    % rather than recomputing it from B_k and V_k.
    %
    % x_k = L z_k,
    %
    % where
    %
    % L = V_ref diag(sqrt(s_ref)).
    %
    % Therefore
    %
    % z_k = diag(s_ref^{-1/2}) V_ref' x_k.

    xk = Xmax(:,k);

    coeff = ref.V' * xk;

    m_approx = coeff ./ sqrt(sref);


    %%----------------------------------------------------------
    % Check whether x_k is well represented in the reference space

    xk_proj = ref.V * coeff;

    proj_err = norm(xk - xk_proj) / max(norm(xk),eps);

    if proj_err > 1e-6

        warning(['Q-GKB posterior mean at k = %d is not well ', ...
                 'resolved by the reference space: relative ', ...
                 'projection error = %.3e.'], ...
                 k, proj_err);

    end


    %%----------------------------------------------------------
    % Q-GKB posterior covariance in whitened coordinates
    %
    % C_hat =
    %
    % lambda^{-1} I
    % - lambda^{-1} U_k
    %   (lambda I + J_k)^{-1} J_k U_k'.

    T = (lambda * eye(k) + Jk) \ (Jk * Uk');

    C_approx = (1/lambda) * ...
        (eye(r) - Uk*T);

    C_approx = 0.5 * ...
        (C_approx + C_approx');


    %%----------------------------------------------------------
    % KL divergence
    %
    % D_KL(N(m_hat,C_hat) || N(m,C))
    %
    % = 1/2 [
    %
    %     tr(P C_hat) - r
    %     + log det(C) - log det(C_hat)
    %     + (m-m_hat)' P (m-m_hat)
    %
    %   ].

    trace_term = trace(P * C_approx);

    logdetC_approx = logdet_spd(C_approx);

    diff = m_exact - m_approx;

    quad_term = diff' * P * diff;

    D = 0.5 * ...
        (trace_term - r ...
        + logdetC_exact ...
        - logdetC_approx ...
        + quad_term);

    D = real(D);


    %%----------------------------------------------------------
    % A genuine KL divergence cannot be negative.
    %
    % Only truncate a negative value when it is at roundoff level.

    if D < 0

        tolD = 1e3 * eps * max(1, ...
            abs(trace_term) ...
            + abs(logdetC_exact) ...
            + abs(logdetC_approx) ...
            + abs(quad_term));

        if abs(D) <= tolD

            D = 0;

        else

            warning(['Computed KL divergence is significantly ', ...
                     'negative at k = %d: %e'], ...
                     k,D);

        end

    end

    DKL_vals(k) = D;


    %%----------------------------------------------------------
    % Theoretical/reference-space upper bound

    bnd_vals(k) = (1/(2*lambda)) * ...
        (zeta(k+1) ...
        + (alpha1^2 * beta1^2 * gamma(k+1)^2) ...
        / (lambda * (lambda + gamma(k+1))));

end

end


% -------------------------------------------------------------
function val = logdet_spd(A)
% Stable logarithm of the determinant of an SPD matrix.

A = 0.5 * (A + A');

[R,p] = chol(A,'lower');

if p ~= 0
    error(['Matrix is not numerically SPD. ', ...
           'Increase the reference dimension or check the model.'])
end

val = 2 * sum(log(diag(R)));

end