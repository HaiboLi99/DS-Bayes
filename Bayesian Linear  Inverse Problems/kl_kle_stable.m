function [DKL_vals, Dcov_vals, Dmean_vals] = ...
    kl_kle_stable(ref, Xmax, Vkle, skle, Lam)
% Compute the KL divergence between the lifted KLE posterior
% approximation and the full posterior.
%
% The strictly truncated KLE posterior is supported only on a
% k-dimensional subspace of R^n. Hence its KL divergence from the
% full-dimensional posterior is infinite for k < n.
%
% Here we use the full-dimensional lifted KLE approximation:
%   - update the first k KL modes using the data;
%   - keep the unresolved KL modes equal to their prior.
%
% The full posterior and the KLE posterior use the SAME lambda_k
% at every iteration.
%
% Inputs:
%   ref   : reference structure from kl_reference
%   Xmax  : n-by-K matrix of posterior means computed by KLE;
%           Xmax(:,k) is the posterior mean using the first k
%           KL modes
%   Vkle  : n-by-K matrix containing the first K KL eigenvectors
%           of Sigma
%   skle  : K-by-1 vector containing the corresponding eigenvalues
%           of Sigma
%   Lam   : K-by-1 vector containing lambda_k
%
% Outputs:
%   DKL_vals   : total KL divergence
%   Dcov_vals  : covariance contribution
%   Dmean_vals : posterior-mean contribution
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 26, Oct, 2026.
%

K = length(Lam);

r = ref.r;
H = ref.H;
f = ref.f;

sref = ref.s(:);
skle = skle(:);

if K > r
    error('Reference dimension must be at least the maximum KLE rank.')
end

if size(Xmax,2) < K
    error('Xmax has fewer than K columns.')
end

if size(Vkle,2) < K
    error('Vkle has fewer than K columns.')
end

if length(skle) < K
    error('skle has fewer than K components.')
end

if any(sref <= 0)
    error('Reference eigenvalues must be strictly positive.')
end

if any(skle(1:K) <= 0)
    error('KLE eigenvalues must be strictly positive.')
end


%%--------------------------------------------------------------
% Express the actual KLE basis in the reference whitened coordinates
%
% Reference:
%
%       L_ref = V_ref diag(sqrt(s_ref)).
%
% Actual KLE basis:
%
%       L_KLE = V_KLE diag(sqrt(s_KLE)).
%
% Hence
%
%       L_KLE = L_ref Q,
%
% where
%
%       Q = diag(s_ref^{-1/2})
%           V_ref' V_KLE diag(sqrt(s_KLE)).

Q = ref.V' * Vkle(:,1:K);

Q = Q .* sqrt(skle(1:K))';
Q = Q ./ sqrt(sref);


%%--------------------------------------------------------------
% Check that the KLE basis is resolved by the reference space
%
% Ideally Q'Q = I.

orth_err = norm(Q' * Q - eye(K),'fro');

fprintf('KLE whitened-basis orthogonality error = %.3e\n', ...
    orth_err);

if orth_err > 1e-6

    warning(['The KLE basis is not accurately represented in the ', ...
             'reference space: ||Q''Q-I||_F = %.3e. ', ...
             'Consider increasing rref.'], ...
             orth_err);

end


%%--------------------------------------------------------------

DKL_vals   = zeros(K,1);
Dcov_vals  = zeros(K,1);
Dmean_vals = zeros(K,1);

traceH = trace(H);


for k = 1:K

    lambda = Lam(k);

    if lambda <= 0
        error('All lambda values must be positive.')
    end


    %%----------------------------------------------------------
    % Full posterior in prior-whitened coordinates
    %
    %       C = (H + lambda I)^{-1},
    %
    %       m = (H + lambda I)^{-1} f.

    P = H + lambda * eye(r);
    P = 0.5 * (P + P');

    m_exact = P \ f;

    logdetP = logdet_spd(P);
    logdetC_exact = -logdetP;


    %%----------------------------------------------------------
    % Actual KLE subspace in the reference coordinates

    Qk = Q(:,1:k);

    Hk = Qk' * H * Qk;
    Hk = 0.5 * (Hk + Hk');

    Pk = Hk + lambda * eye(k);
    Pk = 0.5 * (Pk + Pk');


    %%----------------------------------------------------------
    % KLE posterior mean
    %
    % IMPORTANT:
    % Use the posterior mean actually computed by KLE.m,
    % rather than recomputing the mean from the reference basis.

    xk = Xmax(:,k);

    coeff = ref.V' * xk;

    m_approx = coeff ./ sqrt(sref);


    %%----------------------------------------------------------
    % Check whether x_k is resolved by the reference space

    xk_proj = ref.V * coeff;

    proj_err = norm(xk - xk_proj) / max(norm(xk),eps);

    if proj_err > 1e-6

        warning(['KLE posterior mean at k = %d is not well ', ...
                 'resolved by the reference space: relative ', ...
                 'projection error = %.3e.'], ...
                 k, proj_err);

    end


    %%----------------------------------------------------------
    % Optional consistency check
    %
    % The mean obtained from Xmax should agree with the one
    % reconstructed directly from the KLE basis.

    fk = Qk' * f;

    mk = Pk \ fk;

    m_check = Qk * mk;

    mean_err = norm(m_approx - m_check) / ...
        max(norm(m_check),eps);

    if mean_err > 1e-6

        warning(['KLE posterior mean inconsistency at k = %d: ', ...
                 'relative error = %.3e.'], ...
                 k, mean_err);

    end


    %%----------------------------------------------------------
    % Covariance contribution
    %
    % In the reference whitened coordinates, the lifted KLE
    % covariance is
    %
    %   C_KLE =
    %
    %       Qk (Hk + lambda I)^{-1} Qk'
    %
    %       + lambda^{-1} (I - Qk Qk').
    %
    % Since Qk is orthonormal,
    %
    %   tr(P*C_KLE) - r
    %
    %       = [tr(H) - tr(Hk)] / lambda.

    trace_minus_r = ...
        (traceH - trace(Hk)) / lambda;


    %%----------------------------------------------------------
    % Log determinant of the lifted KLE covariance
    %
    % Its eigenvalues consist of
    %
    %   eig((Hk + lambda I)^{-1})
    %
    % on the KLE subspace, and lambda^{-1} on its complement.

    logdetPk = logdet_spd(Pk);

    logdetC_approx = ...
        -logdetPk - (r-k)*log(lambda);


    %%----------------------------------------------------------
    % Covariance part of KL divergence

    Dcov = 0.5 * ...
        (trace_minus_r ...
        + logdetC_exact ...
        - logdetC_approx);

    Dcov = real(Dcov);


    %%----------------------------------------------------------
    % Mean contribution

    diff = m_exact - m_approx;

    quad_term = diff' * P * diff;

    Dmean = 0.5 * real(quad_term);


    %%----------------------------------------------------------
    % Total KL divergence

    D = real(Dcov + Dmean);


    %%----------------------------------------------------------
    % KL divergence cannot be negative except for roundoff

    if D < 0

        tolD = 1e3 * eps * ...
            max(1,abs(Dcov)+abs(Dmean));

        if abs(D) <= tolD

            D = 0;

        else

            warning(['Computed KLE KL divergence is significantly ', ...
                     'negative at k = %d: %e'], ...
                     k,D);

        end

    end


    %%----------------------------------------------------------

    DKL_vals(k)   = D;
    Dcov_vals(k)  = Dcov;
    Dmean_vals(k) = Dmean;

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