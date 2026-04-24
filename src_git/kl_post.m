function D = kl_post(lambda, Sigma, G, Gamma, y, Vk, Bk)
% Compute KL divergence:
%   D_KL( approx || exact )
% WITHOUT forming covariance matrices.

    % dimensions
    if ~isa(G,'function_handle')
        [m,n] = size(G);
    else
        [m,n] = sizemm(G);
    end

    % materialize
    Sig = materialize_sigma(Sigma,n);
    Gmat = materialize_G(G);

    % Cholesky
    Sig = 0.5*(Sig+Sig');
    L = chol(Sig,'lower');

    % ---- build precision matrices ----
    GL = Gmat * L;

    P = lambda*eye(n) + GL'*(Gamma \ GL);   % exact precision
    P = 0.5*(P+P');

    Uk = L'*(Gmat'*Vk);
    Jk = Bk'*Bk;

    Phat = lambda*eye(n) + Uk*Jk*Uk';
    Phat = 0.5*(Phat+Phat');

    % ---- trace term ----
    % tr(P * Phat^{-1})
    R = chol(Phat,'lower');
    X = R \ (P / R');     % Phat^{-1/2} P Phat^{-1/2}
    trace_term = trace(X);

    % ---- logdet term ----
    logdetP = 2*sum(log(diag(chol(P))));
    logdetPhat = 2*sum(log(diag(R)));

    logdet_term = logdetP - logdetPhat;

    % ---- mean term ----
    rhs = Gmat'*(Gamma \ y);

    x_exact = P \ rhs;
    x_approx = Phat \ rhs;

    diff = x_exact - x_approx;

    quad = diff' * P * diff;

    % ---- KL ----
    D = 0.5*(trace_term - n - logdet_term + quad);

    % safeguard
    if D < 0 && abs(D) < 1e-10
        D = 0;
    end
end