function dF = forstner_post(lambda, Sigma, G, Gamma, Vk, Bk)
% Compute d_F(C_lambda, C_lambda^(k)) WITHOUT explicitly forming
% the ill-conditioned covariance matrices.
%
% Uses the identity
%   d_F(C, Chat) = d_F(H + lambda Sigma^{-1}, Hhat + lambda Sigma^{-1})
% and then a congruence transform with Sigma = L L'.

    % dimensions
    if ~isa(G, 'function_handle')
        [m, n] = size(G);
    else
        [m, n] = sizemm(G);   % only if your G supports this safely
    end

    % explicit Sigma and G
    Sig = materialize_sigma(Sigma, n);
    Gmat = materialize_G(G);

    % Cholesky factor of Sigma
    Sig = 0.5 * (Sig + Sig');
    L = chol(Sig, 'lower');

    % Exact precision-side matrix:
    % P = L' H L + lambda I
    %   = lambda I + L' G' Gamma^{-1} G L
    GL = Gmat * L;
    P = lambda * eye(n) + GL' * (Gamma \ GL);
    P = 0.5 * (P + P');

    % Approximate precision-side matrix:
    % Q = lambda I + L' Hhat_k L
    %   = lambda I + U_k (Bk'Bk) U_k'
    Uk = L' * (Gmat' * Vk);       % n-by-k
    Jk = Bk' * Bk;                % k-by-k
    Q = lambda * eye(n) + Uk * Jk * Uk';
    Q = 0.5 * (Q + Q');

    % Fostner distance
    dF = forstner_dist(P, Q);
end