function ref = kl_reference(G, Sigma, Gamma, y, rref)
% KL_REFERENCE
% Construct a numerically stable reference representation of the
% full posterior in prior-whitened coordinates.
%
% Let
%
%       Sigma V = V diag(s),
%
% and define
%
%       L = V diag(sqrt(s)).
%
% In the prior-whitened coordinates x = L z, the posterior precision
% matrix is
%
%       H + lambda I,
%
% where
%
%       H = L' G' Gamma^{-1} G L.
%
% This function constructs H and the corresponding right-hand side
%
%       f = L' G' Gamma^{-1} y
%
% in a numerically resolved subspace of the prior covariance.
%
% Inputs:
%   G      : forward matrix or function handle
%   Sigma  : prior covariance matrix or function handle
%   Gamma  : noise covariance matrix
%   y      : observation vector
%   rref   : dimension of the numerical reference space
%
% Output:
%   ref    : structure containing the prior-whitened reference problem
%
% Haibo Li, School of Mathematics and Statistics, HUST
%

%% dimensions

if ~isa(G, 'function_handle')
    [m, n] = size(G);
else
    [m, n] = sizemm(G);
end

if length(y) ~= m
    error('Dimension of y is inconsistent with G.')
end

if rref > n
    rref = n;
end


%% Compute the leading eigenpairs of Sigma

opts.issym  = true;
opts.isreal = true;
opts.tol    = 1e-12;
opts.maxit  = 5000;
opts.disp   = 0;

% IMPORTANT:
% Do not use an even vector such as ones(n,1) as the starting vector.
% For a symmetric covariance kernel this may restrict the Krylov
% process to an invariant even subspace and miss odd eigenfunctions.
%
% Use a reproducible generic vector containing both even and odd parts,
% without changing the random state of the calling program.

rng_state = rng;
rng(12345);
v0 = randn(n,1);
rng(rng_state);

opts.v0 = v0 / norm(v0);


if ~isa(Sigma, 'function_handle')

    S = 0.5 * (Sigma + Sigma');

    if rref < n

        [V, D] = eigs(S, rref, 'largestreal', opts);
        s = diag(D);

    else

        [V, D] = eig(full(S), 'vector');
        s = D;

    end

else

    if rref >= n
        error(['For a function-handle Sigma, rref must be smaller ', ...
               'than the parameter dimension n.'])
    end

    Sfun = @(v) real(Sigma(v));

    [V, D] = eigs(Sfun, n, rref, 'largestreal', opts);
    s = diag(D);

end


%% Sort eigenpairs in decreasing order

s = real(s);
V = real(V);

[s, idx] = sort(s, 'descend');
V = V(:,idx);


%% Remove only roundoff-size negative eigenvalues

tol_neg = 1e-13 * max(abs(s(1)),1);

if any(s < -tol_neg)
    error('Sigma has significantly negative computed eigenvalues.')
end

s(s < 0) = 0;


%% Construct the resolved square root of Sigma
%
%       L = V diag(sqrt(s))

L = V .* sqrt(s(:))';


%% Compute G*L

if ~isa(G, 'function_handle')

    GL = G * L;

else

    GL = zeros(m, rref);

    for j = 1:rref
        GL(:,j) = G(L(:,j), 'notransp');
    end

end


%% Whiten the noise covariance Gamma

Gamma = 0.5 * (Gamma + Gamma');

if isdiag(Gamma)

    g = diag(Gamma);

    if any(g <= 0)
        error('Gamma must be positive definite.')
    end

    sqrtg = sqrt(g);

    % F = Gamma^{-1/2} G L
    F = GL ./ sqrtg;

    % yw = Gamma^{-1/2} y
    yw = y ./ sqrtg;

    ref.gamma_is_diag = true;
    ref.sqrtg = sqrtg;
    ref.Rgamma = [];

else

    Rgamma = chol(Gamma, 'lower');

    % Gamma = Rgamma * Rgamma'
    % Hence Gamma^{-1/2} can be represented by Rgamma^{-1}

    F  = Rgamma \ GL;
    yw = Rgamma \ y;

    ref.gamma_is_diag = false;
    ref.sqrtg = [];
    ref.Rgamma = Rgamma;

end


%% Prior-whitened Hessian and right-hand side
%
%       H = L' G' Gamma^{-1} G L = F'F
%
%       f = L' G' Gamma^{-1} y   = F'yw

H = F' * F;
H = 0.5 * (H + H');

f = F' * yw;


%% Store quantities

ref.V = V;
ref.s = s;
ref.L = L;

ref.F = F;
ref.H = H;
ref.f = f;
ref.yw = yw;

ref.r = rref;
ref.n = n;
ref.m = m;

ref.prior_ratio = s(end) / s(1);

end