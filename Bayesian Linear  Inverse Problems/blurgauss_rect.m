function [A, b, x, ProbInfo] = blurgauss_rect(opts)
% Construct a 2D image deblurring test problem on [0,1]^2:
%   latent image x on an N-by-N grid,
%   blurred observation b on an M-by-M grid, A in R^(M^2 x N^2).
% The forward model is
%   y(s1,s2) = \int_{[0,1]^2} exp(-((s1-t1)^2 + (s2-t2)^2)/t_blur) x(t1,t2) dt1 dt2
% The true image x is sampled from a separable Matérn Gaussian prior:
%   Cov(vec(X)) = K1 \kron K1,
% where K1 is a 1D Matérn covariance matrix on the N-point latent grid.
%
% Input:
%   opts : struct with optional fields
%       .N            latent grid size, default 256
%       .M            observation grid size, default 128
%       .nu           Matérn smoothness, default 3
%       .rho          Matérn correlation length, default 0.1
%       .sigma        Matérn marginal std, default 1
%       .t_blur       Gaussian PSF blur parameter, default 0.02
%       .noise_std    std of additive Gaussian noise, default 0
%       .seed         RNG seed, default 2026
%       .truncate_tol threshold for sparsifying 1D Gaussian kernel,
%                     default 0 (no truncation, exact dense Kronecker)
%
% Output:
%   A        : M^2-by-N^2 matrix (sparse if truncate_tol > 0, else full)
%   b        : M^2-by-1 blurred data vector
%   x        : N^2-by-1 true image vector
%   ProbInfo : struct with metadata
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

if nargin < 1
    opts = struct();
end

% -------- defaults --------
if ~isfield(opts, 'N'),            opts.N = 256;        end
if ~isfield(opts, 'M'),            opts.M = 128;        end
if ~isfield(opts, 'nu'),           opts.nu = 3;         end
if ~isfield(opts, 'rho'),          opts.rho = 0.1;      end
if ~isfield(opts, 'sigma'),        opts.sigma = 1;      end
if ~isfield(opts, 't_blur'),       opts.t_blur = 0.02;  end
if ~isfield(opts, 'noise_std'),    opts.noise_std = 0;  end
if ~isfield(opts, 'seed'),         opts.seed = 2026;    end
if ~isfield(opts, 'truncate_tol'), opts.truncate_tol = 0; end

N = opts.N;
M = opts.M;
n = N^2;
m = M^2;

% rng(opts.seed);

% -------- grids: use cell centers on [0,1]^2 --------
% latent grid: N x N
t = ((1:N) - 0.5) / N;   % cell centers
h = 1 / N;               % quadrature weight in each dimension

% observation grid: M x M
s = ((1:M) - 0.5) / M;

% % Sample true image from separable Matérn prior
% K1 = matern1d_cov(t, opts.nu, opts.rho, opts.sigma);
% K1 = 0.5 * (K1 + K1');
% K1 = K1 + 1e-10 * eye(N);   % numerical safeguard
% L1 = chol(K1, 'lower');

% % If Z has iid N(0,1), then X = L1 * Z * L1' has vec(X) ~ N(0, K1 \kron K1)
% Z = randn(N, N);
% Xtrue = L1 * Z * L1';
% x = Xtrue(:);

x_low  = 0;
x_high = 1;

[x, K1] = matern2d_sep_sample(x_low, x_high, N, opts.nu, opts.rho, opts.sigma, opts.seed);

% Build Gaussian PSF operator A = h^2 (G1 kron G1)
G1 = gaussian_1d_cross(s, t, opts.t_blur);

% Optional sparsification by thresholding small entries in G1
if opts.truncate_tol > 0
    mx = max(G1(:));
    G1(G1 < opts.truncate_tol * mx) = 0;
    G1 = sparse(G1);
end

A = (h^2) * kron(G1, G1);

% Generate blurred data
b = A * x;
if opts.noise_std > 0
    b = b + opts.noise_std * randn(m,1);
end

% Pack metadata
ProbInfo = struct();
ProbInfo.problemType = 'deblurring_rectangular';
ProbInfo.xType = 'image2D';
ProbInfo.bType = 'image2D';
ProbInfo.xSize = [N, N];
ProbInfo.bSize = [M, M];
ProbInfo.n = n;
ProbInfo.m = m;
ProbInfo.domain = [0, 1; 0, 1];
ProbInfo.prior = 'separable_Matern';
ProbInfo.nu = opts.nu;
ProbInfo.rho = opts.rho;
ProbInfo.sigma = opts.sigma;
ProbInfo.PSF = 'Gaussian';
ProbInfo.t_blur = opts.t_blur;
ProbInfo.noise_std = opts.noise_std;
ProbInfo.latent_grid = t;
ProbInfo.obs_grid = s;
ProbInfo.quadrature_weight = h^2;
ProbInfo.truncate_tol = opts.truncate_tol;
ProbInfo.G1 = G1;
ProbInfo.K1 = K1;
ProbInfo.h  = h;
end


%%-------------------------------------------------------------------
function G = gaussian_1d_cross(s, t, t_blur)
% Cross-kernel matrix G(i,j) = exp(-(s_i - t_j)^2 / t_blur)
s = s(:);
t = t(:).';
G = exp(-((s - t).^2) / t_blur);
end