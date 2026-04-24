function [mhat, diagChat] = approx_post_mean_var_fft(lambda, G, Gamma, y, Bk, Vk, opts)
% Approximate posterior (mean + diagonal) for 2D non-separable Matérn prior
% 
% Inputs:
%   lambda    : prior precision scaling
%   G         : forward matrix
%   Gamma  : diagonal SPD matrix in data space
%   y      : data vector, used to compute beta_1
%   Bk     : (k+1)-by-k bidiagonal matrix
%   Vk     : m-by-k basis matrix
%   opts  : struct for hyperparameters of Matern kernel
%
% Outputs:
%   m_hat     : N^2-by-1 approxiamte posterior mean
%   diagC_post : N^2-by-1 approximate posterior covariance diagonal
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 14, April, 2026.
% 

x_low = opts.x_low;
x_high = opts.x_high;
N = opts.N;
nu = opts.nu;
rho = opts.rho;
sigma = opts.sigma;

Lam = matern2d_covar_symbol(x_low, x_high, N, nu, rho, sigma);

% ---------- U = Sigma G' Vk ----------
GtVk = apply_Gt_cols(G, Vk);
U = Sigma_apply_fft(GtVk, Lam);

% ---------- low-rank covariance correction ----------
Jk = Bk' * Bk;
Jk = 0.5 * (Jk + Jk');

Kmat = (Jk + lambda * eye(size(Jk))) \ Jk;
Kmat = 0.5 * (Kmat + Kmat');

% Robust computation of diag(U*Kmat*U')
[Q, D] = eig(Kmat);
d = max(real(diag(D)), 0);
R = Q * diag(sqrt(d));

UR = U * R;
diag_lowrank = sum(UR.^2, 2);

% ---------- diag(Sigma) ----------
% c0 = real(ifft2(Lam));
% diagSigma = c0(1,1) * ones(N^2,1);
diagSigma = (sigma^2) * ones(N^2,1);

% ---------- diag posterior ----------
diagChat = (1/lambda) * (diagSigma - diag_lowrank);
% diagChat = max(real(diagChat), 0);
diagChat = abs(real(diagChat));

% ---------- mean ----------
g = diag(Gamma);
beta1 = sqrt(sum((y.^2) ./ g));

rhs = Bk' * [beta1; zeros(size(Bk,1)-1,1)];
coeff = (Jk + lambda * eye(size(Jk))) \ rhs;

mhat = Sigma_apply_fft(GtVk * coeff, Lam);

end


%%-------------------------------------------------------------------
function Lam = matern2d_covar_symbol(x_low, x_high, N, nu, rho, sigma)
% Precompute the FFT symbol for the 2D Matérn covariance operator.

L = x_high - x_low;     % physical domain length

% Frequency grid on [x_low, x_high]^2
% Angular frequencies: omega = 2*pi*k / L
% k = [0:(N/2-1), -N/2:-1];
% [k1, k2] = meshgrid(k, k);

% omega1 = 2*pi*k1 / L;
% omega2 = 2*pi*k2 / L;
% omega_sq = omega1.^2 + omega2.^2;

% using padding strategy for FFT
N_pad = 2 * N;
L_pad = 2 * L; % 物理长度也需同步翻倍以保持频率分辨率一致

% 频率网格基于 2N
k = [0:(N_pad/2-1), -N_pad/2:-1];
[k1, k2] = meshgrid(k, k);

omega1 = 2*pi*k1 / L_pad;
omega2 = 2*pi*k2 / L_pad;
omega_sq = omega1.^2 + omega2.^2;



kappa = sqrt(2*nu) / rho;
Lam = (kappa^2 + omega_sq).^(-(nu + 1));

% Normalize so that the variance at a grid point is sigma^2.
% If c = ifft2(Lam), then c(1,1) is the variance of the induced
% periodic covariance kernel on the grid.
c0 = real(ifft2(Lam));
Lam = (sigma^2 / c0(1,1)) * Lam;

end


%%----------------------------------------------
% function Y = Sigma_apply_fft(X, Lam)
% % Apply Sigma using FFT columnwise

% N = size(Lam,1);
% k = size(X,2);
% Y = zeros(size(X));

% for j = 1:k
%     Xt = reshape(X(:,j), N, N);
%     Yt = real(ifft2(Lam .* fft2(Xt)));
%     Y(:,j) = Yt(:);
% end

% end

function Y = Sigma_apply_fft(X, Lam)
% X 是 N^2 x k 的矩阵
% Lam 是 2N x 2N 的矩阵

N_orig = sqrt(size(X, 1)); 
N_pad = size(Lam, 1); % 应该是 2*N_orig
k = size(X, 2);
Y = zeros(size(X));

for j = 1:k
    Xt_orig = reshape(X(:,j), N_orig, N_orig);
    
    % 1. Zero-padding: 将 N x N 扩展到 2N x 2N
    Xt_pad = zeros(N_pad, N_pad);
    Xt_pad(1:N_orig, 1:N_orig) = Xt_orig;
    
    % 2. 频域相乘
    Yt_pad = real(ifft2(Lam .* fft2(Xt_pad)));
    
    % 3. Cropping: 只保留左上角的 N x N 部分
    % （注意：取决于你的核函数中心，有时取中心区域，但在循环卷积下通常取左上角）
    Yt_orig = Yt_pad(1:N_orig, 1:N_orig);
    
    Y(:,j) = Yt_orig(:);
end
end