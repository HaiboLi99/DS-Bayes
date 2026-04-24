function y = matern2d_covar(v, x_low, x_high, N, nu, rho, sigma)
% Apply the discrete 2D isotropic Matérn covariance operator on
% [x_low, x_high]^2 using an N-by-N uniform grid in each dimension. 
% Compute y = Sigma * v,
% where Sigma is the N^2-by-N^2 covariance matrix induced by the
% 2D Matérn kernel, implemented in matrix-free form via FFT.
%
% Inputs:
%   v       : vector of length N^2
%   x_low   : left endpoint of the interval
%   x_high  : right endpoint of the interval
%   N       : number of uniform grid points in each dimension
%   nu      : Matérn smoothness parameter
%   rho     : correlation length
%   sigma   : marginal standard deviation
%
% Output:
%   y       : Sigma * v
%
% Notes:
%   1. This implementation uses a periodic/circulant approximation.
%   2. The operator is not formed explicitly.
%   3. The scaling is chosen so that the variance is approximately sigma^2.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 14, April, 2026.
% 

if numel(v) ~= N^2
    error('Input vector v must have length N^2.');
end

Lam = matern2d_covar_symbol(x_low, x_high, N, nu, rho, sigma);

% Apply covariance operator
% X = reshape(v, N, N);
% Y = real(ifft2(Lam .* fft2(X)));
% y = Y(:);

% use zero padding
X_orig = reshape(v, N, N);
X_pad = zeros(2*N, 2*N);
X_pad(1:N, 1:N) = X_orig;
Y_pad = real(ifft2(Lam .* fft2(X_pad)));
Y_orig = Y_pad(1:N, 1:N);
y = Y_orig(:);

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