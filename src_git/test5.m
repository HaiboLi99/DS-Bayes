% Test for 2D X-ray computed tomography using parallel-beam geometry.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 14, April, 2026.
%

clear; clc;
close all;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2026);

%------------ Generate CT test problem -----------------
opts = struct();
opts.phantomImage = 'smooth';
opts.sm = true;                 % sparse matrix
opts.angles = 0:2:179;          % projection angles in degrees
opts.p = [];                    % default: round(sqrt(2)*N)
opts.d = [];                    % default: p-1
opts.isDisp = 0;

N1 = 256;
[A, b_true, x_true, ProbInfo] = CT_paral(opts);

% Add white Gaussian noise
nel = 2e-3;                     
[e, M] = genNoise(b_true, nel, 'white');
b = b_true + e;

NN = ProbInfo.xSize(1);
p = ProbInfo.bSize(1);
angles = ProbInfo.angles;
nAngles = length(angles);

Xtrue = reshape(x_true, NN, NN);
Bobs  = reshape(b, p, nAngles);

% ----------- Plot true image and noisy sinogram ----------------
fig = figure('Units','pixels', 'Position',[100, 80, 1400, 600]);
t = tiledlayout(1,2,'TileSpacing','normal','Padding','compact');
nexttile
imagesc([-1,1], [-1,1], Xtrue)
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','Interpreter','latex','FontSize',20);
ylabel('$t_{2}$','Interpreter','latex','FontSize',20);
title('True image','FontSize',18,'FontWeight','normal')
nexttile
imagesc(angles, 1:p, Bobs)
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('Projection  angle','FontSize',16);
ylabel('Ray  index','FontSize',16);
title('Noisy data','FontSize',18,'FontWeight','normal')


%%-------------- set for the computation -----------------------
nu    = 5/2;
rho   = 2;
sigma = 1;
N = @(v) matern2d_covar(v, -1, 1, NN, nu, rho, 1);
reorth = 2;
kk = 300;
   
[X1, V1, B1, Lam1, L_vals] = QGKB_HB(A, b, M, N, kk);
k = size(X1,2);

er1 = zeros(k,1);  % errors of solution
xn = norm(x_true);
for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
end

% convergence history 
fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, er1, '-o','Color',[0.0000,0.4470,0.7410],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.0000,0.4470,0.7410],'LineWidth',1.5);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
title('Error of iterated posterior mean','fontsize',18,'FontWeight','normal')


% ---- prior and reconstructed image ---------------------
opts1 = struct();
opts1.x_low = -1;
opts1.x_high = 1;
opts1.N = NN;
opts1.nu = nu;
opts1.rho = rho;
opts1.sigma = sigma;
lambda = Lam1(k);
Vk = V1(:,1:k);
Bk = B1(1:k+1,1:k);

[mhat, diagChat] = approx_post_mean_var_fft(lambda, A, M, b, Bk, Vk, opts1);
mstd = sqrt(mean(diagChat));
fprintf('k = %d: Relative Error = %.4f, mstd = %.2e\n', k, er1(k), mstd);

X_deblur = reshape(X1(:,k), NN, NN);
X_deblur_var  = reshape(diagChat, NN, NN);
X_appr_post = reshape(mhat, NN, NN);

x_prior = matern_grf_2d(NN, nu, rho, sigma, 2024);
X_prior = reshape(x_prior, NN, NN);

fig = figure('Units','pixels', 'Position',[100, 80, 2000, 600]);
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile
imagesc([0,1], [0,1], X_prior)
axis image
set(gca,'YDir','normal')
colormap(parula)
cb = colorbar; 
cb.FontSize = 14; 
cb.Label.FontSize = 14; 
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Prior image','FontSize',18,'FontWeight','normal');
nexttile
imagesc([-1,1], [-1,1], X_appr_post);
axis image
set(gca,'YDir','normal')
colormap(parula)
cb = colorbar;
cb.FontSize = 16; 
cb.Label.FontSize = 16; 
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Reconstructed image','FontSize',18,'FontWeight','normal');
nexttile
imagesc([-1,1], [-1,1], X_deblur_var);
axis image
set(gca,'YDir','normal')
colormap(parula)
% colorbar
cb = colorbar;
cb.FontSize = 14; 
cb.Label.FontSize = 14; 
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Approx posterior variance','FontSize',18,'FontWeight','normal');



%%---------------------------------------------------------------
function img = matern_grf_2d(N, nu, rho, sigma, seed)
% Generate a 2D Gaussian random field with Matérn covariance
%
% Inputs:
%   N     : grid size (NxN)
%   nu    : smoothness parameter
%   rho   : correlation length
%   sigma : variance scale
%
% Output:
%   img   : N-by-N random field

% frequency grid
[k1, k2] = meshgrid( ...
    [0:(N/2-1), -N/2:-1], ...
    [0:(N/2-1), -N/2:-1]);

k_sq = k1.^2 + k2.^2;

% Matérn spectral density
kappa = sqrt(2*nu) / rho;
S = (kappa^2 + (2*pi)^2 * k_sq).^(-(nu + 1));

% normalize variance
S = sigma^2 * S / max(S(:));

% sample in Fourier domain
rng(seed); 
Z = randn(N,N) + 1i*randn(N,N);
F = sqrt(S) .* Z;

% inverse FFT
img = real(ifft2(F));

% normalize
img = img - mean(img(:));
img = img / std(img(:));

end

