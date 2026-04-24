% Test for 2D image deblurring using Gaussian prior 
% with separable Matern kernel.
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
% 

clear, clc;
close all;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2026);  

opts = struct();
opts.N = 256;
opts.M = 128;
opts.nu = 3;
opts.rho = 0.1;
opts.sigma = 1;
opts.t_blur = 0.01;      
opts.noise_std = 0;    
opts.seed = 2026;
opts.truncate_tol = 1e-8;   

[A, b_true, x_true, ProbInfo] = blurgauss_rect(opts);
sigma = opts.sigma;
nel = 1e-2; 
[e, M] = genNoise(b_true, nel, 'white'); 
b = b_true + e;

NN = ProbInfo.xSize(1);
MM = ProbInfo.bSize(1);

% reshape to images
Xtrue = reshape(x_true, NN, NN);
Bobs  = reshape(b, MM, MM);

% set for the computation
N = @(v) matern2d_sep_covar(v, 0, 1, NN, opts.nu, opts.rho, 1);
reorth = 2;
kk = 250;
   
[X1, V1, B1, Lam1, L_vals] = QGKB_HB(A, b, M, N, kk);
sigma_comp = 1.0 ./ sqrt(Lam1);
k = size(X1,2);

er1 = zeros(k,1);  % errors of solution
er2 = zeros(k,1);  % errors of lambda
xn = norm(x_true);
for i =1:k
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = abs(sigma_comp(i)-sigma) / sigma;
end


% -------- plot true/blurred images --------------------------
fig = figure('Units','pixels', 'Position',[100, 80, 1400, 600]);
t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile
imagesc([0,1], [0,1], Xtrue)
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('True image','FontSize',18,'FontWeight','normal')
nexttile
imagesc([0,1], [0,1], Bobs)
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Blurred image','FontSize',18,'FontWeight','normal')


% -------- convergence history ----------------------------------
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

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, er2, '->','Color',[1,0.47,0.1],'MarkerIndices',1:9:k,...
    'MarkerSize',6,'MarkerFaceColor',[1.0,0.47,0.1],'LineWidth',1.5);
% legend('QGKB\_hyb', 'Location', 'northeast','fontsize',15);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
title('Error of iterated hyperparameter','fontsize',18,'FontWeight','normal');


% ---- prior and deblurred image ------------------------------------------
[x_prior, ~] = matern2d_sep_sample(0, 1, NN, opts.nu, opts.rho, opts.sigma, 2024);
X_prior = reshape(x_prior, NN, NN);
X_deblur = reshape(X1(:,k), NN, NN);
lambda = Lam1(k);
Vk = V1(:,1:k);
Bk = B1(1:k+1,1:k);
[mhat, diagChat] = approx_post_mean_var(lambda, A, M, b, Bk, Vk, ProbInfo);
X_deblur_var  = reshape(diagChat, NN, NN);

fig = figure('Units','pixels', 'Position',[100, 80, 2000, 600]);
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile
imagesc([0,1], [0,1], X_prior)
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Prior image','FontSize',18,'FontWeight','normal');
nexttile
imagesc([0,1], [0,1], X_deblur);
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Deblurred image','FontSize',18,'FontWeight','normal');
nexttile
imagesc([0,1], [0,1], X_deblur_var);
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Approx posterior variance','FontSize',18,'FontWeight','normal');


%%---- compact exact and approximate mean/covariance -----------
sigma_eps = sqrt(M(1,1));
[m_post, diagC_post, ~, ~] = exact_post_mean_var(b, lambda, sigma_eps, ProbInfo);
X_post = reshape(m_post, NN, NN);
X_err  = reshape(diagC_post, NN, NN);

fig = figure('Units','pixels', 'Position',[100, 80, 1400, 600]);
t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile
imagesc([0,1], [0,1], X_post)
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Posterior mean','FontSize',18,'FontWeight','normal');
nexttile
imagesc([0,1], [0,1], X_err);
axis image
set(gca,'YDir','normal')
colormap(parula)
colorbar
set(gca, 'FontSize', 12);
xlabel('$t_{1}$','interpreter','latex','fontsize',20); 
ylabel('$t_{2}$','interpreter','latex','fontsize',20);
title('Exact posterior variance','FontSize',18,'FontWeight','normal');


%---plot generalized eigenvalues of (M,Gamma) and projections---
[coeff, mu, W] = proj_coeff(A, b, M, N, 300, 1);

