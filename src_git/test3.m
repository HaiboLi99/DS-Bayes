% Compare the Q-GKB with direct LIS as k increase, 
% for the 2D deblurring problem,
% where we use 2D separable Matern kernel.
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 16, April, 2026.
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
nel = 1e-2; 
[e, M] = genNoise(b_true, nel, 'white'); 
b = b_true + e;
NN = ProbInfo.xSize(1);
xn = norm(x_true);
N = @(v) matern2d_sep_covar(v, 0, 1, NN, opts.nu, opts.rho, 1);
kk = 250;

[X1, V1, B1, Lam1, L_vals] = QGKB_HB(A, b, M, N, kk);
k = size(X1,2);
[X2, postVar2, mstd2] = LIS_est_separ1(A, b, M, Lam1, k, ProbInfo);

er1 = zeros(k,1);
er2 = zeros(k,1);
mstd1 = zeros(k,1);

for i = 1:k 
    fprintf('Running test: the %d-th loop -------\n', i);
    lambda = Lam1(i);
    Vk = V1(:,1:i);
    Bk = B1(1:i+1,1:i);

    [~, diagChat] = approx_post_mean_var(lambda, A, M, b, Bk, Vk, ProbInfo);
    er1(i) = norm(x_true-X1(:,i)) / xn;
    er2(i) = norm(x_true-X2(:,i)) / xn;
    mstd1(i) = sqrt(mean(diagChat));
end


%-------- convergence history ----------------------------------
fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, er1, '-s','Color',[0.0000,0.2470,0.8410],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.0000,0.2470,0.8410],'LineWidth',1.5);
hold on;
semilogy(1:k, er2, '-^','Color',[0.8500 0.1250 0.0980],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.1250 0.0980],'LineWidth',1.5);
set(gca, 'FontSize', 12);
legend('QGKB','LIS','fontsize',16);
xlabel('Iteration','fontsize',16);
ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
title('Error of iterated posterior mean','fontsize',18,'FontWeight','normal')

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, mstd1, '-o','Color',[0.0000,0.4470,0.7410],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.0000,0.4470,0.7410],'LineWidth',1.5);
hold on;
semilogy(1:k, mstd2, '-v','Color',[0.8500 0.3250 0.0980],'MarkerIndices',1:9:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.8500 0.3250 0.0980],'LineWidth',1.5);
set(gca, 'FontSize', 12);
legend('QGKB','LIS','fontsize',16);
xlabel('Iteration','fontsize',16);
ylabel('Mstd','fontsize',16);
grid on;
grid minor;
title('Iterated mstd','fontsize',18,'FontWeight','normal')

