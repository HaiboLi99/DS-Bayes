% Test for solving 1-dim Fredholm integrale equation using Gaussian prior
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
% 

clear, clc;
close all;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
rng(2026);  

%%------------ test problem --------------
m = 3000;
n = 5000;
[A, b_true, x_true] = Fred_gauss(m, n, 'poly');    % x \in [-pi/2,pi/2]
a1 = -pi/2;  a2 = pi/2;  b1 = -pi/2;  b2 = pi/2;
[~, I1] = vec2fun(x_true, a1, a2);
[~, I2] = vec2fun(b_true, b1, b2);
sigma = 0.2;

% add noise
nel = 5e-3; % Noise level
[e, M] = genNoise(b_true, nel, 'white');
% [e, M] = genNoise(b_true, nel, 'nonwt');
b = b_true + e;

% set for the computation
l = 0.4;
N = @(v) gauss_covar(v, a1, a2, l);
reorth = 2;
kk = 25;
   
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


%%-------- plot -----------------------------------------
[coeff, mu, W] = proj_coeff(A, b, M, N, 50, 1);

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
plot(I1, x_true,'b-', 'LineWidth', 2.0);
legend('True sol','fontsize',16);
xlim([-pi/2 pi/2]);
xticks(-pi/2:pi/4:pi/2)
xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
set(gca, 'FontSize', 14);

darkGray = [0.25, 0.25, 0.25];
fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
scatter(I2, b, 2,'MarkerEdgeColor', darkGray, 'MarkerFaceColor', ...
    darkGray,'MarkerFaceAlpha', 0.6);
legend('Noisy data','fontsize',16);
xlim([-pi/2 pi/2]);
xticks(-pi/2:pi/4:pi/2)
xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
set(gca, 'FontSize', 14, 'LineWidth', 1.1, 'Box', 'on');


fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, er1, '-o','Color',[0.0000,0.4470,0.7410],'MarkerIndices',1:1:k,...
    'MarkerSize',5,'MarkerFaceColor',[0.0000,0.4470,0.7410],'LineWidth',1.5);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
title('Error of iterated posterior mean','fontsize',18)

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, er2, '->','Color',[1,0.47,0.1],'MarkerIndices',1:1:k,...
    'MarkerSize',6,'MarkerFaceColor',[1.0,0.47,0.1],'LineWidth',1.5);
% legend('QGKB\_hyb', 'Location', 'northeast','fontsize',15);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('Relative  error','fontsize',16);
grid on;
grid minor;
title('Error of iterated hyperparameter','fontsize',18)

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
plot(I1, x_true,'b-', 'LineWidth', 2.0);
hold on
plot(I1, X1(:,k),'m--', 'LineWidth', 2.0);
legend('True sol', 'Iter sol','fontsize',16);
xlim([-pi/2 pi/2]);
xticks(-pi/2:pi/4:pi/2)
xticklabels({'-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2'});
set(gca, 'FontSize', 14);


%%------------------------------------------------------------------------
% compute the Fostner distance and KL divergence w.r.t. k
% Lam1 = 1/sigma^2 * ones(k,1);

% [zeta0, gamma0] = zeta_gamma(A, M, N, V1, B1);
% [zeta1, gamma1] = qgkb_zeta_gamma(A, M, N, B1);

% figure;
% semilogy(1:k+1, zeta0, '-o','Color','b','MarkerIndices',1:1:k,...
%     'MarkerSize',6,'LineWidth',1.5);
% hold on
% semilogy(1:k+1, zeta1, '--x','Color','m','MarkerIndices',1:1:k,...
%     'MarkerSize',6,'MarkerFaceColor','m','LineWidth',1.5);
% legend('exact $\zeta_{k}$', 'recursive $\zeta_{k}$', 'Location', 'northeast','interpreter','latex','fontsize',20);
% xlabel('Iteration','fontsize',18);
% ylabel('Fostner distance','fontsize',18);
% grid on;
% grid minor;

% figure;
% semilogy(1:k+1, gamma0, '-o','Color','b','MarkerIndices',1:1:k,...
%     'MarkerSize',6,'MarkerFaceColor','b','LineWidth',1.5);
% hold on
% semilogy(1:k+1, gamma1, '-s','Color','m','MarkerIndices',1:1:k,...
%     'MarkerSize',6,'MarkerFaceColor','m','LineWidth',1.5);
% legend('exact $\gamma_{k}$', 'recursive $\gamma_{k}$', 'Location', 'northeast','interpreter','latex','fontsize',20);
% xlabel('Iteration','fontsize',18);
% ylabel('Fostner distance','fontsize',18);
% grid on;
% grid minor;


[dF, dF_bnd, zeta, gamma] = forstner_qgkb(A, N, M, b, V1, B1, Lam1);
[dKL, dKL_bnd, ~, ~] = kl_qgkb(A, N, M, b, V1, B1, Lam1);

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, dF, '-o','Color','[0.3010 0.7450 0.9330]','MarkerIndices',1:1:k,'MarkerSize',6,'MarkerFaceColor','[0.3010 0.7450 0.9330]','LineWidth',1.5);
hold on
semilogy(1:k, dF_bnd, '-s','Color','[0.8500 0.3250 0.0980]','MarkerIndices',1:1:k,'MarkerSize',6,'MarkerFaceColor','[0.8500 0.3250 0.0980]','LineWidth',1.5);
legend('$d_{F}(C_{\lambda},\widehat{C}_{\lambda}^{(k)})$', 'upper bound', 'Location', 'northeast','interpreter','latex','fontsize',20);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('Föstner  distance','fontsize',16);
grid on;
grid minor;

fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(1:k, dKL, '-o','Color','[0.1 0.2 0.8]','MarkerIndices',1:1:k,...
    'MarkerSize',6,'MarkerFaceColor','[0.1 0.2 0.8]','LineWidth',1.5);
hold on
semilogy(1:k, dKL_bnd, '-d','Color','[0.7350 0.0780 0.0840]','MarkerIndices',1:1:k,...
    'MarkerSize',6,'MarkerFaceColor','[0.7350 0.0780 0.0840]','LineWidth',1.5);
legend('$D_{KL}(\widehat{\pi}_{k}\|\pi)$', 'upper bound', 'Location', 'northeast','interpreter','latex','fontsize',20);
set(gca, 'FontSize', 12);
xlabel('Iteration','fontsize',16);
ylabel('KL  divergence','fontsize',16);
grid on;
grid minor;


%%--------------------------------------------------------------
% Plot 6 random 2D marginal contour comparisons in a 2x3 layout.
lambda = Lam1(k);
Vk = V1(:,1:k);
Bk = B1(1:k+1,1:k);
[m, C] = exact_post(lambda, N, A, M, b);
[mhat, Chat] = approx_post(lambda, N, A, M, b, Bk, Vk);
plot_gaussian_contour(m, C, mhat, Chat, 2026);