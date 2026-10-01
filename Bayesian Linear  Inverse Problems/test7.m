% Compare the Q-GKB with direct LIS for the 2D deblurring problem,
% where we use 2D separable Matern kernel.
% Several different random seeds are used.
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 28, Sept, 2026.
% 

clear, clc;
close all;
directory = pwd;
path(directory, path)
addpath(genpath('..'))
% rng(2026);  

opts = struct();
opts.N = 256;
opts.M = 128;
opts.nu = 3;
opts.rho = 0.1;
opts.sigma = 1;
opts.t_blur = 0.01;      
opts.noise_std = 0;    
opts.truncate_tol = 1e-8;   

nel = 1e-2; 
kk = 250;

num_t = 5;    % number of tests

er1 = zeros(num_t,1);  % error of QGKB mean
er2 = zeros(num_t,1);  % error of LIS mean
er3 = zeros(num_t,1);  % error of exact mean

mstd1 = zeros(num_t,1);  % QGKB mstd
mstd2 = zeros(num_t,1);  % LIS mstd
mstd3 = zeros(num_t,1);  % exact mstd

for i = 1:num_t
    opts.seed = 2026-i;
    [A, b_true, x_true, ProbInfo] = blurgauss_rect(opts);
    xn = norm(x_true);
    [e, M] = genNoise(b_true, nel, 'white'); 
    b = b_true + e;
    NN = ProbInfo.xSize(1);
    xn = norm(x_true);
    N = @(v) matern2d_sep_covar(v, 0, 1, NN, opts.nu, opts.rho, 1);

    [X1, V1, B1, Lam1, L_vals] = QGKB_HB(A, b, M, N, kk);
    Vk = V1(:,1:kk);
    Bk = B1(1:kk+1,1:kk);
    [~, diagChat] = approx_post_mean_var(Lam1(kk), A, M, b, Bk, Vk, ProbInfo);
    er1(i) = norm(x_true-X1(:,end)) / xn;
    mstd1(i) = sqrt(mean(diagChat));

    [X2, postVar2, mstd22] = LIS_est_separ1(A, b, M, Lam1, kk, ProbInfo);
    er2(i) = norm(x_true-X2(:,end)) / xn;
    mstd2(i) = mstd22(end);

    sigma_eps = sqrt(M(1,1));
    [m_post, diagC_post, ~, ~] = exact_post_mean_var(b, Lam1(kk), sigma_eps, ProbInfo);
    er3(i) = norm(x_true-m_post) / xn;
    mstd3(i) = sqrt(mean(diagC_post));
end


%----------- display table ------------------------------------
fprintf('QGKB vs LIS vs Exact========================\n');

fprintf('QGKB: \n'); 
T1 = table(er1(:), mstd1(:), ...
    'VariableNames', {'QGKB_error', 'QGKB_mstd'});
disp(T1)

fprintf('LIS: \n'); 
T2 = table(er2(:), mstd2(:), ...
    'VariableNames', {'LIS_error', 'LIS_mstd'});
disp(T2)

fprintf('Exact: \n'); 
T3 = table(er3(:), mstd3(:), ...
    'VariableNames', {'Exact_error', 'Exact_mstd'});
disp(T3)