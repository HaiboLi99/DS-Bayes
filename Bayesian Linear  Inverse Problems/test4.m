% Compare the running time between the Q-GKB with direct LIS as k increase, 
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
N1 = materialize_sigma(N, NN^2);


%%-------------------------------------------------
test_ks = [50, 100, 150, 200, 250];
num_pts = length(test_ks);
time1   = zeros(num_pts ,1);
time2   = zeros(num_pts ,1);
er1 = zeros(num_pts ,1);
er2 = zeros(num_pts ,1);
mstd1 = zeros(num_pts ,1);
mstd2 = zeros(num_pts ,1);

for j = 1: num_pts
    kj = test_ks(j);
    t1 = tic;
    [X1, V1, B1, Lam1, ~] = QGKB_HB(A, b, M, N, kj);
    lambda = Lam1(kj);
    Vk = V1(:,1:kj);
    Bk = B1(1:kj+1,1:kj);
    [~, diagChat] = approx_post_mean_var(lambda, A, M, b, Bk, Vk, ProbInfo);
    time1(j) = toc(t1);
    er1(j)   = norm(X1(:,kj)-x_true) / xn;
    mstd1(j) = sqrt(mean(diagChat));

    t2 = tic;
    [x_lis, postVar_lis, mstd_lis] = LIS_est(A, b, M, N1, lambda, kj);
    time2(j) = toc(t2);
    er2(j)   = norm(x_lis-x_true) / xn;
    mstd2(j) = mstd_lis;
end 


%----------- display table ----------------------------------------
fprintf('QGKB vs Direct-LIS========================\n');
fprintf('QGKB: \n'); 
T1 = table(test_ks(:), time1(:), er1(:), mstd1(:), ...
    'VariableNames', {'k', 'QGKB_time', 'QGKB_error', 'QGKB_mstd'});
disp(T1)
fprintf('LIS: \n'); 
T2 = table(test_ks(:), time2(:), er2(:), mstd2(:), ...
    'VariableNames', {'k', 'LIS_time', 'LIS_error', 'LIS_mstd'});
disp(T2)

fprintf('QGKB vs Direct-LIS:=========================\n');
vals = [time1.'; er1.'; mstd1.'; time2.'; er2.'; mstd2.'];
row_names = {'QGKB_time','QGKB_error','QGKB_mstd', ...
             'LIS_time','LIS_error','LIS_mstd'};
col_names = matlab.lang.makeValidName("k = " + string(test_ks));
T = array2table(vals, 'RowNames', row_names, 'VariableNames', col_names);
disp(T)