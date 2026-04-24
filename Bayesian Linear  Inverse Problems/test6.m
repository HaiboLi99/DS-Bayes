% Plot evolution of approximate mean/variance for 2D X-ray computed tomography 
% using parallel-beam geometry.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 15, April, 2026.
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

X_deblur = reshape(X1(:,k), NN, NN);
opts1 = struct();
opts1.x_low = -1;
opts1.x_high = 1;
opts1.N = NN;
opts1.nu = nu;
opts1.rho = rho;
opts1.sigma = sigma;


%%---------- plot ------------------------------
test_ks = [40, 120, 200, 280];
num_pts = length(test_ks);

all_var_vals = [];
for j = 1:num_pts
    kj = test_ks(j);
    [~, diagj] = approx_post_mean_var_fft(Lam1(kj), A, M, b, B1(1:kj+1,1:kj), V1(:,1:kj), opts1);
    all_var_vals = [all_var_vals; diagj(:)];
end

var_min = quantile(all_var_vals, 0.02);
var_max = quantile(all_var_vals, 0.98);

fig_final = figure('Units','pixels', 'Position', [50, 50, 1300, 900], 'Color', 'w');
main_ax = axes('Position', [0.1, 0.1, 0.85, 0.85]); 

semilogy(main_ax, 1:k, er1, 'Color', [0, 0.3, 0.1], 'LineWidth', 2);
hold(main_ax, 'on'); grid(main_ax, 'on'); grid(main_ax, 'minor');
set(main_ax, 'FontSize', 14, 'XLim', [0, 310], 'YLim', [min(er1)*0.4, max(er1)*4000]);
xlabel(main_ax, 'Iteration', 'FontSize', 18);
ylabel(main_ax, 'Relative  error', 'FontSize', 18);

inset_w = 0.20;   % height 
inset_h = 0.15;   % weight

y_offsets = [40.0, 700.0, 60.0, 1500.0]; 
x_offsets = [-37, -37, -37, -37]; 

for i = 1:num_pts
    ki = test_ks(i);
    error_i = er1(ki);
    
    [mi, diagi] = approx_post_mean_var_fft(Lam1(ki), A, M, b, B1(1:ki+1,1:ki), V1(:,1:ki), opts1);
    mstd_i = sqrt(mean(diagi));
    
    axPos = main_ax.Position;
    xl = main_ax.XLim; yl = log10(main_ax.YLim);
    
    mark_fig_x = axPos(1) + ((ki - xl(1)) / (xl(2) - xl(1))) * axPos(3);
    mark_fig_y = axPos(2) + ((log10(error_i) - yl(1)) / (yl(2) - yl(1))) * axPos(4);
    
    left = axPos(1) + ((ki + x_offsets(i) - xl(1)) / (xl(2) - xl(1))) * axPos(3);
    bottom_base = axPos(2) + ((log10(error_i * y_offsets(i)) - yl(1)) / (yl(2) - yl(1))) * axPos(4);
    
    ax_v = axes('Position', [left, bottom_base, inset_w, inset_h]);
    imagesc(ax_v, reshape(max(diagi,0), NN, NN)); 
    axis image; axis off; set(ax_v, 'YDir', 'normal');
    colormap(ax_v, parula);
    clim(ax_v, [var_min, var_max]); 
    text(ax_v, 0.5, -0.15, sprintf('mstd = %.2e', mstd_i), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    ax_m = axes('Position', [left, bottom_base + inset_h + 0.002, inset_w, inset_h]);
    imagesc(ax_m, reshape(mi, NN, NN)); 
    axis image; axis off; set(ax_m, 'YDir', 'normal');
    colormap(ax_m, parula);
    title(ax_m, sprintf('$k=%d$', ki), 'Interpreter', 'latex', 'FontSize', 16);
    
    plot(main_ax, ki, error_i, 'rd', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
    line_end_x = left + inset_w/2; 
    text_gap = inset_h * 0.35; 
    line_end_y = bottom_base - text_gap;
    
    annotation('line', [mark_fig_x, line_end_x], [mark_fig_y, line_end_y], ...
               'Color', [0.8, 0.2, 0.2], 'LineStyle', '-.', 'LineWidth', 1.2);
end

title(main_ax, 'Evolution of iterated posterior mean and variance', 'FontSize', 20);