function plot_gaussian_contour(m_exact, C_exact, m_approx, C_approx, seed)
% Plot 6 random 2D marginal contour comparisons in a 2x3 layout, to 
% compare the exact and approximate posterior.
%
% Inputs:
%   m_exact  : exact posterior mean 
%   C_exact  : exact posterior covariance
%   m_approx : approximate posterior mean 
%   C_approx : approximate posterior covariance
%   seed     : random seed for choosing (i,j) marginals
%
% Outputs:
%   2x3 subplots of 2D marginal contours.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 13, April, 2026.
%

if nargin < 5
    seed = 2026;
end

n = length(m_exact);

% Symmetrize
C_exact  = 0.5 * (C_exact  + C_exact');
C_approx = 0.5 * (C_approx + C_approx');

rng(seed);

% ---- choose 6 piars randomly ----
all_pairs = nchoosek(1:n, 2);
idx = randperm(size(all_pairs,1), 6);
pairs = all_pairs(idx,:);

fig = figure('Units','pixels', 'Position',[100, 80, 1200, 760]);
t = tiledlayout(2,3, 'TileSpacing','compact', 'Padding','compact');
for p = 1:6
    i = pairs(p,1);
    j = pairs(p,2);

    % ---- exactract 2D marginal ----
    m1 = [m_exact(i); m_exact(j)];
    S1 = C_exact([i,j],[i,j]);

    m2 = [m_approx(i); m_approx(j)];
    S2 = C_approx([i,j],[i,j]);

    S1 = 0.5*(S1+S1');
    S2 = 0.5*(S2+S2');

    [xg, yg] = make_grid(m1,S1,m2,S2);

    z1 = gaussian_pdf(xg, yg, m1, S1);
    z2 = gaussian_pdf(xg, yg, m2, S2);

    % subplot(2,3,p);
    % hold on;
    nexttile;
    hold on;

    contour(xg, yg, z1, 6, '-', 'LineWidth', 1.2);
    contour(xg, yg, z2, 6, '--', 'LineWidth', 1.8);   
    plot(m1(1), m1(2), 'o', ...
        'Color', [0.0000, 0.4470, 0.7410], ...
        'MarkerSize', 8, 'LineWidth', 1.8);
    plot(m2(1), m2(2), 'x', ...
        'Color', [0.8500, 0.3250, 0.0980], ...
        'MarkerSize', 8, 'LineWidth', 1.8);
    xlabel(sprintf('$x_{%d}$',i), 'interpreter','latex','fontsize',20);
    ylabel(sprintf('$x_{%d}$',j), 'interpreter','latex','fontsize',20);
    title(sprintf('(%d,%d)',i,j), 'fontsize',16, 'FontWeight','normal');
    % axis equal;
    % axis tight;
    grid on;
    grid minor;
    box on;

    pbaspect([1 1 1]);
    hold off;
end

lgd = legend({'Exact posterior','Approx posterior','Exact mean','Approx mean'},'Orientation','horizontal');
lgd.Layout.Tile = 'south';
lgd.FontSize = 18;
lgd.FontWeight = 'bold';
end


%-------------------------------------------------
function [xg, yg] = make_grid(m1,S1,m2,S2)
    s1 = sqrt(max(diag(S1),1e-14));
    s2 = sqrt(max(diag(S2),1e-14));

    xmin = min(m1(1)-3*s1(1), m2(1)-3*s2(1));
    xmax = max(m1(1)+3*s1(1), m2(1)+3*s2(1));
    ymin = min(m1(2)-3*s1(2), m2(2)-3*s2(2));
    ymax = max(m1(2)+3*s1(2), m2(2)+3*s2(2));

    xv = linspace(xmin,xmax,120);
    yv = linspace(ymin,ymax,120);

    [xg,yg] = meshgrid(xv,yv);
end


%------------------------------------------------
function z = gaussian_pdf(xg, yg, m, S)
    [R,p] = chol(S);
    if p>0
        S = S + 1e-12*eye(2);
        R = chol(S);
    end

    Sinv = R \ (R' \ eye(2));
    detS = prod(diag(R))^2;

    dx = xg - m(1);
    dy = yg - m(2);

    quad = Sinv(1,1)*dx.^2 + 2*Sinv(1,2)*dx.*dy + Sinv(2,2)*dy.^2;
    z = exp(-0.5*quad) / (2*pi*sqrt(detS));
end