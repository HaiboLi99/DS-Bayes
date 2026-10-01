% Counting the computational time for Q-GKB and LIS for 2D deblurring problem

clear, clc;
directory = pwd;


% time for QGKB
t1 = [138.58, 282.82, 420.96, 562.26, 692.95];   

% time for LIS
t2 = [660.75 , 778.73, 898.38, 1112.9, 1319.1];   

% time for full-space Bayes
t0 = 5980.5812;

% rank scale
k = [50, 100, 150, 200, 250];


%------ plot ------------------------------------------------------
fig = figure('Units','pixels', 'Position',[100, 80, 800, 600]);
t = tiledlayout(1, 1, 'TileSpacing','compact', 'Padding','compact');
semilogy(k, t1, '-d','Color',[0.6350 0.0780 0.1840],'MarkerIndices',1:1:5,...
    'MarkerSize',8,'MarkerFaceColor',[0.6350 0.0780 0.1840],'LineWidth',1.5);
hold on;
semilogy(k, t2, '-s','Color',[0 0.4470 0.7410],'MarkerIndices',1:1:5,...
    'MarkerSize',8,'MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',1.5);
hold on;
semilogy(k, ones(length(k),1)*t0, '-','Color','k','LineWidth',2);
set(gca, 'FontSize', 14);
xlabel('$k$','interpreter','latex','fontsize',22);
legend('QGKB','LIS','full-space Bayes','Fontsize',16, 'Location', 'northwest');
ylabel('Time (seconds)','Fontsize',18);
grid on;
grid minor;
title('Running time for different $k$','interpreter','latex','fontsize',22)
