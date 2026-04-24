function [A, b, x] = Fred_gauss(m, n, type)
% Test problem: one-dimensional Fredholm integral 
% equation of the first kind, where the true x is sampled
% from a Gaussian distribution with Gaussian prior.
% the discretized linear system is Ax=b, where a uniform
% method is used.
%
% Inputs:
%   m, n: size of A
%   type: 
%       'exp': integral kernel with exponetial decaying
%              singular values of the operator
%       'poly': integral kernel with polynomial decaying
%              singular values of the operator
%
% Outputs:
%   A, x, b: linear system Ax=b
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
%

h_s = pi / m; %
h_t = pi / n; % 
s = -pi/2 + (0.5:m-0.5)*h_s; 
t = -pi/2 + (0.5:n-0.5)*h_t;

A = zeros(m, n);

if strcmp(type, 'exp')
    % K(s,t) = (cos(s) + cos(t))^2 * (sin(u)/u)^2, 其中 u = pi*(sin(s) + sin(t))
    for i = 1:m
        for j = 1:n
            u_val = pi * (sin(s(i)) + sin(t(j)));
        
            if abs(u_val) < 1e-12
                sterm = 1;
            else
                sterm = (sin(u_val)/u_val)^2;
            end
        
            A(i,j) = (cos(s(i)) + cos(t(j)))^2 * sterm;
        end
    end
elseif strcmp(type, 'poly')
    for i = 1:m
        for j = 1:n
            % sterm  = abs(sin(s(i)*t(j)+1));
            % A(i,j) = sterm / t(j);
            A(i,j) = exp(-abs(s(i)-t(j))/10);
        end
    end
else
    error('No such type of kernel')
end

A = A * h_t;

% generate a true x
l = 0.4;
sigma = 0.2;

s_start = -pi/2;
s_end = pi/2;
s = linspace(s_start, s_end, n)'; 

% Using the squared distance matrix: (s_i - s_j)^2
[S1, S2] = meshgrid(s, s);
dist_sq = (S1 - S2).^2;
K = sigma^2 * exp(-dist_sq / (2 * l^2)) + 1e-10 * eye(n);
L = chol(K, 'lower');

% Sample from the GP（x = mean + L * white_noise）
rng(2026); 
u = randn(n, 1);
x = L * u;

b = A * x;

end