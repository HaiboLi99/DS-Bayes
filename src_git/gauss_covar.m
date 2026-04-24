function y = gauss_covar(v, s1, s2, l)
% Function handle of Gaussian prior covariance matrix,
% compute y = Sigma*v, where Sigma is the discretized covariance matrix
% 
% Inputs:
%   v: vector discretized from a 1D uniform grid 
%   s_1, s_2: interval [s1, s2]
%   l: correlation length of the Gaussian prior
%
% Outputs: 
%   y: y = Sigma*v
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
%

n = size(v,1);
h = (s2-s1)/n;
p = -s1 + (0.5:n-0.5)*h;
col1 = exp(-(p - p(1)).^2 / (2*l^2))';
col1(1) = col1(1) + 1e-10;

c_full = [col1; 0; col1(end:-1:2)]; 
v_full = [v; zeros(n, 1)];
res = ifft(fft(c_full) .* fft(v_full));
y = res(1:n);

end
