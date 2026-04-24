% function dF = forstner_dist(A, B)
% % forstner_dist
% % Compute the Forstner distance between two SPD matrices A and B:
% %   d_F(A,B)^2 = sum_i log^2(sigma_i),
% % where sigma_i are generalized eigenvalues of (A,B)
% % 
% % Haibo Li, School of Mathematics and Statistics, HUST
% % 10, April, 2026.
% %

% A = 0.5 * (A + A');
% B = 0.5 * (B + B');

% sigma = eig(A, B);
% sigma = real(sigma);
% sigma = sigma(sigma > 0);

% dF = sqrt(sum(log(sigma).^2));

% end


%%--------------------------------------------
function dF = forstner_dist(A, B)
% forstner_dist
% Compute the Forstner distance between two SPD matrices A and B:
%   d_F(A,B)^2 = sum_i log^2(sigma_i),
% where sigma_i are generalized eigenvalues of (A,B)
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

A = 0.5 * (A + A');
B = 0.5 * (B + B');

% sigma = eig(A, B);
% sigma = real(sigma);
% sigma = sigma(sigma > 0);

% dF = sqrt(sum(log(sigma).^2));


% Cholesky
L = chol(A, 'lower');

% form A^{-1/2} B A^{-1/2}
C = L \ B / L';

C = 0.5*(C + C');  % ensure symmetry

% eigenvalues
sigma = eig(C);

% remove tiny/negative due to roundoff
sigma = max(sigma, 1e-16);

% compute distance
dF = sqrt(sum(log(sigma).^2));

end