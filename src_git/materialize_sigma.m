function Sig = materialize_sigma(Sigma, n)
% Convert Sigma into an explicit n-by-n matrix
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

if ~isa(Sigma, 'function_handle')
    Sig = Sigma;
    return;
end

Sig = zeros(n,n);

for j = 1:n
    ej = zeros(n,1);
    ej(j) = 1;
    Sig(:,j) = Sigma(ej);
end
Sig = 0.5 * (Sig + Sig');

end