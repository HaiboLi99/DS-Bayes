function GtV = apply_Gt_cols(G, V)
% Compute G' * V columnwise, output is n-by-k
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

if ~isa(G, 'function_handle')
    [m, n] = size(G);
else
    [m, n] = sizemm(G);
end

if size(G,2) ~= n || size(V,1) ~= m
    error('Dimensions not consistent')
end

k = size(V,2);
GtV = zeros(n, k);

if ~isa(G, 'function_handle')
    GtV = G' * V;
    return;
end

for j = 1:k
    GtV(:,j) = G(V(:,j), 'transp');
end

end