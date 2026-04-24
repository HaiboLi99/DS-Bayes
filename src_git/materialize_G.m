function Gmat = materialize_G(G)
% Convert G into an explicit m-by-n matrix
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

if ~isa(G, 'function_handle')
    [m, n] = size(G);
else
    [m, n] = sizemm(G);
end

if ~isa(G, 'function_handle')
    Gmat = G;
    return;
end

Gmat = zeros(m,n);

for j = 1:n
    ej = zeros(n,1);
    ej(j) = 1;
    Gmat(:,j) = G(ej, 'notransp');
end

end