function y = iso_embed(z, A, N)
% Computes the isometric embedding N*A'*z for a vector z.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
%

if isa(A, 'function_handle')
    az = A(z, 'transp');
else
    az = A' * z;
end

if isa(N, 'function_handle')
    % gz = N(az, 'notransp');
    y = N(az);
else
    y = N * az;
end


end
