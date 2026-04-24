function y = Gfun(z, A, N)
% Let G = A*N*A', computes G*z for a vector z.
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
%

if isa(A, 'function_handle')
    % az = A(z, 'notransp');
    az = A(z, 'transp');
else
    az = A' * z;
end

if isa(N, 'function_handle')
    % gz = N(az, 'notransp');
    gz = N(az);
else
    gz = N * az;
end

if isa(A, 'function_handle')
    y = A(gz, 'notransp');
else
    y = A * gz;
end

end

