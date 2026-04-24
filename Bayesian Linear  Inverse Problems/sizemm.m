function [m ,n] = sizemm(A)
% Get the size for a matrix A, where A can be a 
%   (a). full or sparse mxn matrix;
%   (b). a matrix object that performs the matrix*vector operation
%   (c). a functional handle
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 07, April, 2026.
%

if isa(A, 'function_handle')
    dim = A([], 'size');
    m = dim(1);
    n = dim(2);
else
    [m, n] = size(A);
end

