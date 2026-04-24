function [A, b, x, ProbInfo] = CT_paral(varargin)
% Generate a 2D parallel-beam X-ray CT test problem.
%
% Inputs:
%   N    : image size, so the image is N-by-N. Default: 256.
%   opts : optional struct with fields
%          .phantomImage : numeric N-by-N image, or string:
%                          'shepplogan' (default);
%                           'smooth':  Gaussian mixture;
%                           'matern': 1d Matern sample.
%          .sm           : true  -> return sparse matrix A (default)
%                          false -> return function handle A
%          .angles       : projection angles in degrees.
%                          Default: 0:1:179
%          .p            : number of rays per angle.
%                          Default: round(sqrt(2)*N)
%          .d            : distance from first ray to last.
%                          Default: p - 1
%          .isDisp       : display rays during construction.
%                          Default: 0
%
% Outputs:
%   A        : sparse matrix or function handle for forward/adjoint operator
%   b        : noiseless projection data
%   x        : vectorized exact image
%   ProbInfo : struct with problem information
%
% Haibo Li, School of Mathematics and Statistics, HUST
% 14, April, 2026.
% 

% Parse inputs
default_N = 256;
default_opts = struct( ...
    'phantomImage', 'shepplogan', ...
    'sm', true, ...
    'angles', 0:1:179, ...
    'p', [], ...
    'd', [], ...
    'isDisp', 0);

switch nargin
    case 0
        N = default_N;
        opts = default_opts;
    case 1
        if isnumeric(varargin{1}) && isscalar(varargin{1})
            N = varargin{1};
            opts = default_opts;
        elseif isstruct(varargin{1})
            N = default_N;
            opts = set_defaults(varargin{1}, default_opts);
        else
            error('Single input must be either a scalar N or an options struct.');
        end
    case 2
        if ~(isnumeric(varargin{1}) && isscalar(varargin{1}))
            error('When two inputs are given, the first must be scalar N.');
        end
        if ~isstruct(varargin{2})
            error('When two inputs are given, the second must be an options struct.');
        end
        N = varargin{1};
        opts = set_defaults(varargin{2}, default_opts);
    otherwise
        error('Too many input arguments.');
end

% Set defaults depending on N
if isempty(opts.p)
    opts.p = round(sqrt(2) * N);
end
if isempty(opts.d)
    opts.d = opts.p - 1;
end

angles = double(opts.angles(:).');  % force row vector, double precision
p = opts.p;
d = opts.d;
isDisp = opts.isDisp;
sm = opts.sm;

% Generate phantom
if isnumeric(opts.phantomImage)
    img = double(opts.phantomImage);
    if ~ismatrix(img)
        error('User supplied phantomImage must be a 2D array.');
    end
    [n1, n2] = size(img);
    if n1 ~= n2
        error('User supplied phantomImage must be square.');
    end
    N = n1;
else
    img = generate_phantom(opts.phantomImage, N);
end

x = img(:);

% Construct forward operator
if sm
    A = paralleltomo_simple(N, angles, p, d, isDisp, true);
    b = A * x;
else
    A = @(u, transp_flag) paralleltomo_simple(N, angles, p, d, isDisp, false, u, transp_flag);
    b = A(x, 'notransp');
end

% Pack problem information
ProbInfo = struct();
ProbInfo.problemType = 'tomography';
ProbInfo.xType = 'image2D';
ProbInfo.bType = 'image2D';
ProbInfo.xSize = [N, N];
ProbInfo.bSize = [p, length(angles)];
ProbInfo.N = N;
ProbInfo.angles = angles;
ProbInfo.p = p;
ProbInfo.d = d;
ProbInfo.sm = sm;
ProbInfo.geometry = 'parallel';
ProbInfo.phantomImage = opts.phantomImage;

end



%%---------------------------------------------------------------
function opts = set_defaults(opts, default_opts)
% Fill missing fields in opts using default_opts

fn = fieldnames(default_opts);
for i = 1:length(fn)
    if ~isfield(opts, fn{i})
        opts.(fn{i}) = default_opts.(fn{i});
    end
end
end


%%---------------------------------------------------------
function img = generate_phantom(name, N)
% Generate a simple test image
switch lower(name)
    case 'shepplogan'
        img = shepplogan_phantom(N);

    case 'smooth'
        % [X, Y] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));
        % img = exp(-4*(X.^2 + Y.^2)) + 0.8*exp(-10*((X-0.35).^2 + (Y+0.2).^2));
        % img = img / max(img(:));
        img = smooth(N,4);
    case 'matern'
        img = matern_grf_2d(N, 2, 0.2, 1);
    otherwise
        error('Unknown phantomImage option.');
end
end


%%----------------------------------------------------------------
function P = shepplogan_phantom(N)
% Simple modified Shepp-Logan phantom without toolbox dependency

% Parameters: [A, a, b, x0, y0, phi(deg)]
E = [ ...
     1.0   .6900   .920   0       0        0
    -0.8   .6624   .8740  0      -0.0184   0
    -0.2   .1100   .3100  0.22    0      -18
    -0.2   .1600   .4100 -0.22    0       18
     0.1   .2100   .2500  0       0.35     0
     0.1   .0460   .0460  0       0.1      0
     0.1   .0460   .0460  0      -0.1      0
     0.1   .0460   .0230 -0.08   -0.605    0
     0.1   .0230   .0230  0      -0.606    0
     0.1   .0230   .0460  0.06   -0.605    0];

[x, y] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));
P = zeros(N,N);

for k = 1:size(E,1)
    A = E(k,1);
    a = E(k,2);
    b = E(k,3);
    x0 = E(k,4);
    y0 = E(k,5);
    phi = E(k,6) * pi / 180;

    xrot =  (x - x0)*cos(phi) + (y - y0)*sin(phi);
    yrot = -(x - x0)*sin(phi) + (y - y0)*cos(phi);

    idx = (xrot.^2 / a^2 + yrot.^2 / b^2) <= 1;
    P(idx) = P(idx) + A;
end

P = max(P, 0);
if max(P(:)) > 0
    P = P / max(P(:));
end
end


%%----------------------------------------
function img = smooth(N,p)
%SMOOTH Creates a 2D test image of a smooth function

if nargin==1, p = 4; end

% Generate the image
[I,J] = meshgrid(1:N);
sigma = 0.25*N;
c = [0.6*N 0.6*N; 0.5*N 0.3*N; 0.2*N 0.7*N; 0.8*N 0.2*N];
a = [1 0.5 0.7 0.9];
im = zeros(N,N);

for i=1:p
    im = im + a(i)*exp( - (I-c(i,1)).^2/(1.2*sigma)^2 - (J-c(i,2)).^2/sigma^2);
end

img = im/max(im(:));

end


%%---------------------------------------------------------------
function img = matern_grf_2d(N, nu, rho, sigma)
% Generate a 2D Gaussian random field with Matérn covariance
%
% Inputs:
%   N     : grid size (NxN)
%   nu    : smoothness parameter
%   rho   : correlation length
%   sigma : variance scale
%
% Output:
%   img   : N-by-N random field

% frequency grid
[k1, k2] = meshgrid( ...
    [0:(N/2-1), -N/2:-1], ...
    [0:(N/2-1), -N/2:-1]);

k_sq = k1.^2 + k2.^2;

% Matérn spectral density
kappa = sqrt(2*nu) / rho;
S = (kappa^2 + (2*pi)^2 * k_sq).^(-(nu + 1));

% normalize variance
S = sigma^2 * S / max(S(:));

% sample in Fourier domain
Z = randn(N,N) + 1i*randn(N,N);
F = sqrt(S) .* Z;

% inverse FFT
img = real(ifft2(F));

% normalize
img = img - mean(img(:));
img = img / std(img(:));

end


%%---------------------------------------------------------------------
function A = paralleltomo_simple(N, theta, p, d, isDisp, isMatrix, u, transp_flag)
% Construct or apply the 2D parallel-beam tomography operator

nA = length(theta);

% Starting points of rays before rotation
x0 = linspace(-d/2, d/2, p)';
y0 = zeros(p,1);

% Grid lines
xgrid = (-N/2:N/2)';
ygrid = xgrid;

if isDisp
    AA = generate_phantom('smooth', N);
    figure;
end

if isMatrix
    rows = zeros(2 * N * nA * p, 1);
    cols = rows;
    vals = rows;
    idxend = 0;

    II = 1:nA;
    JJ = 1:p;
else
    switch lower(transp_flag)
        case 'size'
            A = [p*nA, N^2];
            return
        case 'notransp'
            if length(u) ~= N^2
                error('Incorrect length of input vector for forward operation.');
            end
            A = zeros(p*nA, 1);
        case 'transp'
            if length(u) ~= p*nA
                error('Incorrect length of input vector for adjoint operation.');
            end
            A = zeros(N^2, 1);
        otherwise
            error('Unknown transp_flag.');
    end

    II = 1:nA;
    JJ = 1:p;
end

for i = II
    if isDisp
        clf
        pause(isDisp)
        imagesc((-N/2+0.5):(N/2-0.5), (-N/2+0.5):(N/2-0.5), flipud(AA))
        colormap gray
        axis xy equal
        hold on
        axis(0.7 * [-N N -N N])
    end

    % Rotate ray start points
    x0theta = cosd(theta(i)) * x0 - sind(theta(i)) * y0;
    y0theta = sind(theta(i)) * x0 + cosd(theta(i)) * y0;

    % Direction vector of rays
    a = -sind(theta(i));
    b =  cosd(theta(i));

    for j = JJ

        % Intersections with vertical grid lines
        tx = (xgrid - x0theta(j)) / a;
        yx = b * tx + y0theta(j);

        % Intersections with horizontal grid lines
        ty = (ygrid - y0theta(j)) / b;
        xy = a * ty + x0theta(j);

        if isDisp
            plot(xgrid, yx, '-', 'color', [220 0 0]/255, 'LineWidth', 1.5)
            plot(xy, ygrid, '-', 'color', [220 0 0]/255, 'LineWidth', 1.5)
            set(gca, 'XTick', [], 'YTick', [])
            pause(isDisp)
        end

        % Collect all intersections
        t = [tx; ty];
        xxy = [xgrid; xy];
        yxy = [yx; ygrid];

        % Sort by parameter t
        [~, I] = sort(t);
        xxy = xxy(I);
        yxy = yxy(I);

        % Keep only points inside box
        I = (xxy >= -N/2 & xxy <= N/2 & yxy >= -N/2 & yxy <= N/2);
        xxy = xxy(I);
        yxy = yxy(I);

        % Remove duplicate points
        Idup = (abs(diff(xxy)) <= 1e-10 & abs(diff(yxy)) <= 1e-10);
        xxy(Idup) = [];
        yxy(Idup) = [];

        % Segment lengths inside cells
        aval = sqrt(diff(xxy).^2 + diff(yxy).^2);
        col = [];

        if ~isempty(aval)
            if ~((b == 0 && abs(y0theta(j) - N/2) < 1e-15) || ...
                 (a == 0 && abs(x0theta(j) - N/2) < 1e-15))

                xm = 0.5 * (xxy(1:end-1) + xxy(2:end)) + N/2;
                ym = 0.5 * (yxy(1:end-1) + yxy(2:end)) + N/2;

                col = floor(xm) * N + (N - floor(ym));
            end
        end

        if ~isempty(col)
            if isMatrix
                idxstart = idxend + 1;
                idxend = idxstart + numel(col) - 1;
                idx = idxstart:idxend;

                rows(idx) = (i-1)*p + j;
                cols(idx) = col;
                vals(idx) = aval;
            else
                switch lower(transp_flag)
                    case 'notransp'
                        A((i-1)*p + j) = aval' * u(col);
                    case 'transp'
                        A(col) = A(col) + u((i-1)*p + j) * aval;
                end
            end
        end
    end
end

if isMatrix
    rows = rows(1:idxend);
    cols = cols(1:idxend);
    vals = vals(1:idxend);

    A = sparse(rows, cols, vals, p*nA, N^2);
end
end