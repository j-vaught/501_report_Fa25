function [X, Y, Z, dx, dy] = create_gaussian_surface(Nx, Ny, amplitude, sigma)
% CREATE_GAUSSIAN_SURFACE Generate a Gaussian bump surface
%   [X, Y, Z, dx, dy] = create_gaussian_surface(Nx, Ny, amplitude, sigma)
%
%   Returns:
%       X, Y - 2D coordinate meshgrids
%       Z    - Height field
%       dx, dy - Grid spacing

    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, amplitude = 1.0; end
    if nargin < 4, sigma = 0.5; end
    
    x = linspace(-2, 2, Nx);
    y = linspace(-2, 2, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    Z = amplitude * exp(-(X.^2 + Y.^2) / (2*sigma^2));
end
