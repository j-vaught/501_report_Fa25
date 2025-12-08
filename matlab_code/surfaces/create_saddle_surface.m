function [X, Y, Z, dx, dy] = create_saddle_surface(Nx, Ny, scale)
% CREATE_SADDLE_SURFACE Generate a hyperbolic paraboloid (saddle)
    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, scale = 0.3; end
    
    x = linspace(-2, 2, Nx);
    y = linspace(-2, 2, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    Z = scale * (X.^2 - Y.^2);
end
