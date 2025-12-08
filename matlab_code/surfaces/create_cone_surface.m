function [X, Y, Z, dx, dy] = create_cone_surface(Nx, Ny, height)
% CREATE_CONE_SURFACE Generate a conical surface
    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, height = 1.5; end
    
    x = linspace(-2, 2, Nx);
    y = linspace(-2, 2, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    r = sqrt(X.^2 + Y.^2);
    Z = max(0, height - r);
end
