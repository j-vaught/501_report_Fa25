function [X, Y, Z, dx, dy] = create_sphere_surface(Nx, Ny, radius)
% CREATE_SPHERE_SURFACE Generate a hemispherical surface
    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, radius = 1.5; end
    
    x = linspace(-2, 2, Nx);
    y = linspace(-2, 2, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    r2 = X.^2 + Y.^2;
    Z = zeros(size(X));
    mask = r2 < radius^2;
    Z(mask) = sqrt(radius^2 - r2(mask));
end
