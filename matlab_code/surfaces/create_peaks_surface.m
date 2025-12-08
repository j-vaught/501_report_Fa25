function [X, Y, Z, dx, dy] = create_peaks_surface(Nx, Ny, scale)
% CREATE_PEAKS_SURFACE Generate MATLAB peaks function surface
    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, scale = 0.15; end
    
    x = linspace(-3, 3, Nx);
    y = linspace(-3, 3, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    % MATLAB peaks function
    Z = scale * (3*(1-X).^2.*exp(-X.^2 - (Y+1).^2) ...
        - 10*(X/5 - X.^3 - Y.^5).*exp(-X.^2 - Y.^2) ...
        - 1/3*exp(-(X+1).^2 - Y.^2));
end
