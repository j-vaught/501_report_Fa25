function [X, Y, Z, dx, dy] = create_sinusoid_surface(Nx, Ny, amplitude, freq)
% CREATE_SINUSOID_SURFACE Generate a 2D sinusoidal surface
    if nargin < 1, Nx = 128; end
    if nargin < 2, Ny = 128; end
    if nargin < 3, amplitude = 0.5; end
    if nargin < 4, freq = 2; end
    
    x = linspace(-2, 2, Nx);
    y = linspace(-2, 2, Ny);
    [X, Y] = meshgrid(x, y);
    
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    
    Z = amplitude * sin(freq * pi * X) .* sin(freq * pi * Y);
end
