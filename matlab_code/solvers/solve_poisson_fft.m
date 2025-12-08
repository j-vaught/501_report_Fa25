function Z = solve_poisson_fft(f, dx, dy)
% SOLVE_POISSON_FFT Solve Poisson equation using FFT (periodic BC)
%   Z = solve_poisson_fft(f, dx, dy)
%
%   Solves: nabla^2 Z = f with periodic boundary conditions
%
%   Inputs:
%       f  - divergence field (Ny x Nx)
%       dx, dy - grid spacing
%
%   Output:
%       Z - height field solution

    [Ny, Nx] = size(f);
    
    % Build frequency grid
    kx = (2*pi / (Nx*dx)) * [0:Nx/2-1, -Nx/2:-1];
    ky = (2*pi / (Ny*dy)) * [0:Ny/2-1, -Ny/2:-1];
    [KX, KY] = meshgrid(kx, ky);
    
    % Laplacian in Fourier space
    denom = -(KX.^2 + KY.^2);
    denom(1, 1) = 1;  % Avoid division by zero
    
    % Transform, solve, inverse transform
    F_hat = fft2(f);
    Z_hat = F_hat ./ denom;
    Z_hat(1, 1) = 0;  % Set DC component to zero
    
    Z = real(ifft2(Z_hat));
    
    % Mean-center the result
    Z = Z - mean(Z(:));
end
