function Z = solve_poisson_dct(f, dx, dy)
% SOLVE_POISSON_DCT Solve Poisson equation using DCT (Neumann BC)
%   Z = solve_poisson_dct(f, dx, dy)
%
%   Solves: nabla^2 Z = f with Neumann (zero-flux) boundary conditions
%
%   Inputs:
%       f  - divergence field (Ny x Nx)
%       dx, dy - grid spacing
%
%   Output:
%       Z - height field solution

    [Ny, Nx] = size(f);
    
    % Enforce compatibility: mean(f) must be zero for Neumann
    f = f - mean(f(:));
    
    % Build eigenvalue array for DCT
    ii = 0:Nx-1;
    jj = 0:Ny-1;
    
    lambda_x = -2/dx^2 * (1 - cos(pi * ii / Nx));
    lambda_y = -2/dy^2 * (1 - cos(pi * jj / Ny));
    
    [LX, LY] = meshgrid(lambda_x, lambda_y);
    lambda = LX + LY;
    lambda(1, 1) = 1;  % Avoid division by zero
    
    % Forward DCT-II
    F_hat = dct2(f);
    
    % Spectral solve
    Z_hat = F_hat ./ lambda;
    Z_hat(1, 1) = 0;  % DC component
    
    % Inverse DCT-II
    Z = idct2(Z_hat);
    
    % Mean-center
    Z = Z - mean(Z(:));
end
