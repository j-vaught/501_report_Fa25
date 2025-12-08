function Z = solve_poisson_fd(f, dx, dy)
% SOLVE_POISSON_FD Finite Difference solver with Dirichlet BC (direct)
%   Z = solve_poisson_fd(f, dx, dy)
%
%   Same as CG but uses direct sparse solve (backslash)
%   Equivalent to Python's FD-Dirichlet solver

    [Ny, Nx] = size(f);
    N = Nx * Ny;
    
    cx = 1 / dx^2;
    cy = 1 / dy^2;
    cc = -2 * (cx + cy);
    
    % Build Laplacian using spdiags
    d0 = cc * ones(N, 1);
    d1 = cx * ones(N, 1);
    dm1 = cx * ones(N, 1);
    dNx = cy * ones(N, 1);
    dmNx = cy * ones(N, 1);
    
    % Remove cross-row connections
    for i = 1:Ny
        d1((i-1)*Nx + Nx) = 0;
        dm1((i-1)*Nx + 1) = 0;
    end
    
    A = spdiags([dmNx, dm1, d0, d1, dNx], [-Nx, -1, 0, 1, Nx], N, N);
    
    % Apply Dirichlet BC
    b = f(:);
    for i = 1:Ny
        for j = 1:Nx
            k = (i-1)*Nx + j;
            if i == 1 || i == Ny || j == 1 || j == Nx
                A(k, :) = 0;
                A(k, k) = 1;
                b(k) = 0;
            end
        end
    end
    
    % Direct solve
    z_vec = A \ b;
    
    Z = reshape(z_vec, [Ny, Nx]);
    Z = Z - mean(Z(:));
end
