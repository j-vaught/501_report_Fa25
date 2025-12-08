function N = height_to_normals(Z, dx, dy)
% HEIGHT_TO_NORMALS Compute surface normals from height field
%   N = height_to_normals(Z, dx, dy)
%
%   Inputs:
%       Z  - height field (Ny x Nx)
%       dx, dy - grid spacing
%
%   Output:
%       N - normal map (Ny x Nx x 3)

    % Compute gradients
    [Zy, Zx] = gradient(Z, dx, dy);
    
    % Normal is (-dz/dx, -dz/dy, 1) normalized
    Nx = -Zx;
    Ny = -Zy;
    Nz = ones(size(Z));
    
    % Normalize
    norm_val = sqrt(Nx.^2 + Ny.^2 + Nz.^2);
    
    N = zeros([size(Z), 3]);
    N(:,:,1) = Nx ./ norm_val;
    N(:,:,2) = Ny ./ norm_val;
    N(:,:,3) = Nz ./ norm_val;
end
