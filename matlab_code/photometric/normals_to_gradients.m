function [p, q] = normals_to_gradients(N)
% NORMALS_TO_GRADIENTS Convert normal vectors to surface gradients (p, q)
%   [p, q] = normals_to_gradients(N)
%
%   Input:
%       N - Ny x Nx x 3 normal map
%
%   Outputs:
%       p - dz/dx gradient
%       q - dz/dy gradient

    Nx_comp = N(:,:,1);
    Ny_comp = N(:,:,2);
    Nz_comp = N(:,:,3);
    
    % Avoid division by zero
    Nz_comp(Nz_comp < 1e-10) = 1e-10;
    
    p = -Nx_comp ./ Nz_comp;
    q = -Ny_comp ./ Nz_comp;
end
