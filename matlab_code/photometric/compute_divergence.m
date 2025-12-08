function f = compute_divergence(p, q, dx, dy)
% COMPUTE_DIVERGENCE Compute divergence of gradient field
%   f = compute_divergence(p, q, dx, dy)
%
%   Computes: f = dp/dx + dq/dy
%
%   Inputs:
%       p    - dz/dx gradient field
%       q    - dz/dy gradient field
%       dx, dy - grid spacing
%
%   Output:
%       f - divergence field (RHS for Poisson equation)

    % Use central differences
    [~, dp_dx] = gradient(p, dx, dy);
    [dq_dy, ~] = gradient(q, dx, dy);
    
    f = dp_dx + dq_dy;
end
