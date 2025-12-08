function lights = make_rotating_lights(num_lights, elevation_deg)
% MAKE_ROTATING_LIGHTS Generate light directions uniformly around azimuth
%   lights = make_rotating_lights(num_lights, elevation_deg)
%
%   Inputs:
%       num_lights    - number of light directions (default: 32)
%       elevation_deg - elevation angle in degrees (default: 45)
%
%   Output:
%       lights - num_lights x 3 matrix of unit light vectors

    if nargin < 1, num_lights = 32; end
    if nargin < 2, elevation_deg = 45; end
    
    elev_rad = deg2rad(elevation_deg);
    azimuth = linspace(0, 2*pi, num_lights + 1);
    azimuth = azimuth(1:end-1);  % Remove duplicate at 2*pi
    
    lights = zeros(num_lights, 3);
    for i = 1:num_lights
        lights(i, 1) = cos(elev_rad) * cos(azimuth(i));  % x
        lights(i, 2) = cos(elev_rad) * sin(azimuth(i));  % y
        lights(i, 3) = sin(elev_rad);                     % z
    end
end
