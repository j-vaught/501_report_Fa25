% runner.m - Main experiment runner for photometric stereo pipeline
% Compares all solvers across all test surfaces
%
% Usage: run this script from matlab_code directory
%   >> runner

clear; clc; close all;

% Add paths
addpath('surfaces');
addpath('solvers');
addpath('photometric');

% Load config
run('config.m');

fprintf('Photometric Stereo Solver Comparison\n');
fprintf('=====================================\n');
fprintf('Grid: %d x %d\n\n', NX, NY);

% Define surfaces
surfaces = struct();
surfaces.gaussian = @() create_gaussian_surface(NX, NY);
surfaces.sphere = @() create_sphere_surface(NX, NY);
surfaces.cone = @() create_cone_surface(NX, NY);
surfaces.saddle = @() create_saddle_surface(NX, NY);
surfaces.sinusoid = @() create_sinusoid_surface(NX, NY);
surfaces.peaks = @() create_peaks_surface(NX, NY);

% Define solvers
solvers = struct();
solvers.FFT = @(f, dx, dy) solve_poisson_fft(f, dx, dy);
solvers.DCT = @(f, dx, dy) solve_poisson_dct(f, dx, dy);
solvers.CG = @(f, dx, dy) solve_poisson_cg(f, dx, dy);

% Results storage
surface_names = fieldnames(surfaces);
solver_names = fieldnames(solvers);
results = zeros(length(surface_names), length(solver_names));

% Header
fprintf('%-12s', 'Surface');
for j = 1:length(solver_names)
    fprintf('%12s', solver_names{j});
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 12 + 12*length(solver_names)));

% Run experiments
for i = 1:length(surface_names)
    sname = surface_names{i};
    create_fn = surfaces.(sname);
    
    % Create surface
    [X, Y, Z_true, dx, dy] = create_fn();
    Z_true = Z_true - mean(Z_true(:));
    
    % Compute true normals
    N_true = height_to_normals(Z_true, dx, dy);
    
    % Compute gradients from normals
    [p, q] = normals_to_gradients(N_true);
    
    % Compute divergence (RHS for Poisson)
    f = compute_divergence(p, q, dx, dy);
    
    fprintf('%-12s', sname);
    
    for j = 1:length(solver_names)
        solver_name = solver_names{j};
        solve_fn = solvers.(solver_name);
        
        % Solve
        Z_est = solve_fn(f, dx, dy);
        
        % Compute RMSE
        rmse = sqrt(mean((Z_true(:) - Z_est(:)).^2));
        results(i, j) = rmse;
        
        fprintf('%12.4f', rmse);
    end
    fprintf('\n');
end

fprintf('%s\n', repmat('=', 1, 12 + 12*length(solver_names)));
fprintf('Lower RMSE = better reconstruction\n\n');

% Find best solver for each surface
fprintf('Best solver per surface:\n');
for i = 1:length(surface_names)
    [min_rmse, best_idx] = min(results(i,:));
    fprintf('  %s: %s (RMSE = %.4f)\n', surface_names{i}, solver_names{best_idx}, min_rmse);
end

fprintf('\nDone!\n');
