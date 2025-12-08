% runner_with_figures.m - Run experiments and generate figures
% Creates publication-quality figures for each surface-solver combination

clear; clc; close all;

% Add paths
addpath('surfaces');
addpath('solvers');
addpath('photometric');
addpath('visualization');

% Load config
run('config.m');

% Create output directory
output_dir = 'output/figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Photometric Stereo Solver Comparison (all 4 solvers)\n');
fprintf('====================================================\n');
fprintf('Grid: %d x %d\n\n', NX, NY);

% Define surfaces
surfaces = struct();
surfaces.gaussian = @() create_gaussian_surface(NX, NY);
surfaces.sphere = @() create_sphere_surface(NX, NY);
surfaces.cone = @() create_cone_surface(NX, NY);
surfaces.saddle = @() create_saddle_surface(NX, NY);
surfaces.sinusoid = @() create_sinusoid_surface(NX, NY);
surfaces.peaks = @() create_peaks_surface(NX, NY);

% Define all 4 solvers
solvers = struct();
solvers.fft = @(f, dx, dy) solve_poisson_fft(f, dx, dy);
solvers.dct = @(f, dx, dy) solve_poisson_dct(f, dx, dy);
solvers.fd = @(f, dx, dy) solve_poisson_fd(f, dx, dy);
solvers.cg = @(f, dx, dy) solve_poisson_cg(f, dx, dy);

surface_names = fieldnames(surfaces);
solver_names = fieldnames(solvers);

% Header
fprintf('%-12s', 'Surface');
for j = 1:length(solver_names)
    fprintf('%10s', upper(solver_names{j}));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 12 + 10*length(solver_names)));

% Results matrix for summary
results = zeros(length(surface_names), length(solver_names));

% Run experiments
for i = 1:length(surface_names)
    sname = surface_names{i};
    create_fn = surfaces.(sname);
    
    % Create surface
    [X, Y, Z_true, dx, dy] = create_fn();
    Z_true = Z_true - mean(Z_true(:));
    
    % Compute true normals
    N_true = height_to_normals(Z_true, dx, dy);
    
    % Compute gradients
    [p, q] = normals_to_gradients(N_true);
    
    % Compute divergence
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
        fprintf('%10.4f', rmse);
        
        % Save figures
        save_surface_figures(X, Y, Z_true, Z_est, output_dir, sname, solver_name);
    end
    fprintf('\n');
end

fprintf('%s\n', repmat('=', 1, 12 + 10*length(solver_names)));

% Find best solver for each surface
fprintf('\nBest solver per surface:\n');
for i = 1:length(surface_names)
    [min_rmse, best_idx] = min(results(i,:));
    fprintf('  %-12s: %s (RMSE = %.6f)\n', surface_names{i}, upper(solver_names{best_idx}), min_rmse);
end

fprintf('\nFigures saved to: %s/\n', output_dir);
fprintf('Total: %d surfaces x %d solvers x 7 figures = %d images\n', ...
    length(surface_names), length(solver_names), length(surface_names)*length(solver_names)*7);
fprintf('Done!\n');
