function save_surface_figures(X, Y, Z_true, Z_est, output_dir, shape_name, solver_name)
% SAVE_SURFACE_FIGURES Generate and save all visualization figures
%   save_surface_figures(X, Y, Z_true, Z_est, output_dir, shape_name, solver_name)
%
%   Saves 7 figures: 3d_true, 3d_est, depth_true, depth_est, error_map, profile, histogram

    % Create output directory
    full_dir = fullfile(output_dir, shape_name, solver_name);
    if ~exist(full_dir, 'dir')
        mkdir(full_dir);
    end
    
    error = Z_true - Z_est;
    rmse = sqrt(mean(error(:).^2));
    
    % 1. 3D True Surface
    fig = figure('Visible', 'off');
    surf(X, Y, Z_true, 'EdgeColor', 'none');
    title(sprintf('%s - Ground Truth', shape_name));
    colormap parula;
    colorbar;
    saveas(fig, fullfile(full_dir, '3d_true.png'));
    close(fig);
    
    % 2. 3D Estimated Surface
    fig = figure('Visible', 'off');
    surf(X, Y, Z_est, 'EdgeColor', 'none');
    title(sprintf('%s - %s (RMSE=%.4f)', shape_name, solver_name, rmse));
    colormap parula;
    colorbar;
    saveas(fig, fullfile(full_dir, '3d_est.png'));
    close(fig);
    
    % 3. Depth True
    fig = figure('Visible', 'off');
    imagesc(Z_true);
    axis image;
    title('True Depth');
    colorbar;
    saveas(fig, fullfile(full_dir, 'depth_true.png'));
    close(fig);
    
    % 4. Depth Estimated
    fig = figure('Visible', 'off');
    imagesc(Z_est);
    axis image;
    title('Estimated Depth');
    colorbar;
    saveas(fig, fullfile(full_dir, 'depth_est.png'));
    close(fig);
    
    % 5. Error Map
    fig = figure('Visible', 'off');
    imagesc(error);
    axis image;
    caxis([-max(abs(error(:))), max(abs(error(:)))]);
    colormap(redblue);
    title('Error Map');
    colorbar;
    saveas(fig, fullfile(full_dir, 'error_map.png'));
    close(fig);
    
    % 6. Profile (center row)
    fig = figure('Visible', 'off');
    mid = floor(size(Z_true, 1) / 2);
    plot(Z_true(mid, :), 'b-', 'LineWidth', 2); hold on;
    plot(Z_est(mid, :), 'r--', 'LineWidth', 2);
    legend('True', 'Estimated');
    title('Center Profile');
    xlabel('X index');
    ylabel('Height');
    saveas(fig, fullfile(full_dir, 'profile.png'));
    close(fig);
    
    % 7. Error Histogram
    fig = figure('Visible', 'off');
    histogram(error(:), 50);
    title(sprintf('Error Histogram (RMSE=%.4f)', rmse));
    xlabel('Error');
    ylabel('Count');
    saveas(fig, fullfile(full_dir, 'histogram.png'));
    close(fig);
end

function c = redblue(m)
% REDBLUE Red-White-Blue colormap
    if nargin < 1, m = 256; end
    mid = ceil(m/2);
    r = [linspace(0, 1, mid), ones(1, m-mid)];
    g = [linspace(0, 1, mid), linspace(1, 0, m-mid)];
    b = [ones(1, mid), linspace(1, 0, m-mid)];
    c = [r', g', b'];
end
