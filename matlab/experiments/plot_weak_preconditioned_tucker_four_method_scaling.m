function plot_weak_preconditioned_tucker_four_method_scaling()
%PLOT_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING Plot scaling evidence.

experiment_folder = fileparts(mfilename('fullpath'));
matlab_folder = fileparts(experiment_folder);
repository_folder = fileparts(matlab_folder);
addpath(genpath(matlab_folder));

config = weak_preconditioned_tucker_four_method_scaling_config();
processed_folder = fullfile(repository_folder, 'experiments', 'processed');
figure_folder = fullfile(repository_folder, 'experiments', 'figures');
prefix = fullfile(processed_folder, config.outputPrefix);
comparison = readtable(prefix + "_comparison_summary.csv", ...
    'TextType', 'string');
true_residual_history = readtable( ...
    prefix + "_true_residual_history.csv", 'TextType', 'string');
rank_history = readtable(prefix + "_rank_history.csv", ...
    'TextType', 'string');

colours = [ ...
    0.0000, 0.4470, 0.7410; ...
    0.8500, 0.3250, 0.0980; ...
    0.4660, 0.6740, 0.1880; ...
    0.4940, 0.1840, 0.5560];
markers = {'o', 's', '^', 'd'};
line_styles = {'-', '--', '-.', ':'};

residual_figure = figure('Visible', 'off', 'Color', 'w');
residual_layout = tiledlayout(residual_figure, 1, ...
    numel(config.representativeN), 'TileSpacing', 'compact', ...
    'Padding', 'compact');
residual_axes = gobjects(numel(config.representativeN), 1);

residual_minimum = min(true_residual_history.true_relative_residual);
residual_maximum = max(true_residual_history.true_relative_residual);
residual_lower_limit = 0.8 * 10 ^ floor(log10(residual_minimum));
residual_upper_limit = 1.2 * 10 ^ ceil(log10(residual_maximum));

for grid_index = 1:numel(config.representativeN)
    N = config.representativeN(grid_index);
    residual_axes(grid_index) = nexttile(residual_layout);
    hold(residual_axes(grid_index), 'on');

    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        [iteration, residual, minimum_residual, maximum_residual] = ...
            summarise_true_residual_history( ...
            true_residual_history, N, method_id, ...
            config.diagnosticCheckpoints);

        if any(maximum_residual > minimum_residual)
            plot_uncertainty_band(residual_axes(grid_index), iteration, ...
                minimum_residual, maximum_residual, ...
                colours(method_index, :), 'FaceAlpha', 0.12);
        end

        plot(residual_axes(grid_index), iteration, residual, ...
            'Color', colours(method_index, :), ...
            'LineStyle', line_styles{method_index}, ...
            'Marker', markers{method_index}, ...
            'LineWidth', 1.35, 'MarkerFaceColor', 'w', ...
            'DisplayName', config.methodNames(method_index));
    end

    set(residual_axes(grid_index), 'YScale', 'log');
    grid(residual_axes(grid_index), 'on');
    box(residual_axes(grid_index), 'off');
    title(residual_axes(grid_index), sprintf('$N=%d$', N), ...
        'FontWeight', 'normal', 'Interpreter', 'latex');
    xlabel(residual_axes(grid_index), 'Arnoldi iteration $j$', ...
        'Interpreter', 'latex');
    xlim(residual_axes(grid_index), [0, config.maximumIterations]);
    ylim(residual_axes(grid_index), ...
        [residual_lower_limit, residual_upper_limit]);
    xticks(residual_axes(grid_index), config.diagnosticCheckpoints);
    residual_axes(grid_index).FontSize = 8.5;
    residual_axes(grid_index).TickLabelInterpreter = 'latex';
    residual_axes(grid_index).TickDir = 'out';
    residual_axes(grid_index).LineWidth = 0.75;
    residual_axes(grid_index).GridAlpha = 0.12;
    residual_axes(grid_index).GridColor = [0.25, 0.25, 0.25];

    if grid_index == 1
        ylabel(residual_axes(grid_index), ...
            ['$\Vert \mathcal{B}_0-\mathcal{A}_0(\mathcal{X}_j)\Vert_F', ...
            '/\Vert \mathcal{B}_0\Vert_F$'], 'Interpreter', 'latex');
    end
end

residual_legend = legend(residual_axes(1), 'Orientation', 'horizontal', ...
    'NumColumns', 2);
residual_legend.Layout.Tile = 'south';

apply_numerical_figure_style(residual_figure, ...
    'WidthCm', 15.8, 'HeightCm', 7.0);

time_figure = figure('Visible', 'off', 'Color', 'w');
time_axis = axes(time_figure);
hold(time_axis, 'on');

for method_index = 1:numel(config.methodIds)
    method_id = config.methodIds(method_index);
    rows = comparison.method_id == method_id;
    method_data = sortrows(comparison(rows, :), 'N');
    time = method_data.median_solver_time_sec;
    time_lower = time - method_data.minimum_solver_time_sec;
    time_upper = method_data.maximum_solver_time_sec - time;
    errorbar(time_axis, method_data.N, time, time_lower, time_upper, ...
        'Color', colours(method_index, :), ...
        'LineStyle', line_styles{method_index}, ...
        'Marker', markers{method_index}, 'LineWidth', 1.35, ...
        'MarkerFaceColor', 'w', 'DisplayName', ...
        config.methodNames(method_index));
end

set(time_axis, 'YScale', 'log');
format_axis(time_axis, 'Mode size $N$', ...
    'Solver time for 100 Arnoldi products (s)', '');
legend(time_axis, 'Location', 'best', 'NumColumns', 2);
apply_thesis_style(time_figure, 15.8, 8.2);

rank_figure = figure('Visible', 'off', 'Color', 'w');
rank_layout = tiledlayout(rank_figure, 1, numel(config.representativeN), ...
    'TileSpacing', 'compact', 'Padding', 'compact');
rank_axes = gobjects(numel(config.representativeN), 1);
max_rank_value = max(rank_history.maximum_mode_rank);
rank_upper_limit = 20 * ceil(max_rank_value / 20);

for grid_index = 1:numel(config.representativeN)
    N = config.representativeN(grid_index);
    rank_axes(grid_index) = nexttile(rank_layout);
    hold(rank_axes(grid_index), 'on');

    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        [iteration, median_rank, minimum_rank, maximum_rank_by_iteration] = ...
            summarise_rank_history(rank_history, N, method_id, ...
            config.maximumIterations);

        if any(maximum_rank_by_iteration > minimum_rank)
            plot_uncertainty_band(rank_axes(grid_index), iteration, ...
                minimum_rank, maximum_rank_by_iteration, ...
                colours(method_index, :), 'FaceAlpha', 0.12);
        end

        marker_indices = unique([1, 11:10:numel(iteration), ...
            numel(iteration)]);
        plot(rank_axes(grid_index), iteration, median_rank, ...
            'Color', colours(method_index, :), ...
            'LineStyle', line_styles{method_index}, ...
            'Marker', markers{method_index}, ...
            'MarkerIndices', marker_indices, ...
            'LineWidth', 1.35, 'MarkerFaceColor', 'w', ...
            'DisplayName', config.methodNames(method_index));
    end

    xlabel(rank_axes(grid_index), 'Arnoldi iteration $j$', ...
        'Interpreter', 'latex');
    title(rank_axes(grid_index), sprintf('$N=%d$', N), ...
        'FontWeight', 'normal', 'Interpreter', 'latex');
    xlim(rank_axes(grid_index), [0, config.maximumIterations]);
    ylim(rank_axes(grid_index), [0, rank_upper_limit]);
    xticks(rank_axes(grid_index), 0:20:config.maximumIterations);
    yticks(rank_axes(grid_index), 0:20:rank_upper_limit);

    if grid_index == 1
        ylabel(rank_axes(grid_index), 'Maximum Tucker basis rank', ...
            'Interpreter', 'latex');
    end
end

rank_legend = legend(rank_axes(1), 'Orientation', 'horizontal', ...
    'NumColumns', 2);
rank_legend.Layout.Tile = 'south';

apply_numerical_figure_style(rank_figure, ...
    'WidthCm', 15.8, 'HeightCm', 7.0);

residual_stem = fullfile(figure_folder, ...
    config.outputPrefix + "_true_residual_history");
solver_time_stem = fullfile(figure_folder, ...
    config.outputPrefix + "_solver_time_scaling");
maximum_rank_stem = fullfile(figure_folder, ...
    config.outputPrefix + "_maximum_rank_history");
export_numerical_figure(residual_figure, residual_stem, ...
    'WritePNG', true, 'WriteFIG', true);
export_numerical_figure(time_figure, solver_time_stem, ...
    'WritePNG', true, 'WriteFIG', true);
export_numerical_figure(rank_figure, maximum_rank_stem, ...
    'WritePNG', true, 'WriteFIG', true);
close(residual_figure);
close(time_figure);
close(rank_figure);

fprintf('Saved %s.pdf and %s.png\n', residual_stem, residual_stem);
fprintf('Saved %s.pdf and %s.png\n', solver_time_stem, solver_time_stem);
fprintf('Saved %s.pdf and %s.png\n', maximum_rank_stem, maximum_rank_stem);

end

function [iteration, median_residual, minimum_residual, maximum_residual] = ...
    summarise_true_residual_history( ...
    residual_history, N, method_id, diagnostic_checkpoints)

method_rows = residual_history( ...
    residual_history.N == N & residual_history.method_id == method_id, :);
assert(~isempty(method_rows), ...
    'The residual history is missing for N=%d and method %s.', ...
    N, method_id);

seed_values = unique(method_rows.residual_seed);
if all(isnan(seed_values))
    seed_values = NaN;
end

iteration = diagnostic_checkpoints(:);
run_histories = NaN(numel(iteration), numel(seed_values));
for seed_index = 1:numel(seed_values)
    seed = seed_values(seed_index);
    if isnan(seed)
        seed_rows = method_rows(isnan(method_rows.residual_seed), :);
    else
        seed_rows = method_rows(method_rows.residual_seed == seed, :);
    end
    seed_rows = sortrows(seed_rows, 'arnoldi_iteration');
    assert(isequal(seed_rows.arnoldi_iteration, iteration), ...
        'A residual history has incomplete checkpoints.');
    run_histories(:, seed_index) = seed_rows.true_relative_residual;
end

assert(all(isfinite(run_histories) & run_histories > 0, 'all'), ...
    'The completed residual history contains an invalid value.');
median_residual = median(run_histories, 2);
minimum_residual = min(run_histories, [], 2);
maximum_residual = max(run_histories, [], 2);

end

function [iteration, median_rank, minimum_rank, max_rank_value] = ...
    summarise_rank_history(rank_history, N, method_id, maxit)

method_rows = rank_history( ...
    rank_history.N == N & rank_history.method_id == method_id, :);
assert(~isempty(method_rows), ...
    'The rank history is missing for N=%d and method %s.', N, method_id);

seed_values = unique(method_rows.residual_seed);
if all(isnan(seed_values))
    seed_values = NaN;
end

iteration = (0:maxit).';
run_histories = NaN(numel(iteration), numel(seed_values));

for seed_index = 1:numel(seed_values)
    seed = seed_values(seed_index);
    if isnan(seed)
        seed_rows = method_rows(isnan(method_rows.residual_seed), :);
    else
        seed_rows = method_rows(method_rows.residual_seed == seed, :);
    end
    seed_rows = sortrows(seed_rows, 'basis_index');

    basis_iteration = seed_rows.basis_index - 1;
    running_maximum = cummax(seed_rows.maximum_mode_rank);
    next_history = NaN(size(iteration));
    next_history(basis_iteration + 1) = running_maximum;

    for iteration_index = 2:numel(iteration)
        if isnan(next_history(iteration_index))
            next_history(iteration_index) = next_history(iteration_index - 1);
        end
    end
    run_histories(:, seed_index) = next_history;
end

assert(all(isfinite(run_histories), 'all'), ...
    'The completed rank history contains a missing value.');
median_rank = median(run_histories, 2);
minimum_rank = min(run_histories, [], 2);
max_rank_value = max(run_histories, [], 2);

end

function format_axis(axis_handle, x_label, y_label, title_text)

grid(axis_handle, 'on');
box(axis_handle, 'off');
xlabel(axis_handle, x_label, 'Interpreter', 'latex');
ylabel(axis_handle, y_label, 'Interpreter', 'latex');
if strlength(string(title_text)) > 0
    title(axis_handle, title_text, 'FontWeight', 'normal', ...
        'Interpreter', 'latex');
end
axis_handle.FontSize = 8.5;
axis_handle.TickLabelInterpreter = 'latex';
axis_handle.TickDir = 'out';
axis_handle.LineWidth = 0.75;
axis_handle.GridAlpha = 0.12;
axis_handle.GridColor = [0.25, 0.25, 0.25];
axis_handle.XTick = [150, 200, 250];
xlim(axis_handle, [140, 260]);

end

function apply_thesis_style(figure_handle, width_cm, height_cm)

figure_handle.Units = 'centimeters';
figure_handle.Position = [2, 2, width_cm, height_cm];
figure_handle.PaperUnits = 'centimeters';
figure_handle.PaperPositionMode = 'auto';
figure_handle.Renderer = 'painters';

legend_handles = findall(figure_handle, 'Type', 'legend');
for legend_handle = reshape(legend_handles, 1, [])
    legend_handle.Interpreter = 'latex';
    legend_handle.FontSize = 7.8;
    legend_handle.Box = 'off';
end

end
