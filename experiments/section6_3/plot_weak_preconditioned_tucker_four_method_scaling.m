function plot_weak_preconditioned_tucker_four_method_scaling()
%PLOT_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING Plot scaling evidence.

experimentFolder = fileparts(mfilename('fullpath'));
matlabFolder = fileparts(experimentFolder);
repositoryFolder = fileparts(matlabFolder);
addpath(repositoryFolder);
setup();

config = weak_preconditioned_tucker_four_method_scaling_config();
processedFolder = fullfile(repositoryFolder, 'experiments', 'processed');
figureFolder = fullfile(repositoryFolder, 'experiments', 'figures');
prefix = fullfile(processedFolder, config.outputPrefix);
comparison = readtable(prefix + "_comparison_summary.csv", ...
    'TextType', 'string');
trueResidualHistory = readtable( ...
    prefix + "_true_residual_history.csv", 'TextType', 'string');
rankHistory = readtable(prefix + "_rank_history.csv", ...
    'TextType', 'string');

colours = [ ...
    0.0000, 0.4470, 0.7410; ...
    0.8500, 0.3250, 0.0980; ...
    0.4660, 0.6740, 0.1880; ...
    0.4940, 0.1840, 0.5560];
markers = {'o', 's', '^', 'd'};
lineStyles = {'-', '--', '-.', ':'};

residualFigure = figure('Visible', 'off', 'Color', 'w');
residualLayout = tiledlayout(residualFigure, 1, ...
    numel(config.representativeN), 'TileSpacing', 'compact', ...
    'Padding', 'compact');
residualAxes = gobjects(numel(config.representativeN), 1);

residualMinimum = min(trueResidualHistory.true_relative_residual);
residualMaximum = max(trueResidualHistory.true_relative_residual);
residualLowerLimit = 0.8 * 10 ^ floor(log10(residualMinimum));
residualUpperLimit = 1.2 * 10 ^ ceil(log10(residualMaximum));

for gridIndex = 1:numel(config.representativeN)
    N = config.representativeN(gridIndex);
    residualAxes(gridIndex) = nexttile(residualLayout);
    hold(residualAxes(gridIndex), 'on');

    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        [iteration, residual, minimumResidual, maximumResidual] = ...
            summarise_true_residual_history( ...
            trueResidualHistory, N, methodId, ...
            config.diagnosticCheckpoints);

        if any(maximumResidual > minimumResidual)
            plot_uncertainty_band(residualAxes(gridIndex), iteration, ...
                minimumResidual, maximumResidual, ...
                colours(methodIndex, :), 'FaceAlpha', 0.12);
        end

        plot(residualAxes(gridIndex), iteration, residual, ...
            'Color', colours(methodIndex, :), ...
            'LineStyle', lineStyles{methodIndex}, ...
            'Marker', markers{methodIndex}, ...
            'LineWidth', 1.35, 'MarkerFaceColor', 'w', ...
            'DisplayName', config.methodNames(methodIndex));
    end

    set(residualAxes(gridIndex), 'YScale', 'log');
    grid(residualAxes(gridIndex), 'on');
    box(residualAxes(gridIndex), 'off');
    title(residualAxes(gridIndex), sprintf('$N=%d$', N), ...
        'FontWeight', 'normal', 'Interpreter', 'latex');
    xlabel(residualAxes(gridIndex), 'Arnoldi iteration $j$', ...
        'Interpreter', 'latex');
    xlim(residualAxes(gridIndex), [0, config.maximumIterations]);
    ylim(residualAxes(gridIndex), ...
        [residualLowerLimit, residualUpperLimit]);
    xticks(residualAxes(gridIndex), config.diagnosticCheckpoints);
    residualAxes(gridIndex).FontSize = 8.5;
    residualAxes(gridIndex).TickLabelInterpreter = 'latex';
    residualAxes(gridIndex).TickDir = 'out';
    residualAxes(gridIndex).LineWidth = 0.75;
    residualAxes(gridIndex).GridAlpha = 0.12;
    residualAxes(gridIndex).GridColor = [0.25, 0.25, 0.25];

    if gridIndex == 1
        ylabel(residualAxes(gridIndex), ...
            ['$\Vert \mathcal{B}_0-\mathcal{A}_0(\mathcal{X}_j)\Vert_F', ...
            '/\Vert \mathcal{B}_0\Vert_F$'], 'Interpreter', 'latex');
    end
end

residualLegend = legend(residualAxes(1), 'Orientation', 'horizontal', ...
    'NumColumns', 2);
residualLegend.Layout.Tile = 'south';

apply_numerical_figure_style(residualFigure, ...
    'WidthCm', 15.8, 'HeightCm', 7.0);

timeFigure = figure('Visible', 'off', 'Color', 'w');
timeAxis = axes(timeFigure);
hold(timeAxis, 'on');

for methodIndex = 1:numel(config.methodIds)
    methodId = config.methodIds(methodIndex);
    rows = comparison.method_id == methodId;
    methodData = sortrows(comparison(rows, :), 'N');
    time = methodData.median_solver_time_sec;
    timeLower = time - methodData.minimum_solver_time_sec;
    timeUpper = methodData.maximum_solver_time_sec - time;
    errorbar(timeAxis, methodData.N, time, timeLower, timeUpper, ...
        'Color', colours(methodIndex, :), ...
        'LineStyle', lineStyles{methodIndex}, ...
        'Marker', markers{methodIndex}, 'LineWidth', 1.35, ...
        'MarkerFaceColor', 'w', 'DisplayName', ...
        config.methodNames(methodIndex));
end

set(timeAxis, 'YScale', 'log');
format_axis(timeAxis, 'Mode size $N$', ...
    'Solver time for 100 Arnoldi products (s)', '');
legend(timeAxis, 'Location', 'best', 'NumColumns', 2);
apply_thesis_style(timeFigure, 15.8, 8.2);

rankFigure = figure('Visible', 'off', 'Color', 'w');
rankLayout = tiledlayout(rankFigure, 1, numel(config.representativeN), ...
    'TileSpacing', 'compact', 'Padding', 'compact');
rankAxes = gobjects(numel(config.representativeN), 1);
maximumRank = max(rankHistory.maximum_mode_rank);
rankUpperLimit = 20 * ceil(maximumRank / 20);

for gridIndex = 1:numel(config.representativeN)
    N = config.representativeN(gridIndex);
    rankAxes(gridIndex) = nexttile(rankLayout);
    hold(rankAxes(gridIndex), 'on');

    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        [iteration, medianRank, minimumRank, maximumRankByIteration] = ...
            summarise_rank_history(rankHistory, N, methodId, ...
            config.maximumIterations);

        if any(maximumRankByIteration > minimumRank)
            plot_uncertainty_band(rankAxes(gridIndex), iteration, ...
                minimumRank, maximumRankByIteration, ...
                colours(methodIndex, :), 'FaceAlpha', 0.12);
        end

        markerIndices = unique([1, 11:10:numel(iteration), ...
            numel(iteration)]);
        plot(rankAxes(gridIndex), iteration, medianRank, ...
            'Color', colours(methodIndex, :), ...
            'LineStyle', lineStyles{methodIndex}, ...
            'Marker', markers{methodIndex}, ...
            'MarkerIndices', markerIndices, ...
            'LineWidth', 1.35, 'MarkerFaceColor', 'w', ...
            'DisplayName', config.methodNames(methodIndex));
    end

    xlabel(rankAxes(gridIndex), 'Arnoldi iteration $j$', ...
        'Interpreter', 'latex');
    title(rankAxes(gridIndex), sprintf('$N=%d$', N), ...
        'FontWeight', 'normal', 'Interpreter', 'latex');
    xlim(rankAxes(gridIndex), [0, config.maximumIterations]);
    ylim(rankAxes(gridIndex), [0, rankUpperLimit]);
    xticks(rankAxes(gridIndex), 0:20:config.maximumIterations);
    yticks(rankAxes(gridIndex), 0:20:rankUpperLimit);

    if gridIndex == 1
        ylabel(rankAxes(gridIndex), 'Maximum Tucker basis rank', ...
            'Interpreter', 'latex');
    end
end

rankLegend = legend(rankAxes(1), 'Orientation', 'horizontal', ...
    'NumColumns', 2);
rankLegend.Layout.Tile = 'south';

apply_numerical_figure_style(rankFigure, ...
    'WidthCm', 15.8, 'HeightCm', 7.0);

residualStem = fullfile(figureFolder, ...
    config.outputPrefix + "_true_residual_history");
solverTimeStem = fullfile(figureFolder, ...
    config.outputPrefix + "_solver_time_scaling");
maximumRankStem = fullfile(figureFolder, ...
    config.outputPrefix + "_maximum_rank_history");
export_numerical_figure(residualFigure, residualStem, ...
    'WritePNG', true, 'WriteFIG', true);
export_numerical_figure(timeFigure, solverTimeStem, ...
    'WritePNG', true, 'WriteFIG', true);
export_numerical_figure(rankFigure, maximumRankStem, ...
    'WritePNG', true, 'WriteFIG', true);
close(residualFigure);
close(timeFigure);
close(rankFigure);

fprintf('Saved %s.pdf and %s.png\n', residualStem, residualStem);
fprintf('Saved %s.pdf and %s.png\n', solverTimeStem, solverTimeStem);
fprintf('Saved %s.pdf and %s.png\n', maximumRankStem, maximumRankStem);

end


function [iteration, medianResidual, minimumResidual, maximumResidual] = ...
    summarise_true_residual_history( ...
    residualHistory, N, methodId, diagnosticCheckpoints)

methodRows = residualHistory( ...
    residualHistory.N == N & residualHistory.method_id == methodId, :);
assert(~isempty(methodRows), ...
    'The residual history is missing for N=%d and method %s.', ...
    N, methodId);

seedValues = unique(methodRows.residual_seed);
if all(isnan(seedValues))
    seedValues = NaN;
end

iteration = diagnosticCheckpoints(:);
runHistories = NaN(numel(iteration), numel(seedValues));
for seedIndex = 1:numel(seedValues)
    seed = seedValues(seedIndex);
    if isnan(seed)
        seedRows = methodRows(isnan(methodRows.residual_seed), :);
    else
        seedRows = methodRows(methodRows.residual_seed == seed, :);
    end
    seedRows = sortrows(seedRows, 'arnoldi_iteration');
    assert(isequal(seedRows.arnoldi_iteration, iteration), ...
        'A residual history has incomplete checkpoints.');
    runHistories(:, seedIndex) = seedRows.true_relative_residual;
end

assert(all(isfinite(runHistories) & runHistories > 0, 'all'), ...
    'The completed residual history contains an invalid value.');
medianResidual = median(runHistories, 2);
minimumResidual = min(runHistories, [], 2);
maximumResidual = max(runHistories, [], 2);

end


function [iteration, medianRank, minimumRank, maximumRank] = ...
    summarise_rank_history(rankHistory, N, methodId, maximumIteration)

methodRows = rankHistory( ...
    rankHistory.N == N & rankHistory.method_id == methodId, :);
assert(~isempty(methodRows), ...
    'The rank history is missing for N=%d and method %s.', N, methodId);

seedValues = unique(methodRows.residual_seed);
if all(isnan(seedValues))
    seedValues = NaN;
end

iteration = (0:maximumIteration).';
runHistories = NaN(numel(iteration), numel(seedValues));

for seedIndex = 1:numel(seedValues)
    seed = seedValues(seedIndex);
    if isnan(seed)
        seedRows = methodRows(isnan(methodRows.residual_seed), :);
    else
        seedRows = methodRows(methodRows.residual_seed == seed, :);
    end
    seedRows = sortrows(seedRows, 'basis_index');

    basisIteration = seedRows.basis_index - 1;
    runningMaximum = cummax(seedRows.maximum_mode_rank);
    nextHistory = NaN(size(iteration));
    nextHistory(basisIteration + 1) = runningMaximum;

    % Tucker sGMRES does not form an unused next basis tensor after the
    % final operator product. Carry forward the running maximum so that the
    % value at the final iteration still means the largest rank among all
    % basis tensors formed by that point.
    for iterationIndex = 2:numel(iteration)
        if isnan(nextHistory(iterationIndex))
            nextHistory(iterationIndex) = nextHistory(iterationIndex - 1);
        end
    end
    runHistories(:, seedIndex) = nextHistory;
end

assert(all(isfinite(runHistories), 'all'), ...
    'The completed rank history contains a missing value.');
medianRank = median(runHistories, 2);
minimumRank = min(runHistories, [], 2);
maximumRank = max(runHistories, [], 2);

end


function format_axis(axisHandle, xLabel, yLabel, titleText)

grid(axisHandle, 'on');
box(axisHandle, 'off');
xlabel(axisHandle, xLabel, 'Interpreter', 'latex');
ylabel(axisHandle, yLabel, 'Interpreter', 'latex');
if strlength(string(titleText)) > 0
    title(axisHandle, titleText, 'FontWeight', 'normal', ...
        'Interpreter', 'latex');
end
axisHandle.FontSize = 8.5;
axisHandle.TickLabelInterpreter = 'latex';
axisHandle.TickDir = 'out';
axisHandle.LineWidth = 0.75;
axisHandle.GridAlpha = 0.12;
axisHandle.GridColor = [0.25, 0.25, 0.25];
axisHandle.XTick = [150, 200, 250];
xlim(axisHandle, [140, 260]);

end


function apply_thesis_style(figureHandle, widthCm, heightCm)

figureHandle.Units = 'centimeters';
figureHandle.Position = [2, 2, widthCm, heightCm];
figureHandle.PaperUnits = 'centimeters';
figureHandle.PaperPositionMode = 'auto';
figureHandle.Renderer = 'painters';

legendHandles = findall(figureHandle, 'Type', 'legend');
for legendHandle = reshape(legendHandles, 1, [])
    legendHandle.Interpreter = 'latex';
    legendHandle.FontSize = 7.8;
    legendHandle.Box = 'off';
end

end
