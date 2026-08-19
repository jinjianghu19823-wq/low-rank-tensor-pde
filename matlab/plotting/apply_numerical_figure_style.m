function apply_numerical_figure_style(fig, varargin)
%APPLY_NUMERICAL_FIGURE_STYLE Apply restrained thesis-ready MATLAB styling.

validateattributes(fig, {'matlab.ui.Figure'}, {'scalar'}, mfilename, 'fig');
S = numerical_plot_style();

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'WidthCm', S.size.fullWidthCm(1), ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
addParameter(p, 'HeightCm', S.size.fullWidthCm(2), ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
addParameter(p, 'Grid', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'MinorGrid', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'FontName', 'Times New Roman', ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));
addParameter(p, 'AxesFontSize', S.font.axes, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
addParameter(p, 'LabelFontSize', S.font.label, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
addParameter(p, 'LegendFontSize', S.font.legend, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
addParameter(p, 'TitleFontSize', S.font.title, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}));
parse(p, varargin{:});
opt = p.Results;

fig.Color = 'white';
fig.Units = 'centimeters';
position = fig.Position;
position(3:4) = [opt.WidthCm, opt.HeightCm];
fig.Position = position;
fig.PaperUnits = 'centimeters';
fig.PaperPositionMode = 'auto';
fig.InvertHardcopy = 'off';
if isprop(fig, 'Renderer')
    fig.Renderer = 'painters';
end

axesHandles = findall(fig, 'Type', 'axes');
for ax = reshape(axesHandles, 1, [])
    ax.FontName = char(opt.FontName);
    ax.FontSize = opt.AxesFontSize;
    ax.LineWidth = 0.75;
    ax.TickDir = 'out';
    ax.TickLength = [0.012, 0.012];
    ax.TickLabelInterpreter = 'latex';
    ax.Box = 'off';
    ax.Layer = 'top';
    ax.XColor = [0.15, 0.15, 0.15];
    ax.YColor = [0.15, 0.15, 0.15];
    ax.ZColor = [0.15, 0.15, 0.15];

    if opt.Grid
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.GridColor = [0.25, 0.25, 0.25];
        ax.GridAlpha = 0.12;
        ax.GridLineStyle = '-';
    else
        ax.XGrid = 'off';
        ax.YGrid = 'off';
    end

    if opt.MinorGrid
        ax.XMinorGrid = 'on';
        ax.YMinorGrid = 'on';
        ax.MinorGridColor = [0.35, 0.35, 0.35];
        ax.MinorGridAlpha = 0.06;
    else
        ax.XMinorGrid = 'off';
        ax.YMinorGrid = 'off';
    end

    labelHandles = [ax.XLabel, ax.YLabel, ax.ZLabel];
    for label = reshape(labelHandles, 1, [])
        label.Interpreter = 'latex';
        label.FontName = char(opt.FontName);
        label.FontSize = opt.LabelFontSize;
        label.FontWeight = 'normal';
    end

    ax.Title.Interpreter = 'latex';
    ax.Title.FontName = char(opt.FontName);
    ax.Title.FontSize = opt.TitleFontSize;
    ax.Title.FontWeight = 'normal';

    lineHandles = findall(ax, 'Type', 'line');
    for lineHandle = reshape(lineHandles, 1, [])
        if lineHandle.LineWidth < S.lineWidth
            lineHandle.LineWidth = S.lineWidth;
        end
        if ~strcmp(lineHandle.Marker, 'none') && ...
                lineHandle.MarkerSize < S.markerSize
            lineHandle.MarkerSize = S.markerSize;
        end
    end

    if isprop(ax, 'Toolbar')
        ax.Toolbar.Visible = 'off';
    end
end

textHandles = findall(fig, 'Type', 'text');
for textHandle = reshape(textHandles, 1, [])
    if isprop(textHandle, 'Interpreter')
        textHandle.Interpreter = 'latex';
    end
end

legendHandles = findall(fig, 'Type', 'legend');
for legendHandle = reshape(legendHandles, 1, [])
    legendHandle.Interpreter = 'latex';
    legendHandle.FontName = char(opt.FontName);
    legendHandle.FontSize = opt.LegendFontSize;
    legendHandle.Box = 'off';
    legendHandle.Color = 'none';
end

constantLines = findall(fig, 'Type', 'constantline');
for constantLine = reshape(constantLines, 1, [])
    constantLine.LineWidth = S.referenceLineWidth;
    if isprop(constantLine, 'Interpreter')
        constantLine.Interpreter = 'latex';
    end
end

drawnow;
end
