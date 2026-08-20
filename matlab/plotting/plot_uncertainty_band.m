function band = plot_uncertainty_band(ax, x, lower, upper, color, varargin)
%PLOT_UNCERTAINTY_BAND Plot a non-legend uncertainty or seed-range band.

validateattributes(ax, {'matlab.graphics.axis.Axes'}, {'scalar'}, mfilename, 'ax');
validateattributes(x, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, 'x');
validateattributes(lower, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, 'lower');
validateattributes(upper, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, 'upper');
validateattributes(color, {'numeric'}, ...
    {'vector', 'numel', 3, '>=', 0, '<=', 1}, mfilename, 'color');

x = x(:);
lower = lower(:);
upper = upper(:);
if numel(x) ~= numel(lower) || numel(x) ~= numel(upper)
    error('plot_uncertainty_band:SizeMismatch', ...
        'X, LOWER, and UPPER must have the same number of elements.');
end
if any(lower > upper)
    error('plot_uncertainty_band:InvalidInterval', ...
        'Each LOWER value must be less than or equal to UPPER.');
end
if strcmp(ax.YScale, 'log') && any(lower <= 0)
    error('plot_uncertainty_band:NonpositiveLogValue', ...
        'A logarithmic y-axis requires strictly positive band values.');
end

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'FaceAlpha', 0.14, ...
    @(v) validateattributes(v, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
parse(p, varargin{:});

held = ishold(ax);
x_scale = ax.XScale;
y_scale = ax.YScale;
hold(ax, 'on');
band = fill(ax, [x; flipud(x)], [lower; flipud(upper)], color, ...
    'FaceAlpha', p.Results.FaceAlpha, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off', ...
    'DisplayName', '');
ax.XScale = x_scale;
ax.YScale = y_scale;
uistack(band, 'bottom');
if ~held
    hold(ax, 'off');
end
end
