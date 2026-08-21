function paths = export_numerical_figure(fig, outputStem, varargin)
%EXPORT_NUMERICAL_FIGURE Export a vector PDF and optional PNG/FIG copies.

validateattributes(fig, {'matlab.ui.Figure'}, {'scalar'}, mfilename, 'fig');
if ~(ischar(outputStem) || (isstring(outputStem) && isscalar(outputStem)))
    error('export_numerical_figure:InvalidPath', ...
        'OUTPUTSTEM must be a character vector or scalar string.');
end

p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'WritePNG', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'PNGResolution', 300, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 72}));
addParameter(p, 'WriteFIG', false, @(x) islogical(x) && isscalar(x));
parse(p, varargin{:});
opt = p.Results;

outputStem = char(outputStem);
[folder, name, extension] = fileparts(outputStem);
if isempty(name)
    error('export_numerical_figure:InvalidPath', ...
        'OUTPUTSTEM must include a file name.');
end
if ismember(lower(extension), {'.pdf', '.png', '.fig'})
    outputStem = fullfile(folder, name);
elseif ~isempty(extension)
    error('export_numerical_figure:UnsupportedExtension', ...
        'Use a path without an extension or with .pdf, .png, or .fig.');
end

[folder, ~, ~] = fileparts(outputStem);
if ~isempty(folder) && ~isfolder(folder)
    mkdir(folder);
end

paths.pdf = [outputStem, '.pdf'];
paths.png = '';
paths.fig = '';

originalUnits = fig.Units;
fig.Units = 'centimeters';
figureSizeCm = fig.Position(3:4);
fig.Units = originalUnits;

drawnow;
exportgraphics(fig, paths.pdf, ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'white', ...
    'Width', figureSizeCm(1), ...
    'Height', figureSizeCm(2), ...
    'Units', 'centimeters', ...
    'Padding', 'figure', ...
    'PreserveAspectRatio', 'off');

if opt.WritePNG
    paths.png = [outputStem, '.png'];
    exportgraphics(fig, paths.png, ...
        'Resolution', opt.PNGResolution, ...
        'BackgroundColor', 'white', ...
        'Width', figureSizeCm(1), ...
        'Height', figureSizeCm(2), ...
        'Units', 'centimeters', ...
        'Padding', 'figure', ...
        'PreserveAspectRatio', 'off');
end

if opt.WriteFIG
    paths.fig = [outputStem, '.fig'];
    savefig(fig, paths.fig);
end
end
