function S = numerical_plot_style()
%NUMERICAL_PLOT_STYLE Stable visual vocabulary for numerical figures.
%   S = NUMERICAL_PLOT_STYLE() returns a colorblind-conscious palette,
%   semantic method encodings, line widths, font sizes, and common sizes.

S.colors.charcoal   = [ 45,  45,  45] / 255;
S.colors.blue       = [  0, 114, 178] / 255;
S.colors.vermillion = [213,  94,   0] / 255;
S.colors.green      = [  0, 158, 115] / 255;
S.colors.purple     = [170,  85, 170] / 255;
S.colors.sky        = [ 86, 180, 233] / 255;
S.colors.orange     = [230, 159,   0] / 255;
S.colors.gray       = [115, 115, 115] / 255;
S.colors.lightGray  = [210, 210, 210] / 255;

S.methods.baseline = struct( ...
    'color', S.colors.charcoal, 'line', '-', 'marker', 'o');
S.methods.primary = struct( ...
    'color', S.colors.blue, 'line', '--', 'marker', 's');
S.methods.secondary = struct( ...
    'color', S.colors.vermillion, 'line', '-.', 'marker', 'd');
S.methods.comparator = struct( ...
    'color', S.colors.green, 'line', ':', 'marker', '^');
S.methods.reference = struct( ...
    'color', S.colors.gray, 'line', '--', 'marker', 'none');

S.lineWidth = 1.35;
S.referenceLineWidth = 0.85;
S.markerSize = 4.5;
S.font.axes = 8.5;
S.font.label = 9.5;
S.font.legend = 8.0;
S.font.title = 9.0;
S.size.fullWidthCm = [15.8, 8.8];
S.size.halfWidthCm = [7.8, 6.0];
S.size.threePanelCm = [15.8, 5.8];
end
