clear; clc; close all;

rng(1);

script_dir = string(fileparts(mfilename("fullpath")));
matlab_dir = string(fileparts(script_dir));
project_root = string(fileparts(matlab_dir));

addpath(char(project_root));
add_toolboxes();

raw_data_dir = fullfile(project_root, "experiments", "raw");
figure_dir = fullfile(project_root, "experiments", "figures");
if ~isfolder(raw_data_dir)
    mkdir(raw_data_dir);
end
if ~isfolder(figure_dir)
    mkdir(figure_dir);
end

m = 3000;
n = 1000;

% Fast scaled power-law singular values
scale = 20;
[A_power, s_power] = controlled_svd_matrix_generator(m, n, "scaled_power", scale, 1);

% Slow-decay singular values
beta = 1;
[A_slow, s_slow] = controlled_svd_matrix_generator(m, n, "power", beta, 2);


data_file = fullfile(raw_data_dir, "synthetic_svd_test_data.mat");
save(data_file, "A_power", "s_power", "A_slow", "s_slow", "m", "n");

fig = figure;
semilogy(1:n, s_power, "LineWidth", 1.5);
hold on;
semilogy(1:n, s_slow, "LineWidth", 1.5);
grid on;
xlabel("j");
ylabel("\sigma_j");
legend("fast scaled power law: (1+(j-1)/20)^{-7}", "slow decay: j^{-1}");
title("Prescribed singular value decay");

save_figure(fig, figure_dir, "experiment1_power_decay_singular_values");

function save_figure(fig, figure_dirs, file_stem)

    axes_handles = findall(fig, "Type", "axes");
    for k = 1:numel(axes_handles)
        if isprop(axes_handles(k), "Toolbar")
            axes_handles(k).Toolbar.Visible = "off";
        end
    end
    drawnow;

    for idx = 1:numel(figure_dirs)
        png_path = fullfile(figure_dirs(idx), file_stem + ".png");
        fig_path = fullfile(figure_dirs(idx), file_stem + ".fig");

        exportgraphics(fig, char(png_path), "Resolution", 300);
        savefig(fig, char(fig_path));

        fprintf("Saved figure: %s\n", char(png_path));
        fprintf("Saved figure: %s\n", char(fig_path));
    end
end
