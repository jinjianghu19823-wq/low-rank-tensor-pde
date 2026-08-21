clear; clc; close all;

script_dir = string(fileparts(mfilename("fullpath")));
matlab_dir = string(fileparts(script_dir));
project_root = string(fileparts(matlab_dir));

addpath(char(project_root));
add_toolboxes();

raw_data_dir = fullfile(project_root, "experiments", "raw");
processed_dir = fullfile(project_root, "experiments", "processed");
figure_dir = fullfile(project_root, "experiments", "figures");

data_file = fullfile(raw_data_dir, "synthetic_svd_test_data.mat");
load(data_file);

if ~isfolder(processed_dir)
    mkdir(processed_dir);
end
if ~isfolder(figure_dir)
    mkdir(figure_dir);
end

ranks = [5, 10, 20, 40, 80, 120];
p = 10;
ntrials = 10;

T = compare_svd_rsvd(A_power, "power_decay", ranks, p, ntrials);

disp(T);
writetable(T, fullfile(processed_dir, "experiment1_power_decay_results.csv"));

plot_experiment_results(T, "Fast power-law decay matrix", figure_dir, "experiment1_power_decay");



function plot_experiment_results(T, plot_title, figure_dirs, file_prefix)

    %% Error comparison
    fig = figure;
    semilogy(T.Rank, T.SVD_RelErr, "-o", "LineWidth", 1.5);
    hold on;
    semilogy(T.Rank, T.RSVD_RelErrMean, "-s", "LineWidth", 1.5);
    semilogy(T.Rank, T.GN_RelErrMean, "-d", "LineWidth", 1.5);
    grid on;
    xlabel("target rank r");
    ylabel("relative Frobenius error");
    legend("truncated SVD", "RSVD", "GN", "Location", "best");
    title(plot_title + ": approximation error");
    save_figure(fig, figure_dirs, file_prefix + "_relative_error");

    %% Near-optimality ratio
    fig = figure;
    plot(T.Rank, T.ErrorRatio, "-o", "LineWidth", 1.5);
    hold on;
    plot(T.Rank, T.GN_ErrorRatio, "-d", "LineWidth", 1.5);
    yline(1, "--");
    grid on;
    xlabel("target rank r");
    ylabel("method error / SVD error");
    legend("RSVD", "GN", "Location", "best");
    title(plot_title + ": near-optimality ratio");
    save_figure(fig, figure_dirs, file_prefix + "_error_ratio");

    %% Runtime comparison
    fig = figure;
    plot(T.Rank, T.RSVD_TimeMean, "-s", "LineWidth", 1.5);
    hold on;
    plot(T.Rank, T.GN_TimeMean, "-d", "LineWidth", 1.5);
    yline(T.SVD_Time(1), "--", "SVD time");
    grid on;
    xlabel("target rank r");
    ylabel("time in seconds");
    legend("RSVD time", "GN time", "economy SVD time", "Location", "best");
    title(plot_title + ": runtime");
    save_figure(fig, figure_dirs, file_prefix + "_runtime");

    %% Speedup
    fig = figure;
    plot(T.Rank, T.Speedup, "-o", "LineWidth", 1.5);
    hold on;
    plot(T.Rank, T.GN_Speedup, "-d", "LineWidth", 1.5);
    yline(1, "--");
    grid on;
    xlabel("target rank r");
    ylabel("SVD time / method time");
    legend("RSVD", "GN", "Location", "best");
    title(plot_title + ": randomized speedup");
    save_figure(fig, figure_dirs, file_prefix + "_speedup");
end

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
