clear; clc; close all;

script_dir = string(fileparts(mfilename("fullpath")));
matlab_dir = string(fileparts(script_dir));
project_root = string(fileparts(matlab_dir));

% Add the classified MATLAB source folders and Tensor Toolbox.
addpath(char(matlab_dir));
add_toolboxes();

processed_dir = fullfile(project_root, "experiments", "processed");
figure_dir = fullfile(project_root, "experiments", "figures");
if ~isfolder(processed_dir)
    mkdir(processed_dir);
end
if ~isfolder(figure_dir)
    mkdir(figure_dir);
end

%% Check that Tensor Toolbox and rsvd.m are on path
which tensor -all
which tenmat -all
which ttm -all
which rsvd

%% Parameters
N = 100;          % start with 100 for debugging; later try 160
scale = 20;
seed = 1;

ranks_list = [5, 10, 20, 40, 60, 80];
p = 10;
ntrials = 5;
order = [1, 2, 3];

ranks_list = ranks_list(ranks_list + p <= N);

%% Generate controlled tensor
[X, s, ~] = controlled_tucker_tensor_generator(N, scale, seed);
normX = norm(X);

fprintf("\nControlled tensor size: %d x %d x %d\n", N, N, N);
fprintf("Tensor Frobenius norm: %.6e\n", normX);

%% Sanity check: singular values of mode-1 unfolding
X1 = double(tenmat(X, 1));
sv1 = svd(X1, "econ");
rel_sv_error = norm(sv1 - s) / norm(s);
fprintf("Relative singular-value check, mode 1: %.3e\n", rel_sv_error);

%% Storage
num_ranks = numel(ranks_list);

Rank = zeros(num_ranks,1);

HOSVD_RelErr = zeros(num_ranks,1);
STHOSVD_RelErr = zeros(num_ranks,1);
RHOSVD_RelErrMean = zeros(num_ranks,1);
RHOSVD_RelErrStd = zeros(num_ranks,1);
RSTHOSVD_RelErrMean = zeros(num_ranks,1);
RSTHOSVD_RelErrStd = zeros(num_ranks,1);
MLN_RelErrMean = zeros(num_ranks,1);
MLN_RelErrStd = zeros(num_ranks,1);

HOSVD_Time = zeros(num_ranks,1);
STHOSVD_Time = zeros(num_ranks,1);
RHOSVD_TimeMean = zeros(num_ranks,1);
RHOSVD_TimeStd = zeros(num_ranks,1);
RSTHOSVD_TimeMean = zeros(num_ranks,1);
RSTHOSVD_TimeStd = zeros(num_ranks,1);
MLN_TimeMean = zeros(num_ranks,1);
MLN_TimeStd = zeros(num_ranks,1);
MLN_CondWMaxMean = zeros(num_ranks,1);
MLN_CondWMaxStd = zeros(num_ranks,1);

Speedup_RHOSVD = zeros(num_ranks,1);
Speedup_RSTHOSVD = zeros(num_ranks,1);
Speedup_MLN = zeros(num_ranks,1);

%% Main loop
for k = 1:num_ranks
    r = ranks_list(k);
    ranks = [r, r, r];

    fprintf("\n=====================================\n");
    fprintf("Target multilinear rank: (%d,%d,%d)\n", r, r, r);
    fprintf("=====================================\n");

    %% Deterministic HOSVD
    [~,~,Xhat_hosvd,t_hosvd] = truncated_hosvd(X, ranks);
    err_hosvd = norm(X - Xhat_hosvd) / normX;

    %% Deterministic STHOSVD
    [~,~,Xhat_sthosvd,t_sthosvd] = truncated_sthosvd(X, ranks, order);
    err_sthosvd = norm(X - Xhat_sthosvd) / normX;

    %% Randomized HOSVD trials
    errs_rhosvd = zeros(ntrials,1);
    times_rhosvd = zeros(ntrials,1);

    %% Randomized STHOSVD trials
    errs_rsthosvd = zeros(ntrials,1);
    times_rsthosvd = zeros(ntrials,1);

    %% Multilinear Nystrom trials
    errs_mln = zeros(ntrials,1);
    times_mln = zeros(ntrials,1);
    cond_mln = zeros(ntrials,1);

    for trial = 1:ntrials
        rng(1000 + 100*k + trial);

        [~,~,Xhat_rhosvd,t_rhosvd] = r_hosvd(X, ranks, p);
        errs_rhosvd(trial) = norm(X - Xhat_rhosvd) / normX;
        times_rhosvd(trial) = t_rhosvd;

        rng(2000 + 100*k + trial);

        [~,~,Xhat_rsthosvd,t_rsthosvd] = r_sthosvd(X, ranks, p, order);
        errs_rsthosvd(trial) = norm(X - Xhat_rsthosvd) / normX;
        times_rsthosvd(trial) = t_rsthosvd;

        rng(3000 + 100*k + trial);

        [~,~,Xhat_mln,info_mln,t_mln] = multilinear_nystrom(X, ranks, p * ones(1, numel(ranks)));
        errs_mln(trial) = norm(X - Xhat_mln) / normX;
        times_mln(trial) = t_mln;
        cond_mln(trial) = max(info_mln.condW);
    end

    %% Store results
    Rank(k) = r;

    HOSVD_RelErr(k) = err_hosvd;
    STHOSVD_RelErr(k) = err_sthosvd;

    RHOSVD_RelErrMean(k) = mean(errs_rhosvd);
    RHOSVD_RelErrStd(k) = std(errs_rhosvd);

    RSTHOSVD_RelErrMean(k) = mean(errs_rsthosvd);
    RSTHOSVD_RelErrStd(k) = std(errs_rsthosvd);
    MLN_RelErrMean(k) = mean(errs_mln);
    MLN_RelErrStd(k) = std(errs_mln);

    HOSVD_Time(k) = t_hosvd;
    STHOSVD_Time(k) = t_sthosvd;

    RHOSVD_TimeMean(k) = mean(times_rhosvd);
    RHOSVD_TimeStd(k) = std(times_rhosvd);

    RSTHOSVD_TimeMean(k) = mean(times_rsthosvd);
    RSTHOSVD_TimeStd(k) = std(times_rsthosvd);
    MLN_TimeMean(k) = mean(times_mln);
    MLN_TimeStd(k) = std(times_mln);
    MLN_CondWMaxMean(k) = mean(cond_mln);
    MLN_CondWMaxStd(k) = std(cond_mln);

    Speedup_RHOSVD(k) = t_hosvd / mean(times_rhosvd);
    Speedup_RSTHOSVD(k) = t_sthosvd / mean(times_rsthosvd);
    Speedup_MLN(k) = t_hosvd / mean(times_mln);

    fprintf("HOSVD      err %.3e | time %.3f\n", err_hosvd, t_hosvd);
    fprintf("R-HOSVD    err %.3e ± %.1e | time %.3f ± %.3f | speedup %.2fx\n", ...
        mean(errs_rhosvd), std(errs_rhosvd), mean(times_rhosvd), std(times_rhosvd), Speedup_RHOSVD(k));
    fprintf("STHOSVD    err %.3e | time %.3f\n", err_sthosvd, t_sthosvd);
    fprintf("R-STHOSVD  err %.3e ± %.1e | time %.3f ± %.3f | speedup %.2fx\n", ...
        mean(errs_rsthosvd), std(errs_rsthosvd), mean(times_rsthosvd), std(times_rsthosvd), Speedup_RSTHOSVD(k));
    fprintf("MLN        err %.3e ± %.1e | time %.3f ± %.3f | HOSVD/MLN speedup %.2fx\n", ...
        mean(errs_mln), std(errs_mln), mean(times_mln), std(times_mln), Speedup_MLN(k));
end

%% Save table
T = table(Rank, ...
    HOSVD_RelErr, RHOSVD_RelErrMean, RHOSVD_RelErrStd, ...
    STHOSVD_RelErr, RSTHOSVD_RelErrMean, RSTHOSVD_RelErrStd, ...
    MLN_RelErrMean, MLN_RelErrStd, ...
    HOSVD_Time, RHOSVD_TimeMean, RHOSVD_TimeStd, Speedup_RHOSVD, ...
    STHOSVD_Time, RSTHOSVD_TimeMean, RSTHOSVD_TimeStd, Speedup_RSTHOSVD, ...
    MLN_TimeMean, MLN_TimeStd, Speedup_MLN, MLN_CondWMaxMean, MLN_CondWMaxStd);

disp(T);
writetable(T, fullfile(processed_dir, "experiment2_power_tensor_results.csv"));

%% Singular values
fig = figure;
semilogy(1:N, s, "LineWidth", 1.5);
grid on;
xlabel("j");
ylabel("\sigma_j");
title("Controlled tensor: prescribed mode-unfolding singular values");
save_plot_bundle(fig, figure_dir, "experiment2_power_tensor_singular_values");

%% Error plot
fig = figure;
semilogy(Rank, HOSVD_RelErr, "-o", "LineWidth", 1.5);
hold on;
semilogy(Rank, RHOSVD_RelErrMean, "-s", "LineWidth", 1.5);
semilogy(Rank, STHOSVD_RelErr, "-^", "LineWidth", 1.5);
semilogy(Rank, RSTHOSVD_RelErrMean, "-d", "LineWidth", 1.5);
semilogy(Rank, MLN_RelErrMean, "-v", "LineWidth", 1.5);
grid on;
xlabel("target multilinear rank r");
ylabel("relative Frobenius error");
legend("HOSVD", "R-HOSVD", "STHOSVD", "R-STHOSVD", "MLN", "Location", "best");
title("Controlled tensor: approximation error");
save_plot_bundle(fig, figure_dir, "experiment2_power_tensor_error");

%% Runtime plot
fig = figure;
plot(Rank, HOSVD_Time, "-o", "LineWidth", 1.5);
hold on;
plot(Rank, RHOSVD_TimeMean, "-s", "LineWidth", 1.5);
plot(Rank, STHOSVD_Time, "-^", "LineWidth", 1.5);
plot(Rank, RSTHOSVD_TimeMean, "-d", "LineWidth", 1.5);
plot(Rank, MLN_TimeMean, "-v", "LineWidth", 1.5);
grid on;
xlabel("target multilinear rank r");
ylabel("time in seconds");
legend("HOSVD", "R-HOSVD", "STHOSVD", "R-STHOSVD", "MLN", "Location", "best");
title("Controlled tensor: runtime");
save_plot_bundle(fig, figure_dir, "experiment2_power_tensor_runtime");

%% Speedup plot
fig = figure;
plot(Rank, Speedup_RHOSVD, "-o", "LineWidth", 1.5);
hold on;
plot(Rank, Speedup_RSTHOSVD, "-s", "LineWidth", 1.5);
plot(Rank, Speedup_MLN, "-v", "LineWidth", 1.5);
yline(1, "--");
grid on;
xlabel("target multilinear rank r");
ylabel("deterministic time / randomized time");
legend("HOSVD / R-HOSVD", "STHOSVD / R-STHOSVD", "HOSVD / MLN", "Location", "best");
title("Controlled tensor: randomized speedup");
save_plot_bundle(fig, figure_dir, "experiment2_power_tensor_speedup");

function save_plot_bundle(fig, figure_dirs, file_stem)
    hide_axes_toolbar(fig);

    for idx = 1:numel(figure_dirs)
        fig_path = fullfile(figure_dirs(idx), file_stem + ".fig");
        pdf_path = fullfile(figure_dirs(idx), file_stem + ".pdf");
        png_path = fullfile(figure_dirs(idx), file_stem + ".png");

        savefig(fig, char(fig_path));
        exportgraphics(fig, char(pdf_path), "ContentType", "vector");
        exportgraphics(fig, char(png_path), "Resolution", 300);

        fprintf("Saved figure: %s\n", char(fig_path));
        fprintf("Saved figure: %s\n", char(pdf_path));
        fprintf("Saved figure: %s\n", char(png_path));
    end
end

function hide_axes_toolbar(fig)
    axes_handles = findall(fig, "Type", "axes");
    for k = 1:numel(axes_handles)
        if isprop(axes_handles(k), "Toolbar")
            axes_handles(k).Toolbar.Visible = "off";
        end
    end
    drawnow;
end
