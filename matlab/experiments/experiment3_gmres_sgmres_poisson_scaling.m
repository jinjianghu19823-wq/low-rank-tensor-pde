%% Unrestarted, restarted, and sketched GMRES for the 2D Poisson problem

% The experiment uses five grid sizes:
%
%                 N = 100, 200, 400, 800, and 1000.
%
% For each grid, it compares:
%
%   1. MATLAB built-in unrestarted GMRES;
%   2. MATLAB built-in restarted GMRES(50);
%   3. sGMRES with the sparse map recommended in the paper.
%
% The reported accuracy is the true relative residual
%
%                 ||b - A*x||_2 / ||b||_2.

clear;
clc;
close all;

script_folder = fileparts(mfilename('fullpath'));
matlab_folder = fileparts(script_folder);
project_folder = fileparts(matlab_folder);

addpath(matlab_folder);
add_toolboxes();

processed_folder = fullfile(project_folder, 'experiments', 'processed');

if ~isfolder(processed_folder)
    mkdir(processed_folder);
end

%% 1. Choose the experiment parameters

N_values = [100, 200, 400, 800, 1000];

maximum_iteration = 250;
restart_length = 50;
q = 2;

assert(mod(maximum_iteration, restart_length) == 0, ...
    'The restart length must divide the common iteration budget.');
maximum_restart_cycles = maximum_iteration / restart_length;

target_residual = 1e-6;

% Five independent sketching matrices are used for sGMRES.
seeds = [1, 2, 3, 4, 5];

% Both MATLAB GMRES variants are deterministic, but their runtimes are
% measured three times.
gmres_timing_repetitions = 3;

% Residuals are evaluated at these iteration checkpoints.
checkpoints = [0, 5, 10, 20, 30, 50, 75, 100, ...
    125, 150, 175, 200, 225, 250];

% Nakatsukasa and Tropp recommend these parameters.
s = 2 * (maximum_iteration + 1);
zeta = ceil(2 * log(1 + maximum_iteration));

% Method codes used in the saved CSV files:
%   1 = MATLAB built-in unrestarted GMRES
%   2 = sGMRES with the paper-recommended sparse map
%   3 = MATLAB built-in restarted GMRES
GMRES_CODE = 1;
SGMRES_CODE = 2;
RESTARTED_GMRES_CODE = 3;

fprintf('Maximum Krylov dimension = %d\n', maximum_iteration);
fprintf('Restarted GMRES: length = %d, cycles = %d\n', ...
    restart_length, maximum_restart_cycles);
fprintf('Paper-recommended sketch: s = %d, zeta = %d\n\n', ...
    s, zeta);

%% 2. Perform one small warm-up calculation
% The first call to a MATLAB routine can include one-time setup cost.
% This small calculation is not included in the saved results.

A_warm = poisson2d_dirichlet(20);
b_warm = ones(size(A_warm, 1), 1);

[~, ~, ~, ~, ~] = gmres(A_warm, b_warm, [], eps, 5);
[~, ~, ~, ~, ~] = gmres( ...
    A_warm, b_warm, restart_length, eps, 1);

rng(1, 'twister');
sgmres_trunc_arnoldi( ...
    A_warm, b_warm, zeros(size(b_warm)), 5, q, 20, 4, [0, 5]);

clear A_warm b_warm;

%% 3. Prepare empty matrices for the saved results

number_of_N_values = length(N_values);
number_of_seeds = length(seeds);
number_of_checkpoints = length(checkpoints);

% There are three timing runs for each deterministic GMRES variant and five
% sGMRES sketch runs for each N.
number_of_run_rows = number_of_N_values * ...
    (2 * gmres_timing_repetitions + number_of_seeds);

% Columns:
% 1 N, 2 m, 3 method code, 4 run ID, 5 restart length, 6 q,
% 7 s, 8 zeta, 9 actual iteration, 10 runtime,
% 11 final true residual, 12 reached target, 13 MATLAB solver flag.
run_results = zeros(number_of_run_rows, 13);

% There is one residual curve for every individual run.
number_of_history_rows = number_of_N_values * ...
    (2 + number_of_seeds) * number_of_checkpoints;

% Columns:
% 1 N, 2 m, 3 method code, 4 run ID, 5 restart length, 6 q,
% 7 s, 8 zeta, 9 iteration, 10 true relative residual.
history_results = zeros(number_of_history_rows, 10);

run_row = 0;
history_row = 0;

%% 4. Run all five grid sizes

for N_index = 1:number_of_N_values

    N = N_values(N_index);
    m = N^2;

    % sGMRES stores one m-by-maximum_iteration basis. The sketched image of
    % A*B is formed one column at a time, so the full m-by-maximum_iteration
    % product A*B is not stored. This keeps the N = 1000 run within the
    % available memory while leaving the mathematical method unchanged.
    estimated_basis_gib = ...
        8 * m * maximum_iteration / 2^30;
    estimated_sparse_map_gib = ...
        (16 * zeta * m + 8 * (m + 1)) / 2^30;

    fprintf('====================================================\n');
    fprintf('Starting N = %d, m = %d\n', N, m);
    fprintf(['Estimated sGMRES basis = %.2f GiB; ' ...
        'sparse map = %.2f GiB\n'], ...
        estimated_basis_gib, estimated_sparse_map_gib);
    fprintf('====================================================\n');

    %% 4a. Construct the Poisson system

    A = poisson2d_dirichlet(N);

    % The same seed makes the manufactured problem reproducible.
    rng(120, 'twister');
    x_true = randn(m, 1);

    b = A * x_true;
    x0 = zeros(m, 1);

    norm_b = norm(b);

    %% 4b. Run MATLAB built-in unrestarted GMRES
    %
    % Passing [] as the restart argument selects unrestarted GMRES.
    % A tolerance of eps asks MATLAB to continue up to the iteration
    % budget unless machine precision is reached first.

    gmres_runtimes = zeros(gmres_timing_repetitions, 1);

    for timing_repetition = 1:gmres_timing_repetitions

        gmres_timer = tic;

        [x_gmres_repeat, gmres_flag_repeat, ~, ...
            gmres_iteration_repeat, ...
            gmres_resvec_repeat] = ...
            gmres(A, b, [], eps, maximum_iteration, [], [], x0);

        gmres_runtime_repeat = toc(gmres_timer);

        gmres_actual_iteration_repeat = length(gmres_resvec_repeat) - 1;
        assert(gmres_actual_iteration_repeat == gmres_iteration_repeat(2), ...
            'Unrestarted GMRES iteration outputs are inconsistent.');
        gmres_final_true_residual_repeat = ...
            norm(b - A * x_gmres_repeat) / norm_b;

        gmres_reached_target_repeat = ...
            gmres_final_true_residual_repeat <= target_residual;

        gmres_runtimes(timing_repetition) = gmres_runtime_repeat;

        run_row = run_row + 1;

        run_results(run_row, :) = [ ...
            N, m, GMRES_CODE, timing_repetition, 0, 0, 0, 0, ...
            gmres_actual_iteration_repeat, gmres_runtime_repeat, ...
            gmres_final_true_residual_repeat, ...
            gmres_reached_target_repeat, gmres_flag_repeat];

        % Keep one copy of the deterministic GMRES history.
        if timing_repetition == 1
            x_gmres = x_gmres_repeat;
            gmres_resvec = gmres_resvec_repeat;
        end
    end

    gmres_runtime = median(gmres_runtimes);
    gmres_actual_iteration = length(gmres_resvec) - 1;
    gmres_final_true_residual = norm(b - A * x_gmres) / norm_b;

    % Recompute each saved checkpoint explicitly outside the timing loop.
    for checkpoint_index = 1:number_of_checkpoints

        iteration = checkpoints(checkpoint_index);
        x_checkpoint = gmres_checkpoint_solution( ...
            A, b, x0, iteration, 0);
        relative_residual = norm(b - A * x_checkpoint) / norm_b;

        history_row = history_row + 1;

        history_results(history_row, :) = [ ...
            N, m, GMRES_CODE, 0, 0, 0, 0, 0, ...
            iteration, relative_residual];
    end

    assert(abs(relative_residual - gmres_final_true_residual) <= ...
        1e-10 * max(1, gmres_final_true_residual), ...
        'Unrestarted GMRES final checkpoint does not match the timed run.');

    fprintf(['GMRES: iteration=%d, residual=%.3e, ' ...
        'median time=%.3f s\n'], ...
        gmres_actual_iteration, gmres_final_true_residual, gmres_runtime);

    % The large GMRES workspace is no longer needed.
    clear x_gmres gmres_resvec x_gmres_repeat gmres_resvec_repeat;

    %% 4c. Run MATLAB built-in restarted GMRES(50)
    %
    % Five cycles of length 50 give the same maximum total of 250 Krylov
    % steps as the two other methods. The basis is discarded after every
    % cycle, so at most restart_length + 1 Arnoldi vectors are retained.

    restarted_gmres_runtimes = zeros(gmres_timing_repetitions, 1);

    for timing_repetition = 1:gmres_timing_repetitions

        restarted_gmres_timer = tic;

        [x_restarted_repeat, restarted_flag_repeat, ~, ...
            restarted_iteration_repeat, ...
            restarted_resvec_repeat] = gmres( ...
            A, b, restart_length, eps, maximum_restart_cycles, ...
            [], [], x0);

        restarted_runtime_repeat = toc(restarted_gmres_timer);
        restarted_actual_iteration_repeat = ...
            length(restarted_resvec_repeat) - 1;
        restarted_returned_iteration = ...
            gmres_restarted_iteration_count( ...
            restarted_iteration_repeat, restart_length);
        assert(restarted_actual_iteration_repeat == ...
            restarted_returned_iteration, ...
            'Restarted GMRES iteration outputs are inconsistent.');
        restarted_final_true_residual_repeat = ...
            norm(b - A * x_restarted_repeat) / norm_b;
        restarted_reached_target_repeat = ...
            restarted_final_true_residual_repeat <= target_residual;

        restarted_gmres_runtimes(timing_repetition) = ...
            restarted_runtime_repeat;

        run_row = run_row + 1;

        run_results(run_row, :) = [ ...
            N, m, RESTARTED_GMRES_CODE, timing_repetition, ...
            restart_length, 0, 0, 0, ...
            restarted_actual_iteration_repeat, ...
            restarted_runtime_repeat, ...
            restarted_final_true_residual_repeat, ...
            restarted_reached_target_repeat, restarted_flag_repeat];

        % Keep one copy of the deterministic restarted-GMRES history.
        if timing_repetition == 1
            x_restarted_gmres = x_restarted_repeat;
            restarted_gmres_resvec = restarted_resvec_repeat;
        end
    end

    restarted_gmres_runtime = median(restarted_gmres_runtimes);
    restarted_gmres_actual_iteration = ...
        length(restarted_gmres_resvec) - 1;
    restarted_gmres_final_true_residual = ...
        norm(b - A * x_restarted_gmres) / norm_b;

    for checkpoint_index = 1:number_of_checkpoints

        iteration = checkpoints(checkpoint_index);
        x_checkpoint = gmres_checkpoint_solution( ...
            A, b, x0, iteration, restart_length);
        relative_residual = norm(b - A * x_checkpoint) / norm_b;

        history_row = history_row + 1;

        history_results(history_row, :) = [ ...
            N, m, RESTARTED_GMRES_CODE, 0, restart_length, 0, 0, 0, ...
            iteration, relative_residual];
    end

    assert(abs(relative_residual - restarted_gmres_final_true_residual) <= ...
        1e-10 * max(1, restarted_gmres_final_true_residual), ...
        'Restarted GMRES final checkpoint does not match the timed run.');

    fprintf(['GMRES(%d): iteration=%d, residual=%.3e, ' ...
        'median time=%.3f s\n'], restart_length, ...
        restarted_gmres_actual_iteration, ...
        restarted_gmres_final_true_residual, ...
        restarted_gmres_runtime);

    clear x_restarted_gmres restarted_gmres_resvec ...
        x_restarted_repeat restarted_resvec_repeat;

    %% 4d. Run complete sGMRES solves with the recommended sparse map
    %
    % Every seed is timed as a fresh complete solve. The timer includes the
    % initial residual, q-truncated basis construction, sparse map, sketched
    % products, reduced solve, and cheap sketched-residual indicator.
    % True-residual diagnostics remain outside.

    for seed_index = 1:number_of_seeds

        seed = seeds(seed_index);

        % Use a reproducible but different sketch for every N and seed.
        rng(N * 1000 + zeta * 100 + seed, 'twister');

        [~, sgmres_info] = sgmres_trunc_arnoldi( ...
            A, b, x0, maximum_iteration, q, s, zeta, checkpoints);

        actual_dimension = sgmres_info.actual_dim;
        sgmres_runtime = sgmres_info.solver_time_sec;
        sgmres_final_true_residual = sgmres_info.true_relres;

        assert(~sgmres_info.rank_deficient, ...
            'The sGMRES sketched matrix is rank deficient.');
        assert(isequal(sgmres_info.checkpoint_iterations, checkpoints(:)), ...
            'The sGMRES checkpoint output is inconsistent.');

        sgmres_reached_target = ...
            sgmres_final_true_residual <= target_residual;

        run_row = run_row + 1;

        run_results(run_row, :) = [ ...
            N, m, SGMRES_CODE, seed, 0, q, s, zeta, ...
            actual_dimension, sgmres_runtime, ...
            sgmres_final_true_residual, ...
            sgmres_reached_target, NaN];

        %% Evaluate the true residual at the selected checkpoints

        for checkpoint_index = 1:number_of_checkpoints

            iteration = checkpoints(checkpoint_index);
            relative_residual = ...
                sgmres_info.checkpoint_true_relres(checkpoint_index);

            history_row = history_row + 1;

            history_results(history_row, :) = [ ...
                N, m, SGMRES_CODE, seed, 0, q, s, zeta, ...
                iteration, relative_residual];
        end

        fprintf(['sGMRES: zeta=%d, seed=%d, residual=%.3e, ' ...
            'dimension=%d, time=%.3f s\n'], ...
            zeta, seed, sgmres_final_true_residual, ...
            actual_dimension, sgmres_runtime);

        clear sgmres_info;
    end

    clear A b x0 x_true;
    fprintf('\n');
end

%% 5. Summarise the residual curves

method_codes = [GMRES_CODE, RESTARTED_GMRES_CODE, SGMRES_CODE];
number_of_methods = length(method_codes);
number_of_summary_rows = number_of_N_values * ...
    number_of_methods * number_of_checkpoints;

% Columns:
% 1 N, 2 m, 3 method code, 4 restart length, 5 q, 6 s, 7 zeta,
% 8 iteration, 9 median residual, 10 minimum residual,
% 11 maximum residual.
history_summary = zeros(number_of_summary_rows, 11);

summary_row = 0;

for N_index = 1:number_of_N_values

    N = N_values(N_index);
    m = N^2;

    for method_index = 1:number_of_methods

        method_code = method_codes(method_index);
        method_restart = 0;
        method_q = 0;
        method_s = 0;
        method_zeta = 0;

        if method_code == RESTARTED_GMRES_CODE
            method_restart = restart_length;
        elseif method_code == SGMRES_CODE
            method_q = q;
            method_s = s;
            method_zeta = zeta;
        end

        for checkpoint_index = 1:number_of_checkpoints

            iteration = checkpoints(checkpoint_index);

            matching_rows = ...
                history_results(:, 1) == N & ...
                history_results(:, 3) == method_code & ...
                history_results(:, 9) == iteration;

            values = history_results(matching_rows, 10);

            summary_row = summary_row + 1;

            history_summary(summary_row, :) = [ ...
                N, m, method_code, method_restart, method_q, ...
                method_s, method_zeta, iteration, ...
                median(values), min(values), max(values)];
        end
    end
end

%% 6. Summarise runtime, final residual, and success count

number_of_run_summary_rows = number_of_N_values * number_of_methods;

% Columns:
% 1 N, 2 m, 3 method code, 4 restart length, 5 q, 6 s, 7 zeta,
% 8 number of runs, 9 median runtime, 10 minimum runtime,
% 11 maximum runtime, 12 median final residual,
% 13 minimum final residual, 14 maximum final residual,
% 15 number reaching target.
run_summary = zeros(number_of_run_summary_rows, 15);

run_summary_row = 0;

for N_index = 1:number_of_N_values

    N = N_values(N_index);
    m = N^2;

    for method_index = 1:number_of_methods

        method_code = method_codes(method_index);

        matching_rows = ...
            run_results(:, 1) == N & ...
            run_results(:, 3) == method_code;

        method_runs = run_results(matching_rows, :);

        method_restart = 0;
        method_q = 0;
        method_s = 0;
        method_zeta = 0;

        if method_code == RESTARTED_GMRES_CODE
            method_restart = restart_length;
        elseif method_code == SGMRES_CODE
            method_q = q;
            method_s = s;
            method_zeta = zeta;
        end

        runtimes = method_runs(:, 10);
        final_residuals = method_runs(:, 11);
        successes = method_runs(:, 12);

        run_summary_row = run_summary_row + 1;

        run_summary(run_summary_row, :) = [ ...
            N, m, method_code, method_restart, method_q, ...
            method_s, method_zeta, ...
            size(method_runs, 1), ...
            median(runtimes), min(runtimes), max(runtimes), ...
            median(final_residuals), ...
            min(final_residuals), max(final_residuals), ...
            sum(successes)];
    end
end

%% 7. Save the numerical evidence

run_table = array2table(run_results);
run_table.Properties.VariableNames = { ...
    'N', 'm', 'method_code', 'run_id', 'restart_length', 'q', 's', 'zeta', ...
    'actual_iteration', 'runtime_sec', 'final_true_relres', ...
    'reached_target', 'matlab_solver_flag'};

history_table = array2table(history_results);
history_table.Properties.VariableNames = { ...
    'N', 'm', 'method_code', 'run_id', 'restart_length', 'q', 's', 'zeta', ...
    'iteration', 'true_relres'};

history_summary_table = array2table(history_summary);
history_summary_table.Properties.VariableNames = { ...
    'N', 'm', 'method_code', 'restart_length', 'q', 's', 'zeta', ...
    'iteration', ...
    'median_true_relres', 'min_true_relres', 'max_true_relres'};

run_summary_table = array2table(run_summary);
run_summary_table.Properties.VariableNames = { ...
    'N', 'm', 'method_code', 'restart_length', 'q', 's', 'zeta', ...
    'number_of_runs', ...
    'median_runtime_sec', 'min_runtime_sec', 'max_runtime_sec', ...
    'median_final_true_relres', 'min_final_true_relres', ...
    'max_final_true_relres', 'number_reaching_target'};

writetable(run_table, ...
    fullfile(processed_folder, ...
    'gmres_sgmres_poisson_scaling_runs.csv'));

writetable(history_table, ...
    fullfile(processed_folder, ...
    'gmres_sgmres_poisson_scaling_history.csv'));

writetable(history_summary_table, ...
    fullfile(processed_folder, ...
    'gmres_sgmres_poisson_scaling_history_summary.csv'));

writetable(run_summary_table, ...
    fullfile(processed_folder, ...
    'gmres_sgmres_poisson_scaling_summary.csv'));

fprintf(['Saved the processed evidence. Run ' ...
    'plot_gmres_sgmres_poisson_scaling.m to rebuild the figures.\n']);

function x = gmres_checkpoint_solution( ...
    A, b, x0, total_iterations, restart_length)
%GMRES_CHECKPOINT_SOLUTION Reconstruct one GMRES checkpoint for diagnostics.

    if total_iterations == 0
        x = x0;
        return
    end

    if restart_length == 0
        [x, ~, ~, ~, ~] = gmres( ...
            A, b, [], eps, total_iterations, [], [], x0);
        return
    end

    x = x0;
    remaining_iterations = total_iterations;

    while remaining_iterations > 0
        current_cycle_length = min(restart_length, remaining_iterations);
        [x, flag, ~, ~, ~] = gmres( ...
            A, b, current_cycle_length, eps, 1, [], [], x);
        remaining_iterations = ...
            remaining_iterations - current_cycle_length;

        if flag == 0
            break
        end
    end
end

function total_iterations = gmres_restarted_iteration_count( ...
    iteration_output, restart_length)
%GMRES_RESTARTED_ITERATION_COUNT Convert MATLAB [outer, inner] to a total.

    if all(iteration_output == 0)
        total_iterations = 0;
    else
        total_iterations = ...
            (iteration_output(1) - 1) * restart_length + ...
            iteration_output(2);
    end
end
