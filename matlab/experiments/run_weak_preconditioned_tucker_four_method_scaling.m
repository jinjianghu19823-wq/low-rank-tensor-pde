function run_weak_preconditioned_tucker_four_method_scaling()
%RUN_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING Protected scaling study.

experiment_folder = fileparts(mfilename('fullpath'));
matlab_folder = fileparts(experiment_folder);
repository_folder = fileparts(matlab_folder);
addpath(genpath(matlab_folder));
add_toolboxes();

config = weak_preconditioned_tucker_four_method_scaling_config();
processed_folder = fullfile(repository_folder, 'experiments', 'processed');
raw_folder = fullfile(repository_folder, 'experiments', 'raw');
figure_folder = fullfile(repository_folder, 'experiments', 'figures');
if ~isfolder(processed_folder), mkdir(processed_folder); end
if ~isfolder(raw_folder), mkdir(raw_folder); end
if ~isfolder(figure_folder), mkdir(figure_folder); end
paths = output_paths(config, processed_folder, raw_folder);

fprintf('\nWeak preconditioned four method scaling study\n');
fprintf('N values %s, exactly %d products, s=%d, q=%d\n\n', ...
    strjoin(string(config.candidateN.'), ', '), ...
    config.maximumIterations, config.residualSketchSize, ...
    config.ktrunc);

accuracy_runs = table();
rank_history = table();
configuration_table = table();
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    fprintf('\nAccuracy layer at N=%d\n', N);
    problem = build_weak_preconditioned_tucker_scaling_problem(config, N);
    configuration_table = append_table(configuration_table, ...
        build_configuration_record(config, problem));

    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        method_name = config.methodNames(method_index);
        seeds = accuracy_seeds(config, method_id);
        for seed_index = 1:numel(seeds)
            seed = seeds(seed_index);
            case_file = accuracy_case_file( ...
                config, raw_folder, N, method_id, seed);
            if isfile(case_file)
                fprintf('  Loading %s seed %s\n', ...
                    method_name, printable_seed(seed));
                loaded = load(case_file, 'caseData');
                case_data = loaded.caseData;
            else
                [loaded_pilot, case_data] = load_matching_pilot( ...
                    config, raw_folder, N, method_id, seed);
                if loaded_pilot
                    fprintf('  Reusing feasibility gate for %s seed %s\n', ...
                        method_name, printable_seed(seed));
                else
                    fprintf('  Running %s seed %s\n', ...
                        method_name, printable_seed(seed));
                    [~, info] = ...
                        execute_weak_preconditioned_tucker_scaling_method( ...
                        config, problem, method_id, seed);
                    case_data.info = info;
                end
                case_data.runTable = build_run_record( ...
                    N, "accuracy", NaN, NaN, method_id, ...
                    method_name, seed, case_data.info);
                case_data.rankHistoryTable = build_rank_history( ...
                    N, method_id, seed, case_data.info.basis_ranks);
                save_case_data(case_file, config, case_data);
            end
            accuracy_runs = append_table( ...
                accuracy_runs, case_data.runTable);
            rank_history = append_table( ...
                rank_history, case_data.rankHistoryTable);
            writetable(accuracy_runs, paths.accuracyRuns);
            writetable(rank_history, paths.rankHistory);
            fprintf(['    residual %.6e, time %.3f s, ', ...
                'rank %g, storage %.3f MiB\n'], ...
                case_data.runTable.true_relative_residual, ...
                case_data.runTable.solver_time_sec, ...
                case_data.runTable.maximum_basis_rank, ...
                case_data.runTable.peak_basis_storage_mib);
        end
    end
    writetable(configuration_table, paths.configuration);
    clear problem
end

timing_runs = table();
timing_phases = table();
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    fprintf('\nProtected timing layer at N=%d\n', N);
    if all_timing_cases_exist(config, raw_folder, N)
        fprintf('  All measured timing cases already exist\n');
    else
        problem = build_weak_preconditioned_tucker_scaling_problem( ...
            config, N);
        fprintf('  Unmeasured warm ups\n');
        for method_index = 1:numel(config.methodIds)
            method_id = config.methodIds(method_index);
            seed = timing_seed(config, method_id);
            execute_weak_preconditioned_tucker_scaling_method( ...
                config, problem, method_id, seed);
        end
        clear problem
    end

    for block_index = 1:config.numberOfMeasuredTimingBlocks
        method_order = config.timingMethodOrders(block_index, :);
        for position = 1:numel(method_order)
            method_index = method_order(position);
            method_id = config.methodIds(method_index);
            method_name = config.methodNames(method_index);
            seed = timing_seed(config, method_id);
            case_file = timing_case_file( ...
                config, raw_folder, N, block_index, method_id);
            if isfile(case_file)
                fprintf('  Loading block %d position %d %s\n', ...
                    block_index, position, method_name);
                loaded = load(case_file, 'caseData');
                case_data = loaded.caseData;
            else
                fprintf('  Running block %d position %d %s\n', ...
                    block_index, position, method_name);
                problem = ...
                    build_weak_preconditioned_tucker_scaling_problem( ...
                    config, N);
                [~, info] = ...
                    execute_weak_preconditioned_tucker_scaling_method( ...
                    config, problem, method_id, seed);
                reference_residual = find_reference_residual( ...
                    accuracy_runs, N, method_id, seed);
                case_data.runTable = build_run_record( ...
                    N, "timing", block_index, position, method_id, ...
                    method_name, seed, info);
                case_data.runTable.endpoint_relative_drift = abs( ...
                    case_data.runTable.true_relative_residual - ...
                    reference_residual) / max(reference_residual, realmin);
                case_data.phaseTable = build_phase_table( ...
                    N, block_index, position, method_id, info);
                case_data.info = info;
                save_case_data(case_file, config, case_data);
                clear problem
            end
            timing_runs = append_table(timing_runs, case_data.runTable);
            timing_phases = append_table( ...
                timing_phases, case_data.phaseTable);
            writetable(timing_runs, paths.timingRuns);
            writetable(timing_phases, paths.timingPhases);
            fprintf('    time %.3f s, residual %.6e\n', ...
                case_data.runTable.solver_time_sec, ...
                case_data.runTable.true_relative_residual);
        end
    end
end

accuracy_summary = build_accuracy_summary(config, accuracy_runs);
timing_summary = build_timing_summary(config, timing_runs);
comparison_summary = build_comparison_summary( ...
    config, accuracy_summary, timing_summary);
check_table = build_check_table( ...
    config, configuration_table, accuracy_runs, timing_runs, timing_phases);

writetable(configuration_table, paths.configuration);
writetable(accuracy_summary, paths.accuracySummary);
writetable(timing_summary, paths.timingSummary);
writetable(comparison_summary, paths.comparisonSummary);
writetable(check_table, paths.checks);
payload.config = config;
payload.configurationTable = configuration_table;
payload.accuracyRuns = accuracy_runs;
payload.accuracySummary = accuracy_summary;
payload.rankHistory = rank_history;
payload.timingRuns = timing_runs;
payload.timingSummary = timing_summary;
payload.timingPhases = timing_phases;
payload.comparisonSummary = comparison_summary;
payload.checkTable = check_table;
save(paths.raw, '-struct', 'payload', '-v7.3');

plot_weak_preconditioned_tucker_four_method_scaling();

execution_checks = check_table.pass( ...
    startsWith(check_table.check_id, "execution_") | ...
    startsWith(check_table.check_id, "design_"));
if all(execution_checks)
    marker_file = fopen(paths.completeMarker, 'w');
    if marker_file < 0
        error('Could not create the completion marker.');
    end
    fprintf(marker_file, '%s\n', config.experimentId);
    fclose(marker_file);
else
    warning('Required checks failed. No completion marker was written.');
end

fprintf('\nScaling comparison\n');
disp(comparison_summary);
fprintf('\nValidation checks\n');
disp(check_table);
fprintf('Evidence written with prefix %s\n\n', config.outputPrefix);

end

function save_case_data(path, config, case_data)

payload.config = config;
payload.caseData = case_data;
save(path, '-struct', 'payload', '-v7.3');

end

function paths = output_paths(config, processed_folder, raw_folder)

prefix = fullfile(processed_folder, config.outputPrefix);
paths.configuration = prefix + "_configuration.csv";
paths.accuracyRuns = prefix + "_accuracy_runs.csv";
paths.accuracySummary = prefix + "_accuracy_summary.csv";
paths.rankHistory = prefix + "_rank_history.csv";
paths.timingRuns = prefix + "_timing_runs.csv";
paths.timingPhases = prefix + "_timing_phases.csv";
paths.timingSummary = prefix + "_timing_summary.csv";
paths.comparisonSummary = prefix + "_comparison_summary.csv";
paths.checks = prefix + "_checks.csv";
paths.raw = fullfile(raw_folder, config.outputPrefix + ".mat");
paths.completeMarker = fullfile( ...
    raw_folder, config.outputPrefix + "_COMPLETE.txt");

end

function table_out = build_configuration_record(config, problem)

info = problem.preconditionerInfo;
table_out = table( ...
    string(config.experimentId), problem.N, config.d, ...
    config.maximumIterations, config.gaussianExponent, ...
    config.preconditionerTuckerRank, ...
    info.minimum_preconditioned_eigenvalue, ...
    info.maximum_preconditioned_eigenvalue, info.condition_number, ...
    info.spectral_delta, info.is_positive_definite, ...
    config.residualSketchSize, config.ktrunc, ...
    string(strjoin(string(config.residualSketchSeeds.'), ';')), ...
    'VariableNames', {'experiment_id', 'N', 'number_of_modes', ...
    'maximum_iterations', 'gaussian_exponent', ...
    'preconditioner_rank', ...
    'minimum_preconditioned_eigenvalue', ...
    'maximum_preconditioned_eigenvalue', ...
    'predicted_condition_number', 'spectral_delta', ...
    'preconditioner_positive_definite', 'residual_sketch_size', ...
    'orthogonalisation_window', 'residual_sketch_seeds'});

end

function seeds = accuracy_seeds(config, method_id)

if method_id == "plain" || method_id == "rhosvd_sgmres"
    seeds = config.residualSketchSeeds;
else
    seeds = NaN;
end

end

function seed = timing_seed(config, method_id)

if method_id == "plain" || method_id == "rhosvd_sgmres"
    seed = config.timingResidualSketchSeed;
else
    seed = NaN;
end

end

function text = printable_seed(seed)

if isfinite(seed)
    text = string(seed);
else
    text = "deterministic";
end

end

function path = accuracy_case_file( ...
    config, raw_folder, N, method_id, seed)

if isfinite(seed)
    seed_text = sprintf('seed%06d', seed);
else
    seed_text = 'deterministic';
end
path = fullfile(raw_folder, sprintf('%s_accuracy_N%03d_%s_%s.mat', ...
    config.outputPrefix, N, method_id, seed_text));

end

function path = timing_case_file( ...
    config, raw_folder, N, block_index, method_id)

path = fullfile(raw_folder, sprintf( ...
    '%s_timing_N%03d_block%02d_%s.mat', ...
    config.outputPrefix, N, block_index, method_id));

end

function [loaded_pilot, case_data] = load_matching_pilot( ...
    config, raw_folder, N, method_id, seed)

loaded_pilot = false;
case_data = struct();
if N ~= config.pilotN
    return
end
if isfinite(seed) && seed ~= config.pilotResidualSketchSeed
    return
end
pilot_prefix = config.outputPrefix + "_pilot_N" + string(N);
pilot_file = fullfile(raw_folder, sprintf('%s_%s.mat', ...
    pilot_prefix, method_id));
if ~isfile(pilot_file)
    return
end
loaded = load(pilot_file, 'caseData');
case_data.info = loaded.caseData.info;
loaded_pilot = true;

end

function present = all_timing_cases_exist(config, raw_folder, N)

present = true;
for block_index = 1:config.numberOfMeasuredTimingBlocks
    for method_index = 1:numel(config.methodIds)
        path = timing_case_file(config, raw_folder, N, block_index, ...
            config.methodIds(method_index));
        present = present && isfile(path);
    end
end

end

function value = find_reference_residual( ...
    accuracy_runs, N, method_id, seed)

rows = accuracy_runs.N == N & accuracy_runs.method_id == method_id;
if isfinite(seed)
    rows = rows & accuracy_runs.residual_seed == seed;
end
assert(sum(rows) == 1, ...
    'The timing endpoint must have one accuracy reference.');
value = accuracy_runs.true_relative_residual(rows);

end

function table_out = build_run_record( ...
    N, layer, block_index, position, method_id, method_name, seed, info)

record.N = N;
record.layer = layer;
record.timing_block = block_index;
record.order_position = position;
record.method_id = method_id;
record.method_name = method_name;
record.residual_seed = seed;
record.randomized = isfinite(seed);
record.completed_iterations = info.iterations;
record.stop_reason = string(info.stop_reason);
record.true_relative_residual = info.true_relative_residual(end);
record.independent_preconditioned_relative_residual = NaN;
record.computed_internal_relative_residual = NaN;
record.maximum_basis_rank = max(info.basis_ranks, [], 'all');
record.maximum_solution_rank = max(info.solution_ranks, [], 'all');
record.peak_basis_storage_mib = NaN;
record.sketch_matrix_rank = NaN;
record.sketch_matrix_condition = NaN;
record.basis_orthogonality_error_fro = NaN;
record.rank_cap_active = false;
record.breakdown = false;
record.memory_limit_active = false;
record.solver_time_sec = info.solver_time_sec;
record.diagnostic_time_sec = info.diagnostic_time_sec;
record.wall_time_sec = info.wall_time_sec;
record.timing_closure_relative_error = timing_closure_error(info);

if method_id == "fixed" || method_id == "relaxed"
    record.independent_preconditioned_relative_residual = ...
        info.true_preconditioned_relative_residual(end);
    record.computed_internal_relative_residual = ...
        info.computed_preconditioned_relative_residual(end);
    record.peak_basis_storage_mib = ...
        info.peak_basis_storage_entries * 8 / 2^20;
    record.rank_cap_active = logical(info.rank_cap_active);
    record.memory_limit_active = logical(info.stopped_for_basis_memory);
else
    record.independent_preconditioned_relative_residual = ...
        info.independent_preconditioned_relative_residual;
    record.computed_internal_relative_residual = ...
        info.computed_sketch_relative_residual(end);
    if method_id == "plain"
        record.peak_basis_storage_mib = ...
            info.peak_basis_storage_entries * 8 / 2^20;
        record.rank_cap_active = logical(info.rank_cap_active);
    else
        record.peak_basis_storage_mib = ...
            max(info.basis_storage_history_entries) * 8 / 2^20;
        record.rank_cap_active = any_roundsum_cap(info);
    end
    record.sketch_matrix_rank = info.sketch_matrix_rank(end);
    record.sketch_matrix_condition = ...
        info.sketch_matrix_condition(end);
    record.basis_orthogonality_error_fro = ...
        info.basis_orthogonality_error_fro;
    record.breakdown = logical(info.breakdown);
    record.memory_limit_active = logical(info.stopped_for_basis_memory);
end

record.internal_to_independent_residual_ratio = ...
    record.computed_internal_relative_residual / max( ...
    record.independent_preconditioned_relative_residual, realmin);
table_out = struct2table(record);

end

function table_out = build_rank_history(N, method_id, seed, ranks)

basis_index = (1:size(ranks, 1)).';
table_out = table( ...
    repmat(N, numel(basis_index), 1), ...
    repmat(method_id, numel(basis_index), 1), ...
    repmat(seed, numel(basis_index), 1), basis_index, ...
    ranks(:, 1), ranks(:, 2), ranks(:, 3), max(ranks, [], 2), ...
    'VariableNames', {'N', 'method_id', 'residual_seed', ...
    'basis_index', 'mode_1_rank', 'mode_2_rank', 'mode_3_rank', ...
    'maximum_mode_rank'});

end

function table_out = build_phase_table( ...
    N, block_index, position, method_id, info)

t = info.timing;
phase_names = [ ...
    "initial_residual"; "operator_and_preconditioner"; ...
    "residual_sketch"; "orthogonalisation_inner_products"; ...
    "tucker_sums_and_rounding"; "small_least_squares"; ...
    "basis_normalisation"; "solution_assembly"; ...
    "unclassified_solver_overhead"];
if method_id == "fixed" || method_id == "relaxed"
    phase_times = [ ...
        t.initial_residual_formation_time_sec; ...
        t.operator_application_time_sec + ...
            t.preconditioner_application_time_sec; ...
        0; t.orthogonalisation_inner_product_time_sec; ...
        t.outer_round_time_sec + ...
            t.orthogonalisation_subtraction_round_time_sec; ...
        t.small_least_squares_time_sec; ...
        t.basis_normalisation_time_sec; ...
        t.solution_assembly_time_sec; ...
        t.unclassified_solver_time_sec];
elseif method_id == "plain"
    phase_times = [ ...
        t.initial_residual_time_sec; ...
        t.operator_term_time_sec + t.preconditioner_term_time_sec; ...
        t.residual_sketch_time_sec; ...
        t.orthogonalisation_inner_product_time_sec; ...
        t.operator_exact_sum_time_sec + ...
            t.orthogonalisation_exact_sum_time_sec + ...
            t.basis_round_time_sec; ...
        t.small_least_squares_time_sec; ...
        t.basis_normalisation_time_sec; ...
        t.solution_assembly_time_sec; ...
        t.unclassified_solver_time_sec];
else
    phase_times = [ ...
        t.initial_residual_time_sec; ...
        t.operator_term_time_sec + t.preconditioner_term_time_sec; ...
        t.residual_sketch_time_sec; ...
        t.orthogonalisation_inner_product_time_sec; ...
        t.roundsum_time_sec; ...
        t.small_least_squares_time_sec; ...
        t.basis_normalisation_time_sec; ...
        t.solution_assembly_time_sec; ...
        t.unclassified_solver_time_sec];
end
table_out = table( ...
    repmat(N, numel(phase_names), 1), ...
    repmat(block_index, numel(phase_names), 1), ...
    repmat(position, numel(phase_names), 1), ...
    repmat(method_id, numel(phase_names), 1), phase_names, phase_times, ...
    100 * phase_times / max(info.solver_time_sec, realmin), ...
    'VariableNames', {'N', 'block_index', 'order_position', ...
    'method_id', 'phase', 'time_sec', 'percent_of_solver_time'});

end

function table_out = build_accuracy_summary(config, run_table)

records = cell(numel(config.candidateN) * numel(config.methodIds), 1);
record_index = 0;
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        rows = run_table.N == N & run_table.method_id == method_id;
        record_index = record_index + 1;
        record.N = N;
        record.method_id = method_id;
        record.method_name = config.methodNames(method_index);
        record.number_of_runs = sum(rows);
        record.median_true_relative_residual = median( ...
            run_table.true_relative_residual(rows));
        record.minimum_true_relative_residual = min( ...
            run_table.true_relative_residual(rows));
        record.maximum_true_relative_residual = max( ...
            run_table.true_relative_residual(rows));
        record.median_maximum_basis_rank = median( ...
            run_table.maximum_basis_rank(rows));
        record.minimum_maximum_basis_rank = min( ...
            run_table.maximum_basis_rank(rows));
        record.maximum_maximum_basis_rank = max( ...
            run_table.maximum_basis_rank(rows));
        record.median_peak_basis_storage_mib = median( ...
            run_table.peak_basis_storage_mib(rows));
        record.minimum_peak_basis_storage_mib = min( ...
            run_table.peak_basis_storage_mib(rows));
        record.maximum_peak_basis_storage_mib = max( ...
            run_table.peak_basis_storage_mib(rows));
        record.median_sketch_matrix_condition = median( ...
            run_table.sketch_matrix_condition(rows), 'omitnan');
        records{record_index} = record;
    end
end
table_out = struct2table(vertcat(records{:}));

end

function table_out = build_timing_summary(config, run_table)

records = cell(numel(config.candidateN) * numel(config.methodIds), 1);
record_index = 0;
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        rows = run_table.N == N & run_table.method_id == method_id;
        times = run_table.solver_time_sec(rows);
        record_index = record_index + 1;
        record.N = N;
        record.method_id = method_id;
        record.method_name = config.methodNames(method_index);
        record.number_of_runs = sum(rows);
        record.median_solver_time_sec = median(times);
        record.minimum_solver_time_sec = min(times);
        record.maximum_solver_time_sec = max(times);
        record.relative_timing_range = ...
            (max(times) - min(times)) / max(median(times), realmin);
        record.maximum_endpoint_relative_drift = max( ...
            run_table.endpoint_relative_drift(rows));
        record.maximum_timing_closure_relative_error = max( ...
            run_table.timing_closure_relative_error(rows));
        records{record_index} = record;
    end
end
table_out = struct2table(vertcat(records{:}));

end

function table_out = build_comparison_summary( ...
    config, accuracy_summary, timing_summary)

records = cell(height(accuracy_summary), 1);
record_index = 0;
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    fixed_row = timing_summary.N == N & ...
        timing_summary.method_id == "fixed";
    fixed_time = timing_summary.median_solver_time_sec(fixed_row);
    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        accuracy_row = accuracy_summary.N == N & ...
            accuracy_summary.method_id == method_id;
        timing_row = timing_summary.N == N & ...
            timing_summary.method_id == method_id;
        record_index = record_index + 1;
        record.N = N;
        record.method_id = method_id;
        record.method_name = config.methodNames(method_index);
        record.median_true_relative_residual = ...
            accuracy_summary.median_true_relative_residual(accuracy_row);
        record.minimum_true_relative_residual = ...
            accuracy_summary.minimum_true_relative_residual(accuracy_row);
        record.maximum_true_relative_residual = ...
            accuracy_summary.maximum_true_relative_residual(accuracy_row);
        record.number_of_accuracy_runs = ...
            accuracy_summary.number_of_runs(accuracy_row);
        record.median_solver_time_sec = ...
            timing_summary.median_solver_time_sec(timing_row);
        record.minimum_solver_time_sec = ...
            timing_summary.minimum_solver_time_sec(timing_row);
        record.maximum_solver_time_sec = ...
            timing_summary.maximum_solver_time_sec(timing_row);
        record.speedup_over_fixed_tucker_gmres = fixed_time / ...
            record.median_solver_time_sec;
        record.median_maximum_basis_rank = ...
            accuracy_summary.median_maximum_basis_rank(accuracy_row);
        record.median_peak_basis_storage_mib = ...
            accuracy_summary.median_peak_basis_storage_mib(accuracy_row);
        record.median_sketch_matrix_condition = ...
            accuracy_summary.median_sketch_matrix_condition(accuracy_row);
        records{record_index} = record;
    end
end
table_out = struct2table(vertcat(records{:}));

end

function table_out = build_check_table( ...
    config, configuration_table, accuracy_runs, timing_runs, timing_phases)

records = cell(10, 1);
records{1} = check_record("execution_accuracy_iteration_budget", ...
    all(accuracy_runs.completed_iterations == config.maximumIterations), ...
    "Every accuracy run completed exactly 100 products");
records{2} = check_record("execution_timing_iteration_budget", ...
    all(timing_runs.completed_iterations == config.maximumIterations), ...
    "Every timing run completed exactly 100 products");
records{3} = check_record("execution_no_breakdown", ...
    ~any(accuracy_runs.breakdown) && ~any(timing_runs.breakdown), ...
    "No method reported numerical breakdown");
records{4} = check_record("execution_no_memory_stop", ...
    ~any(accuracy_runs.memory_limit_active) && ...
    ~any(timing_runs.memory_limit_active), ...
    "No method reached the two GiB basis safeguard");
records{5} = check_record("execution_no_rank_cap", ...
    ~any(accuracy_runs.rank_cap_active) && ...
    ~any(timing_runs.rank_cap_active), ...
    "No deterministic or randomized rank cap became active");
records{6} = check_record("execution_finite_metrics", ...
    all(isfinite(accuracy_runs.true_relative_residual)) && ...
    all(isfinite(timing_runs.solver_time_sec)) && ...
    all(timing_runs.solver_time_sec > 0), ...
    "All primary residual and timing metrics are finite");
records{7} = check_record("execution_positive_preconditioner", ...
    all(configuration_table.preconditioner_positive_definite), ...
    "The rank one preconditioner is positive at every grid");

expected_accuracy_runs = numel(config.candidateN) * ...
    (2 + 2 * numel(config.residualSketchSeeds));
records{8} = check_record("design_accuracy_run_count", ...
    height(accuracy_runs) == expected_accuracy_runs, sprintf( ...
    'Observed %d of %d planned accuracy runs', ...
    height(accuracy_runs), expected_accuracy_runs));
expected_timing_runs = numel(config.candidateN) * ...
    config.numberOfMeasuredTimingBlocks * numel(config.methodIds);
records{9} = check_record("design_timing_run_count", ...
    height(timing_runs) == expected_timing_runs, sprintf( ...
    'Observed %d of %d planned timing runs', ...
    height(timing_runs), expected_timing_runs));

phase_sums = groupsummary(timing_phases, ...
    {'N', 'block_index', 'method_id'}, 'sum', 'time_sec');
timing_key = string(timing_runs.N) + "_" + ...
    string(timing_runs.timing_block) + "_" + timing_runs.method_id;
phase_key = string(phase_sums.N) + "_" + ...
    string(phase_sums.block_index) + "_" + phase_sums.method_id;
[matched, locations] = ismember(timing_key, phase_key);
closure = abs(timing_runs.solver_time_sec - ...
    phase_sums.sum_time_sec(locations)) ./ ...
    max(timing_runs.solver_time_sec, realmin);
records{10} = check_record("execution_timing_reproducibility", ...
    all(matched) && ...
    max(closure) <= config.maximumTimingClosureRelativeError && ...
    max(timing_runs.endpoint_relative_drift) <= ...
        config.maximumTimingEndpointRelativeDrift, sprintf( ...
    'Maximum closure %.3e and endpoint drift %.3e', ...
    max(closure), max(timing_runs.endpoint_relative_drift)));
table_out = struct2table(vertcat(records{:}));

end

function value = timing_closure_error(info)

if isfield(info, 'timing') && ...
        isfield(info.timing, 'closure_relative_error')
    value = info.timing.closure_relative_error;
else
    value = abs(info.wall_time_sec - info.solver_time_sec - ...
        info.diagnostic_time_sec) / max(info.wall_time_sec, realmin);
end

end

function active = any_roundsum_cap(info)

active = any(cellfun(@(one_info) logical(one_info.rank_cap_active), ...
    info.basis_roundsum_info));
if isfield(info, 'solution_roundsum_info') && ...
        ~isempty(info.solution_roundsum_info)
    active = active || logical(info.solution_roundsum_info.rank_cap_active);
end

end

function record = check_record(check_id, pass, detail)

record.check_id = check_id;
record.pass = logical(pass);
record.detail = string(detail);

end

function table_out = append_table(table_in, new_rows)

if isempty(table_in)
    table_out = new_rows;
else
    table_out = [table_in; new_rows];
end

end
