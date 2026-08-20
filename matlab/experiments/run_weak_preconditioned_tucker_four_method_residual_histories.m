function run_weak_preconditioned_tucker_four_method_residual_histories()
%RUN_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_RESIDUAL_HISTORIES Diagnostics.

experiment_folder = fileparts(mfilename('fullpath'));
matlab_folder = fileparts(experiment_folder);
repository_folder = fileparts(matlab_folder);
addpath(genpath(matlab_folder));
add_toolboxes();

config = weak_preconditioned_tucker_four_method_scaling_config();
processed_folder = fullfile(repository_folder, 'experiments', 'processed');
raw_folder = fullfile(repository_folder, 'experiments', 'raw');
if ~isfolder(processed_folder), mkdir(processed_folder); end
if ~isfolder(raw_folder), mkdir(raw_folder); end

prefix = fullfile(processed_folder, config.outputPrefix);
endpoint_path = prefix + "_accuracy_runs.csv";
history_path = prefix + "_true_residual_history.csv";
check_path = prefix + "_true_residual_history_checks.csv";
raw_path = fullfile(raw_folder, ...
    config.outputPrefix + "_true_residual_history.mat");
complete_marker = fullfile(raw_folder, ...
    config.outputPrefix + "_TRUE_RESIDUAL_HISTORY_COMPLETE.txt");

assert(isfile(endpoint_path), ...
    'Run the protected scaling study before its diagnostic histories.');
endpoint_runs = readtable(endpoint_path, 'TextType', 'string');
check_it = config.diagnosticCheckpoints(2:end);

fprintf('\nUntimed original true-residual histories\n');
fprintf('N values %s and checkpoints %s\n\n', ...
    strjoin(string(config.candidateN.'), ', '), ...
    strjoin(string(config.diagnosticCheckpoints.'), ', '));

history_table = table();
check_table = table();
for grid_index = 1:numel(config.candidateN)
    N = config.candidateN(grid_index);
    fprintf('Diagnostic histories at N=%d\n', N);
    problem = build_weak_preconditioned_tucker_scaling_problem(config, N);

    for method_index = 1:numel(config.methodIds)
        method_id = config.methodIds(method_index);
        method_name = config.methodNames(method_index);
        seeds = diagnostic_seeds(config, method_id);

        for seed_index = 1:numel(seeds)
            seed = seeds(seed_index);
            case_file = diagnostic_case_file( ...
                config, raw_folder, N, method_id, seed);

            if isfile(case_file)
                fprintf('  Loading %s seed %s\n', ...
                    method_name, printable_seed(seed));
                loaded = load(case_file, 'caseData');
                case_data = loaded.caseData;
            else
                fprintf('  Running %s seed %s\n', ...
                    method_name, printable_seed(seed));
                [~, info] = ...
                    execute_weak_preconditioned_tucker_scaling_method( ...
                    config, problem, method_id, seed, ...
                    check_it);
                reference_residual = endpoint_residual( ...
                    endpoint_runs, N, method_id, seed);
                case_data = build_case_data( ...
                    config, N, method_id, method_name, seed, info, ...
                    reference_residual);
                save_case_data(case_file, config, case_data);
            end

            history_table = append_table( ...
                history_table, case_data.historyTable);
            check_table = append_table(check_table, case_data.checkTable);
            writetable(history_table, history_path);
            writetable(check_table, check_path);
            fprintf('    final residual %.6e, endpoint drift %.3e\n', ...
                case_data.checkTable.final_true_relative_residual, ...
                case_data.checkTable.endpoint_relative_drift);
        end
    end
    clear problem
end

assert(all(check_table.pass), ...
    'At least one true-residual history check failed.');
payload.config = config;
payload.historyTable = history_table;
payload.checkTable = check_table;
save(raw_path, '-struct', 'payload', '-v7.3');

marker_file = fopen(complete_marker, 'w');
assert(marker_file >= 0, 'Could not create the completion marker.');
fprintf(marker_file, '%s\n', config.experimentId);
fclose(marker_file);

plot_weak_preconditioned_tucker_four_method_scaling();
fprintf('\nSaved %s\n', history_path);
fprintf('All diagnostic history checks passed.\n\n');

end

function save_case_data(path, config, case_data)

payload.config = config;
payload.caseData = case_data;
save(path, '-struct', 'payload', '-v7.3');

end

function seeds = diagnostic_seeds(config, method_id)

if method_id == "plain" || method_id == "rhosvd_sgmres"
    seeds = config.residualSketchSeeds;
else
    seeds = NaN;
end

end

function text = printable_seed(seed)

if isfinite(seed)
    text = string(seed);
else
    text = "deterministic";
end

end

function path = diagnostic_case_file( ...
    config, raw_folder, N, method_id, seed)

if isfinite(seed)
    seed_text = sprintf('seed%06d', seed);
else
    seed_text = 'deterministic';
end
path = fullfile(raw_folder, sprintf( ...
    '%s_true_residual_history_N%03d_%s_%s.mat', ...
    config.outputPrefix, N, method_id, seed_text));

end

function residual = endpoint_residual(endpoint_runs, N, method_id, seed)

rows = endpoint_runs.N == N & endpoint_runs.method_id == method_id;
if isfinite(seed)
    rows = rows & endpoint_runs.residual_seed == seed;
else
    rows = rows & isnan(endpoint_runs.residual_seed);
end
assert(nnz(rows) == 1, ...
    'Expected one endpoint reference for N=%d, method %s, seed %s.', ...
    N, method_id, printable_seed(seed));
residual = endpoint_runs.true_relative_residual(rows);

end

function case_data = build_case_data( ...
    config, N, method_id, method_name, seed, info, reference_residual)

iteration = info.true_residual_iteration(:);
residual = info.true_relative_residual(:);
expected_iteration = config.diagnosticCheckpoints(:);
endpoint_relative_drift = abs(residual(end) - reference_residual) / ...
    max(reference_residual, realmin);

iteration_complete = isequal(iteration, expected_iteration);
finite_residuals = numel(residual) == numel(expected_iteration) && ...
    all(isfinite(residual)) && all(residual > 0);
completed_work = info.iterations == config.maximumIterations && ...
    string(info.stop_reason) == "iteration_limit";
safe_execution = ~logical_field(info, 'breakdown') && ...
    ~logical_field(info, 'stopped_for_basis_memory') && ...
    ~roundsum_rank_cap_active(info);
endpoint_consistent = endpoint_relative_drift <= ...
    config.maximumTimingEndpointRelativeDrift;
pass = iteration_complete && finite_residuals && completed_work && ...
    safe_execution && endpoint_consistent;

number_of_rows = numel(iteration);
case_data.historyTable = table( ...
    repmat(N, number_of_rows, 1), ...
    repmat(string(method_id), number_of_rows, 1), ...
    repmat(string(method_name), number_of_rows, 1), ...
    repmat(seed, number_of_rows, 1), iteration, residual, ...
    'VariableNames', {'N', 'method_id', 'method_name', ...
    'residual_seed', 'arnoldi_iteration', ...
    'true_relative_residual'});

case_data.checkTable = table( ...
    N, string(method_id), seed, info.iterations, ...
    string(info.stop_reason), numel(iteration), iteration_complete, ...
    finite_residuals, safe_execution, residual(end), reference_residual, ...
    endpoint_relative_drift, pass, ...
    'VariableNames', {'N', 'method_id', 'residual_seed', ...
    'completed_iterations', 'stop_reason', 'checkpoint_count', ...
    'checkpoint_iterations_complete', 'finite_positive_residuals', ...
    'safe_execution', 'final_true_relative_residual', ...
    'reference_true_relative_residual', 'endpoint_relative_drift', ...
    'pass'});

case_data.info = info;

end

function value = logical_field(info, field_name)

value = false;
if isfield(info, field_name)
    value = logical(info.(field_name));
end

end

function active = roundsum_rank_cap_active(info)

active = logical_field(info, 'rank_cap_active');
if isfield(info, 'basis_roundsum_info')
    active = active || any(cellfun(@(one_info) ...
        logical_field(one_info, 'rank_cap_active'), ...
        info.basis_roundsum_info));
end
if isfield(info, 'solution_roundsum_info')
    active = active || logical_field( ...
        info.solution_roundsum_info, 'rank_cap_active');
end

end

function combined = append_table(combined, next_rows)

if isempty(combined)
    combined = next_rows;
else
    combined = [combined; next_rows];
end

end
