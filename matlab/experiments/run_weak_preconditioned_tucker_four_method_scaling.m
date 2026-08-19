function run_weak_preconditioned_tucker_four_method_scaling()
%RUN_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING Protected scaling study.
%
% Compare Algorithms 5.5 to 5.8 at N=150, 200, and 250 with exactly 100
% Arnoldi operator products. Accuracy and timing are separate evidence
% layers. Randomized accuracy uses five paired residual sketch seeds.
% Timing uses one fixed seed, one warm up per method, and four balanced
% execution orders.


%% 1. Locate the repository and prepare resumable output paths

experimentFolder = fileparts(mfilename('fullpath'));
matlabFolder = fileparts(experimentFolder);
repositoryFolder = fileparts(matlabFolder);
addpath(genpath(matlabFolder));
add_toolboxes();

config = weak_preconditioned_tucker_four_method_scaling_config();
processedFolder = fullfile(repositoryFolder, 'experiments', 'processed');
rawFolder = fullfile(repositoryFolder, 'experiments', 'raw');
figureFolder = fullfile(repositoryFolder, 'experiments', 'figures');
if ~isfolder(processedFolder), mkdir(processedFolder); end
if ~isfolder(rawFolder), mkdir(rawFolder); end
if ~isfolder(figureFolder), mkdir(figureFolder); end
paths = output_paths(config, processedFolder, rawFolder);

fprintf('\nWeak preconditioned four method scaling study\n');
fprintf('N values %s, exactly %d products, s=%d, q=%d\n\n', ...
    strjoin(string(config.candidateN.'), ', '), ...
    config.maximumIterations, config.residualSketchSize, ...
    config.orthogonalisationWindow);


%% 2. Accuracy layer with five paired randomized seeds

accuracyRuns = table();
rankHistory = table();
configurationTable = table();
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    fprintf('\nAccuracy layer at N=%d\n', N);
    problem = build_weak_preconditioned_tucker_scaling_problem(config, N);
    configurationTable = append_table(configurationTable, ...
        build_configuration_record(config, problem));

    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        methodName = config.methodNames(methodIndex);
        seeds = accuracy_seeds(config, methodId);
        for seedIndex = 1:numel(seeds)
            seed = seeds(seedIndex);
            caseFile = accuracy_case_file( ...
                config, rawFolder, N, methodId, seed);
            if isfile(caseFile)
                fprintf('  Loading %s seed %s\n', ...
                    methodName, printable_seed(seed));
                loaded = load(caseFile, 'caseData');
                caseData = loaded.caseData;
            else
                [loadedPilot, caseData] = load_matching_pilot( ...
                    config, rawFolder, N, methodId, seed);
                if loadedPilot
                    fprintf('  Reusing feasibility gate for %s seed %s\n', ...
                        methodName, printable_seed(seed));
                else
                    fprintf('  Running %s seed %s\n', ...
                        methodName, printable_seed(seed));
                    [~, info] = ...
                        execute_weak_preconditioned_tucker_scaling_method( ...
                        config, problem, methodId, seed);
                    caseData.info = info;
                end
                caseData.runTable = build_run_record( ...
                    N, "accuracy", NaN, NaN, methodId, ...
                    methodName, seed, caseData.info);
                caseData.rankHistoryTable = build_rank_history( ...
                    N, methodId, seed, caseData.info.basis_ranks);
                save(caseFile, 'config', 'caseData', '-v7.3');
            end
            accuracyRuns = append_table( ...
                accuracyRuns, caseData.runTable);
            rankHistory = append_table( ...
                rankHistory, caseData.rankHistoryTable);
            writetable(accuracyRuns, paths.accuracyRuns);
            writetable(rankHistory, paths.rankHistory);
            fprintf(['    residual %.6e, time %.3f s, ', ...
                'rank %g, storage %.3f MiB\n'], ...
                caseData.runTable.true_relative_residual, ...
                caseData.runTable.solver_time_sec, ...
                caseData.runTable.maximum_basis_rank, ...
                caseData.runTable.peak_basis_storage_mib);
        end
    end
    writetable(configurationTable, paths.configuration);
    clear problem
end


%% 3. Protected timing with warm ups and balanced method order

timingRuns = table();
timingPhases = table();
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    fprintf('\nProtected timing layer at N=%d\n', N);
    if all_timing_cases_exist(config, rawFolder, N)
        fprintf('  All measured timing cases already exist\n');
    else
        problem = build_weak_preconditioned_tucker_scaling_problem( ...
            config, N);
        fprintf('  Unmeasured warm ups\n');
        for methodIndex = 1:numel(config.methodIds)
            methodId = config.methodIds(methodIndex);
            seed = timing_seed(config, methodId);
            execute_weak_preconditioned_tucker_scaling_method( ...
                config, problem, methodId, seed);
        end
        clear problem
    end

    for blockIndex = 1:config.numberOfMeasuredTimingBlocks
        methodOrder = config.timingMethodOrders(blockIndex, :);
        for position = 1:numel(methodOrder)
            methodIndex = methodOrder(position);
            methodId = config.methodIds(methodIndex);
            methodName = config.methodNames(methodIndex);
            seed = timing_seed(config, methodId);
            caseFile = timing_case_file( ...
                config, rawFolder, N, blockIndex, methodId);
            if isfile(caseFile)
                fprintf('  Loading block %d position %d %s\n', ...
                    blockIndex, position, methodName);
                loaded = load(caseFile, 'caseData');
                caseData = loaded.caseData;
            else
                fprintf('  Running block %d position %d %s\n', ...
                    blockIndex, position, methodName);
                problem = ...
                    build_weak_preconditioned_tucker_scaling_problem( ...
                    config, N);
                [~, info] = ...
                    execute_weak_preconditioned_tucker_scaling_method( ...
                    config, problem, methodId, seed);
                referenceResidual = find_reference_residual( ...
                    accuracyRuns, N, methodId, seed);
                caseData.runTable = build_run_record( ...
                    N, "timing", blockIndex, position, methodId, ...
                    methodName, seed, info);
                caseData.runTable.endpoint_relative_drift = abs( ...
                    caseData.runTable.true_relative_residual - ...
                    referenceResidual) / max(referenceResidual, realmin);
                caseData.phaseTable = build_phase_table( ...
                    N, blockIndex, position, methodId, info);
                caseData.info = info;
                save(caseFile, 'config', 'caseData', '-v7.3');
                clear problem
            end
            timingRuns = append_table(timingRuns, caseData.runTable);
            timingPhases = append_table( ...
                timingPhases, caseData.phaseTable);
            writetable(timingRuns, paths.timingRuns);
            writetable(timingPhases, paths.timingPhases);
            fprintf('    time %.3f s, residual %.6e\n', ...
                caseData.runTable.solver_time_sec, ...
                caseData.runTable.true_relative_residual);
        end
    end
end


%% 4. Consolidate summaries, checks, and figures

accuracySummary = build_accuracy_summary(config, accuracyRuns);
timingSummary = build_timing_summary(config, timingRuns);
comparisonSummary = build_comparison_summary( ...
    config, accuracySummary, timingSummary);
checkTable = build_check_table( ...
    config, configurationTable, accuracyRuns, timingRuns, timingPhases);

writetable(configurationTable, paths.configuration);
writetable(accuracySummary, paths.accuracySummary);
writetable(timingSummary, paths.timingSummary);
writetable(comparisonSummary, paths.comparisonSummary);
writetable(checkTable, paths.checks);
save(paths.raw, 'config', 'configurationTable', 'accuracyRuns', ...
    'accuracySummary', 'rankHistory', 'timingRuns', 'timingSummary', ...
    'timingPhases', 'comparisonSummary', 'checkTable', '-v7.3');

plot_weak_preconditioned_tucker_four_method_scaling();

executionChecks = checkTable.pass( ...
    startsWith(checkTable.check_id, "execution_") | ...
    startsWith(checkTable.check_id, "design_"));
if all(executionChecks)
    markerFile = fopen(paths.completeMarker, 'w');
    if markerFile < 0
        error('Could not create the completion marker.');
    end
    fprintf(markerFile, '%s\n', config.experimentId);
    fclose(markerFile);
else
    warning('Required checks failed. No completion marker was written.');
end

fprintf('\nScaling comparison\n');
disp(comparisonSummary);
fprintf('\nValidation checks\n');
disp(checkTable);
fprintf('Evidence written with prefix %s\n\n', config.outputPrefix);

end


function paths = output_paths(config, processedFolder, rawFolder)

prefix = fullfile(processedFolder, config.outputPrefix);
paths.configuration = prefix + "_configuration.csv";
paths.accuracyRuns = prefix + "_accuracy_runs.csv";
paths.accuracySummary = prefix + "_accuracy_summary.csv";
paths.rankHistory = prefix + "_rank_history.csv";
paths.timingRuns = prefix + "_timing_runs.csv";
paths.timingPhases = prefix + "_timing_phases.csv";
paths.timingSummary = prefix + "_timing_summary.csv";
paths.comparisonSummary = prefix + "_comparison_summary.csv";
paths.checks = prefix + "_checks.csv";
paths.raw = fullfile(rawFolder, config.outputPrefix + ".mat");
paths.completeMarker = fullfile( ...
    rawFolder, config.outputPrefix + "_COMPLETE.txt");

end


function tableOut = build_configuration_record(config, problem)

info = problem.preconditionerInfo;
tableOut = table( ...
    string(config.experimentId), problem.N, config.numberOfModes, ...
    config.maximumIterations, config.gaussianExponent, ...
    config.preconditionerTuckerRank, ...
    info.minimum_preconditioned_eigenvalue, ...
    info.maximum_preconditioned_eigenvalue, info.condition_number, ...
    info.spectral_delta, info.is_positive_definite, ...
    config.residualSketchSize, config.orthogonalisationWindow, ...
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


function seeds = accuracy_seeds(config, methodId)

if methodId == "plain" || methodId == "rhosvd_sgmres"
    seeds = config.residualSketchSeeds;
else
    seeds = NaN;
end

end


function seed = timing_seed(config, methodId)

if methodId == "plain" || methodId == "rhosvd_sgmres"
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
    config, rawFolder, N, methodId, seed)

if isfinite(seed)
    seedText = sprintf('seed%06d', seed);
else
    seedText = 'deterministic';
end
path = fullfile(rawFolder, sprintf('%s_accuracy_N%03d_%s_%s.mat', ...
    config.outputPrefix, N, methodId, seedText));

end


function path = timing_case_file( ...
    config, rawFolder, N, blockIndex, methodId)

path = fullfile(rawFolder, sprintf( ...
    '%s_timing_N%03d_block%02d_%s.mat', ...
    config.outputPrefix, N, blockIndex, methodId));

end


function [loadedPilot, caseData] = load_matching_pilot( ...
    config, rawFolder, N, methodId, seed)

loadedPilot = false;
caseData = struct();
if N ~= config.pilotN
    return
end
if isfinite(seed) && seed ~= config.pilotResidualSketchSeed
    return
end
pilotPrefix = config.outputPrefix + "_pilot_N" + string(N);
pilotFile = fullfile(rawFolder, sprintf('%s_%s.mat', ...
    pilotPrefix, methodId));
if ~isfile(pilotFile)
    return
end
loaded = load(pilotFile, 'caseData');
caseData.info = loaded.caseData.info;
loadedPilot = true;

end


function present = all_timing_cases_exist(config, rawFolder, N)

present = true;
for blockIndex = 1:config.numberOfMeasuredTimingBlocks
    for methodIndex = 1:numel(config.methodIds)
        path = timing_case_file(config, rawFolder, N, blockIndex, ...
            config.methodIds(methodIndex));
        present = present && isfile(path);
    end
end

end


function value = find_reference_residual( ...
    accuracyRuns, N, methodId, seed)

rows = accuracyRuns.N == N & accuracyRuns.method_id == methodId;
if isfinite(seed)
    rows = rows & accuracyRuns.residual_seed == seed;
end
assert(sum(rows) == 1, ...
    'The timing endpoint must have one accuracy reference.');
value = accuracyRuns.true_relative_residual(rows);

end


function tableOut = build_run_record( ...
    N, layer, blockIndex, position, methodId, methodName, seed, info)

record.N = N;
record.layer = layer;
record.timing_block = blockIndex;
record.order_position = position;
record.method_id = methodId;
record.method_name = methodName;
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

if methodId == "fixed" || methodId == "relaxed"
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
    if methodId == "plain"
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
tableOut = struct2table(record);

end


function tableOut = build_rank_history(N, methodId, seed, ranks)

basisIndex = (1:size(ranks, 1)).';
tableOut = table( ...
    repmat(N, numel(basisIndex), 1), ...
    repmat(methodId, numel(basisIndex), 1), ...
    repmat(seed, numel(basisIndex), 1), basisIndex, ...
    ranks(:, 1), ranks(:, 2), ranks(:, 3), max(ranks, [], 2), ...
    'VariableNames', {'N', 'method_id', 'residual_seed', ...
    'basis_index', 'mode_1_rank', 'mode_2_rank', 'mode_3_rank', ...
    'maximum_mode_rank'});

end


function tableOut = build_phase_table( ...
    N, blockIndex, position, methodId, info)

t = info.timing;
phaseNames = [ ...
    "initial_residual"; "operator_and_preconditioner"; ...
    "residual_sketch"; "orthogonalisation_inner_products"; ...
    "tucker_sums_and_rounding"; "small_least_squares"; ...
    "basis_normalisation"; "solution_assembly"; ...
    "unclassified_solver_overhead"];
if methodId == "fixed" || methodId == "relaxed"
    phaseTimes = [ ...
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
elseif methodId == "plain"
    phaseTimes = [ ...
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
    phaseTimes = [ ...
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
tableOut = table( ...
    repmat(N, numel(phaseNames), 1), ...
    repmat(blockIndex, numel(phaseNames), 1), ...
    repmat(position, numel(phaseNames), 1), ...
    repmat(methodId, numel(phaseNames), 1), phaseNames, phaseTimes, ...
    100 * phaseTimes / max(info.solver_time_sec, realmin), ...
    'VariableNames', {'N', 'block_index', 'order_position', ...
    'method_id', 'phase', 'time_sec', 'percent_of_solver_time'});

end


function tableOut = build_accuracy_summary(config, runTable)

records = cell(numel(config.candidateN) * numel(config.methodIds), 1);
recordIndex = 0;
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        rows = runTable.N == N & runTable.method_id == methodId;
        recordIndex = recordIndex + 1;
        record.N = N;
        record.method_id = methodId;
        record.method_name = config.methodNames(methodIndex);
        record.number_of_runs = sum(rows);
        record.median_true_relative_residual = median( ...
            runTable.true_relative_residual(rows));
        record.minimum_true_relative_residual = min( ...
            runTable.true_relative_residual(rows));
        record.maximum_true_relative_residual = max( ...
            runTable.true_relative_residual(rows));
        record.median_maximum_basis_rank = median( ...
            runTable.maximum_basis_rank(rows));
        record.minimum_maximum_basis_rank = min( ...
            runTable.maximum_basis_rank(rows));
        record.maximum_maximum_basis_rank = max( ...
            runTable.maximum_basis_rank(rows));
        record.median_peak_basis_storage_mib = median( ...
            runTable.peak_basis_storage_mib(rows));
        record.minimum_peak_basis_storage_mib = min( ...
            runTable.peak_basis_storage_mib(rows));
        record.maximum_peak_basis_storage_mib = max( ...
            runTable.peak_basis_storage_mib(rows));
        record.median_sketch_matrix_condition = median( ...
            runTable.sketch_matrix_condition(rows), 'omitnan');
        records{recordIndex} = record;
    end
end
tableOut = struct2table(vertcat(records{:}));

end


function tableOut = build_timing_summary(config, runTable)

records = cell(numel(config.candidateN) * numel(config.methodIds), 1);
recordIndex = 0;
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        rows = runTable.N == N & runTable.method_id == methodId;
        times = runTable.solver_time_sec(rows);
        recordIndex = recordIndex + 1;
        record.N = N;
        record.method_id = methodId;
        record.method_name = config.methodNames(methodIndex);
        record.number_of_runs = sum(rows);
        record.median_solver_time_sec = median(times);
        record.minimum_solver_time_sec = min(times);
        record.maximum_solver_time_sec = max(times);
        record.relative_timing_range = ...
            (max(times) - min(times)) / max(median(times), realmin);
        record.maximum_endpoint_relative_drift = max( ...
            runTable.endpoint_relative_drift(rows));
        record.maximum_timing_closure_relative_error = max( ...
            runTable.timing_closure_relative_error(rows));
        records{recordIndex} = record;
    end
end
tableOut = struct2table(vertcat(records{:}));

end


function tableOut = build_comparison_summary( ...
    config, accuracySummary, timingSummary)

records = cell(height(accuracySummary), 1);
recordIndex = 0;
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    fixedRow = timingSummary.N == N & ...
        timingSummary.method_id == "fixed";
    fixedTime = timingSummary.median_solver_time_sec(fixedRow);
    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        accuracyRow = accuracySummary.N == N & ...
            accuracySummary.method_id == methodId;
        timingRow = timingSummary.N == N & ...
            timingSummary.method_id == methodId;
        recordIndex = recordIndex + 1;
        record.N = N;
        record.method_id = methodId;
        record.method_name = config.methodNames(methodIndex);
        record.median_true_relative_residual = ...
            accuracySummary.median_true_relative_residual(accuracyRow);
        record.minimum_true_relative_residual = ...
            accuracySummary.minimum_true_relative_residual(accuracyRow);
        record.maximum_true_relative_residual = ...
            accuracySummary.maximum_true_relative_residual(accuracyRow);
        record.number_of_accuracy_runs = ...
            accuracySummary.number_of_runs(accuracyRow);
        record.median_solver_time_sec = ...
            timingSummary.median_solver_time_sec(timingRow);
        record.minimum_solver_time_sec = ...
            timingSummary.minimum_solver_time_sec(timingRow);
        record.maximum_solver_time_sec = ...
            timingSummary.maximum_solver_time_sec(timingRow);
        record.speedup_over_fixed_tucker_gmres = fixedTime / ...
            record.median_solver_time_sec;
        record.median_maximum_basis_rank = ...
            accuracySummary.median_maximum_basis_rank(accuracyRow);
        record.median_peak_basis_storage_mib = ...
            accuracySummary.median_peak_basis_storage_mib(accuracyRow);
        record.median_sketch_matrix_condition = ...
            accuracySummary.median_sketch_matrix_condition(accuracyRow);
        records{recordIndex} = record;
    end
end
tableOut = struct2table(vertcat(records{:}));

end


function tableOut = build_check_table( ...
    config, configurationTable, accuracyRuns, timingRuns, timingPhases)

records = cell(10, 1);
records{1} = check_record("execution_accuracy_iteration_budget", ...
    all(accuracyRuns.completed_iterations == config.maximumIterations), ...
    "Every accuracy run completed exactly 100 products");
records{2} = check_record("execution_timing_iteration_budget", ...
    all(timingRuns.completed_iterations == config.maximumIterations), ...
    "Every timing run completed exactly 100 products");
records{3} = check_record("execution_no_breakdown", ...
    ~any(accuracyRuns.breakdown) && ~any(timingRuns.breakdown), ...
    "No method reported numerical breakdown");
records{4} = check_record("execution_no_memory_stop", ...
    ~any(accuracyRuns.memory_limit_active) && ...
    ~any(timingRuns.memory_limit_active), ...
    "No method reached the two GiB basis safeguard");
records{5} = check_record("execution_no_rank_cap", ...
    ~any(accuracyRuns.rank_cap_active) && ...
    ~any(timingRuns.rank_cap_active), ...
    "No deterministic or randomized rank cap became active");
records{6} = check_record("execution_finite_metrics", ...
    all(isfinite(accuracyRuns.true_relative_residual)) && ...
    all(isfinite(timingRuns.solver_time_sec)) && ...
    all(timingRuns.solver_time_sec > 0), ...
    "All primary residual and timing metrics are finite");
records{7} = check_record("execution_positive_preconditioner", ...
    all(configurationTable.preconditioner_positive_definite), ...
    "The rank one preconditioner is positive at every grid");

expectedAccuracyRuns = numel(config.candidateN) * ...
    (2 + 2 * numel(config.residualSketchSeeds));
records{8} = check_record("design_accuracy_run_count", ...
    height(accuracyRuns) == expectedAccuracyRuns, sprintf( ...
    'Observed %d of %d planned accuracy runs', ...
    height(accuracyRuns), expectedAccuracyRuns));
expectedTimingRuns = numel(config.candidateN) * ...
    config.numberOfMeasuredTimingBlocks * numel(config.methodIds);
records{9} = check_record("design_timing_run_count", ...
    height(timingRuns) == expectedTimingRuns, sprintf( ...
    'Observed %d of %d planned timing runs', ...
    height(timingRuns), expectedTimingRuns));

phaseSums = groupsummary(timingPhases, ...
    {'N', 'block_index', 'method_id'}, 'sum', 'time_sec');
timingKey = string(timingRuns.N) + "_" + ...
    string(timingRuns.timing_block) + "_" + timingRuns.method_id;
phaseKey = string(phaseSums.N) + "_" + ...
    string(phaseSums.block_index) + "_" + phaseSums.method_id;
[matched, locations] = ismember(timingKey, phaseKey);
closure = abs(timingRuns.solver_time_sec - ...
    phaseSums.sum_time_sec(locations)) ./ ...
    max(timingRuns.solver_time_sec, realmin);
records{10} = check_record("execution_timing_reproducibility", ...
    all(matched) && ...
    max(closure) <= config.maximumTimingClosureRelativeError && ...
    max(timingRuns.endpoint_relative_drift) <= ...
        config.maximumTimingEndpointRelativeDrift, sprintf( ...
    'Maximum closure %.3e and endpoint drift %.3e', ...
    max(closure), max(timingRuns.endpoint_relative_drift)));
tableOut = struct2table(vertcat(records{:}));

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

active = any(cellfun(@(oneInfo) logical(oneInfo.rank_cap_active), ...
    info.basis_roundsum_info));
if isfield(info, 'solution_roundsum_info') && ...
        ~isempty(info.solution_roundsum_info)
    active = active || logical(info.solution_roundsum_info.rank_cap_active);
end

end


function record = check_record(checkId, pass, detail)

record.check_id = checkId;
record.pass = logical(pass);
record.detail = string(detail);

end


function tableOut = append_table(tableIn, newRows)

if isempty(tableIn)
    tableOut = newRows;
else
    tableOut = [tableIn; newRows];
end

end
