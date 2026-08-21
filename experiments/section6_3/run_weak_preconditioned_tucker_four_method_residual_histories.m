function run_weak_preconditioned_tucker_four_method_residual_histories()
%RUN_WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_RESIDUAL_HISTORIES Diagnostics.
%
% Evaluate the original true relative residual at the frozen checkpoints
% j=0,20,...,100. These runs are diagnostic only. Their solution assembly
% and residual evaluation times are excluded from the protected timing
% evidence used in the scaling comparison.


%% 1. Locate the repository and existing endpoint evidence

experimentFolder = fileparts(mfilename('fullpath'));
matlabFolder = fileparts(experimentFolder);
repositoryFolder = fileparts(matlabFolder);
addpath(repositoryFolder);
add_toolboxes();

config = weak_preconditioned_tucker_four_method_scaling_config();
processedFolder = fullfile(repositoryFolder, 'experiments', 'processed');
rawFolder = fullfile(repositoryFolder, 'experiments', 'raw');
if ~isfolder(processedFolder), mkdir(processedFolder); end
if ~isfolder(rawFolder), mkdir(rawFolder); end

prefix = fullfile(processedFolder, config.outputPrefix);
endpointPath = prefix + "_accuracy_runs.csv";
historyPath = prefix + "_true_residual_history.csv";
checkPath = prefix + "_true_residual_history_checks.csv";
rawPath = fullfile(rawFolder, ...
    config.outputPrefix + "_true_residual_history.mat");
completeMarker = fullfile(rawFolder, ...
    config.outputPrefix + "_TRUE_RESIDUAL_HISTORY_COMPLETE.txt");

assert(isfile(endpointPath), ...
    'Run the protected scaling study before its diagnostic histories.');
endpointRuns = readtable(endpointPath, 'TextType', 'string');
diagnosticIterations = config.diagnosticCheckpoints(2:end);

fprintf('\nUntimed original true-residual histories\n');
fprintf('N values %s and checkpoints %s\n\n', ...
    strjoin(string(config.candidateN.'), ', '), ...
    strjoin(string(config.diagnosticCheckpoints.'), ', '));


%% 2. Run one deterministic history or five randomized histories

historyTable = table();
checkTable = table();
for gridIndex = 1:numel(config.candidateN)
    N = config.candidateN(gridIndex);
    fprintf('Diagnostic histories at N=%d\n', N);
    problem = build_weak_preconditioned_tucker_scaling_problem(config, N);

    for methodIndex = 1:numel(config.methodIds)
        methodId = config.methodIds(methodIndex);
        methodName = config.methodNames(methodIndex);
        seeds = diagnostic_seeds(config, methodId);

        for seedIndex = 1:numel(seeds)
            seed = seeds(seedIndex);
            caseFile = diagnostic_case_file( ...
                config, rawFolder, N, methodId, seed);

            if isfile(caseFile)
                fprintf('  Loading %s seed %s\n', ...
                    methodName, printable_seed(seed));
                loaded = load(caseFile, 'caseData');
                caseData = loaded.caseData;
            else
                fprintf('  Running %s seed %s\n', ...
                    methodName, printable_seed(seed));
                [~, info] = ...
                    execute_weak_preconditioned_tucker_scaling_method( ...
                    config, problem, methodId, seed, ...
                    diagnosticIterations);
                referenceResidual = endpoint_residual( ...
                    endpointRuns, N, methodId, seed);
                caseData = build_case_data( ...
                    config, N, methodId, methodName, seed, info, ...
                    referenceResidual);
                save(caseFile, 'config', 'caseData', '-v7.3');
            end

            historyTable = append_table( ...
                historyTable, caseData.historyTable);
            checkTable = append_table(checkTable, caseData.checkTable);
            writetable(historyTable, historyPath);
            writetable(checkTable, checkPath);
            fprintf('    final residual %.6e, endpoint drift %.3e\n', ...
                caseData.checkTable.final_true_relative_residual, ...
                caseData.checkTable.endpoint_relative_drift);
        end
    end
    clear problem
end


%% 3. Save and plot the completed diagnostic evidence

assert(all(checkTable.pass), ...
    'At least one true-residual history check failed.');
save(rawPath, 'config', 'historyTable', 'checkTable', '-v7.3');

markerFile = fopen(completeMarker, 'w');
assert(markerFile >= 0, 'Could not create the completion marker.');
fprintf(markerFile, '%s\n', config.experimentId);
fclose(markerFile);

plot_weak_preconditioned_tucker_four_method_scaling();
fprintf('\nSaved %s\n', historyPath);
fprintf('All diagnostic history checks passed.\n\n');

end


function seeds = diagnostic_seeds(config, methodId)

if methodId == "plain" || methodId == "rhosvd_sgmres"
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
    config, rawFolder, N, methodId, seed)

if isfinite(seed)
    seedText = sprintf('seed%06d', seed);
else
    seedText = 'deterministic';
end
path = fullfile(rawFolder, sprintf( ...
    '%s_true_residual_history_N%03d_%s_%s.mat', ...
    config.outputPrefix, N, methodId, seedText));

end


function residual = endpoint_residual(endpointRuns, N, methodId, seed)

rows = endpointRuns.N == N & endpointRuns.method_id == methodId;
if isfinite(seed)
    rows = rows & endpointRuns.residual_seed == seed;
else
    rows = rows & isnan(endpointRuns.residual_seed);
end
assert(nnz(rows) == 1, ...
    'Expected one endpoint reference for N=%d, method %s, seed %s.', ...
    N, methodId, printable_seed(seed));
residual = endpointRuns.true_relative_residual(rows);

end


function caseData = build_case_data( ...
    config, N, methodId, methodName, seed, info, referenceResidual)

iteration = info.true_residual_iteration(:);
residual = info.true_relative_residual(:);
expectedIteration = config.diagnosticCheckpoints(:);
endpointRelativeDrift = abs(residual(end) - referenceResidual) / ...
    max(referenceResidual, realmin);

iterationComplete = isequal(iteration, expectedIteration);
finiteResiduals = numel(residual) == numel(expectedIteration) && ...
    all(isfinite(residual)) && all(residual > 0);
completedWork = info.iterations == config.maximumIterations && ...
    string(info.stop_reason) == "iteration_limit";
safeExecution = ~logical_field(info, 'breakdown') && ...
    ~logical_field(info, 'stopped_for_basis_memory') && ...
    ~roundsum_rank_cap_active(info);
endpointConsistent = endpointRelativeDrift <= ...
    config.maximumTimingEndpointRelativeDrift;
pass = iterationComplete && finiteResiduals && completedWork && ...
    safeExecution && endpointConsistent;

numberOfRows = numel(iteration);
caseData.historyTable = table( ...
    repmat(N, numberOfRows, 1), ...
    repmat(string(methodId), numberOfRows, 1), ...
    repmat(string(methodName), numberOfRows, 1), ...
    repmat(seed, numberOfRows, 1), iteration, residual, ...
    'VariableNames', {'N', 'method_id', 'method_name', ...
    'residual_seed', 'arnoldi_iteration', ...
    'true_relative_residual'});

caseData.checkTable = table( ...
    N, string(methodId), seed, info.iterations, ...
    string(info.stop_reason), numel(iteration), iterationComplete, ...
    finiteResiduals, safeExecution, residual(end), referenceResidual, ...
    endpointRelativeDrift, pass, ...
    'VariableNames', {'N', 'method_id', 'residual_seed', ...
    'completed_iterations', 'stop_reason', 'checkpoint_count', ...
    'checkpoint_iterations_complete', 'finite_positive_residuals', ...
    'safe_execution', 'final_true_relative_residual', ...
    'reference_true_relative_residual', 'endpoint_relative_drift', ...
    'pass'});

caseData.info = info;

end


function value = logical_field(info, fieldName)

value = false;
if isfield(info, fieldName)
    value = logical(info.(fieldName));
end

end


function active = roundsum_rank_cap_active(info)

active = logical_field(info, 'rank_cap_active');
if isfield(info, 'basis_roundsum_info')
    active = active || any(cellfun(@(oneInfo) ...
        logical_field(oneInfo, 'rank_cap_active'), ...
        info.basis_roundsum_info));
end
if isfield(info, 'solution_roundsum_info')
    active = active || logical_field( ...
        info.solution_roundsum_info, 'rank_cap_active');
end

end


function combined = append_table(combined, nextRows)

if isempty(combined)
    combined = nextRows;
else
    combined = [combined; nextRows];
end

end
