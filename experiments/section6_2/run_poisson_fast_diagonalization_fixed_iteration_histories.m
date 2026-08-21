function run_poisson_fast_diagonalization_fixed_iteration_histories(N)
%RUN_POISSON_FAST_DIAGONALIZATION_FIXED_ITERATION_HISTORIES
% Reconstruct independent residual histories outside the timing study.
%
% The production solvers assemble one solution at the end of a cycle.
% Therefore this diagnostic repeats cycle lengths 1,...,4 and independently
% evaluates the original residual at every endpoint. These calls are never
% used as solver-time measurements.
%
% Thesis/experiment notation (Section 6.2):
%   N                      <->  representative mode size
%   cycleLength            <->  requested Arnoldi endpoint j=1,...,4
%   residualHistory        <->  independently evaluated original residuals
%   rankHistory            <->  Tucker-Arnoldi basis-rank diagnostics
% Despite the legacy variable name, cycleLength does not count restarts;
% each diagnostic call performs one finite run of that many Arnoldi steps.


%% 1. Validate the protected diagnostic case

experimentFolder = fileparts(mfilename('fullpath'));
matlabFolder = fileparts(experimentFolder);
repositoryFolder = fileparts(matlabFolder);
addpath(repositoryFolder);
add_toolboxes();

config = ...
    poisson_fast_diagonalization_fixed_iteration_scaling_config();

if ~isscalar(N) || N ~= round(N) || ...
        ~ismember(N, config.representativeN)
    error('N must be one of the representative diagnostic grid sizes.');
end

processedFolder = fullfile(repositoryFolder, 'experiments', 'processed');
rawFolder = fullfile(repositoryFolder, 'experiments', 'raw');

if ~isfolder(processedFolder)
    mkdir(processedFolder);
end

if ~isfolder(rawFolder)
    mkdir(rawFolder);
end

casePrefix = sprintf('%s_n%d_diagnostic', ...
    char(config.outputPrefix), N);
residualPath = fullfile( ...
    processedFolder, [casePrefix, '_residual_history.csv']);
rankPath = fullfile( ...
    processedFolder, [casePrefix, '_rank_history.csv']);
rawPath = fullfile(rawFolder, [casePrefix, '.mat']);

if isfile(residualPath) || isfile(rankPath) || isfile(rawPath)
    error('The protected diagnostic output already exists.');
end


%% 2. Build the common Poisson problem and preconditioner

gridIndex = find(config.candidateN == N, 1);
preconditionerRank = config.preconditionerTuckerRanks(gridIndex);
h = 1 / (N + 1);
onesVector = ones(N, 1);
A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

gridPoints = (1:N).' / (N + 1);
gaussianVector = exp( ...
    -config.gaussianExponent * (gridPoints - 1/2).^2);

[preconditioner, preconditionerInfo] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner( ...
        A1, preconditionerRank);

assert(preconditionerInfo.is_positive_definite, ...
    'The selected preconditioner is not positive definite.');
assert(preconditionerInfo.spectral_delta <= ...
    config.maximumPreconditionerSpectralDelta, ...
    'The selected preconditioner violates the spectral-quality rule.');

maximumMultilinearRank = ...
    N * ones(1, config.numberOfModes);


%% 3. Evaluate the true original residual after each cycle length

residualHistory = table();
rankHistory = table();
diagnosticInfo = cell(numel(config.methodIds), ...
    config.maximumIteration);

for methodIndex = 1:numel(config.methodIds)

    methodData = build_method_data( ...
        methodIndex, config, A1, gaussianVector, ...
        preconditioner, maximumMultilinearRank);

    initialResidual = initial_original_residual( ...
        methodIndex, methodData);

    residualHistory = append_residual_row( ...
        residualHistory, N, methodIndex, config, ...
        0, initialResidual);

    for cycleLength = 1:config.maximumIteration

        fprintf( ...
            'Diagnostic N=%d, %s, cycle length %d/%d\n', ...
            N, char(config.methodNames(methodIndex)), ...
            cycleLength, config.maximumIteration);

        info = execute_method( ...
            methodIndex, cycleLength, config, methodData, ...
            maximumMultilinearRank);
        diagnosticInfo{methodIndex, cycleLength} = info;

        assert(info.iterations == cycleLength, ...
            'A diagnostic cycle ended before its requested length.');

        residualHistory = append_residual_row( ...
            residualHistory, N, methodIndex, config, ...
            cycleLength, final_original_residual(methodIndex, info));

    end

    if methodIndex > 1

        fullCycleInfo = diagnosticInfo{ ...
            methodIndex, config.maximumIteration};
        basisRanks = fullCycleInfo.basis_ranks;

        for iteration = 0:config.maximumIteration

            ranksAvailable = basisRanks(1:iteration + 1, :);
            cumulativeMaximumRank = max(ranksAvailable, [], 'all');
            newBasisVectorMaximumRank = ...
                max(basisRanks(iteration + 1, :));

            nextRankRow = table( ...
                N, methodIndex, config.methodIds(methodIndex), ...
                config.methodNames(methodIndex), iteration, ...
                cumulativeMaximumRank, newBasisVectorMaximumRank, ...
                'VariableNames', { ...
                    'N', 'method_index', 'method_id', 'method_name', ...
                    'iteration', 'maximum_basis_rank_to_iteration', ...
                    'new_basis_vector_maximum_rank'});

            rankHistory = append_table(rankHistory, nextRankRow);

        end
    end
end


%% 4. Save the diagnostic evidence

writetable(residualHistory, residualPath);
writetable(rankHistory, rankPath);
save(rawPath, ...
    'config', 'N', 'A1', 'preconditionerInfo', ...
    'diagnosticInfo', 'residualHistory', 'rankHistory');

fprintf('Saved diagnostic residual history: %s\n', residualPath);
fprintf('Saved diagnostic rank history: %s\n', rankPath);

end


function methodData = build_method_data( ...
    methodIndex, config, A1, gaussianVector, ...
    preconditioner, maximumMultilinearRank)
%BUILD_METHOD_DATA Construct only the representation needed by one method.

N = length(gaussianVector);

if methodIndex == 1

    methodData.b = kron( ...
        gaussianVector, kron(gaussianVector, gaussianVector));
    methodData.x0 = zeros(N^3, 1);
    inverseMultiplier = ...
        double(full(preconditioner.inverse_eigenvalue_tensor));
    methodData.operator_function = @(x) ...
        poisson_action_full_vector_3d(x, A1, N);
    methodData.preconditioner_function = @(x) ...
        apply_poisson_fast_diagonalization_full_preconditioner( ...
            x, inverseMultiplier);

else

    methodData.F = ttensor(tensor(1, [1, 1, 1]), ...
        {gaussianVector, gaussianVector, gaussianVector});
    methodData.U0 = 0 * methodData.F;
    methodData.operator_function = @(U, tolerance, maximumRanks) ...
        poisson_action_tucker(U, A1, tolerance, maximumRanks);
    methodData.preconditioner_function = ...
        @(Y, tolerance, maximumRanks) ...
        apply_poisson_fast_diagonalization_tucker_preconditioner( ...
            Y, preconditioner, tolerance, maximumRanks);
    methodData.original_residual_function = @(U) ...
        true_residual_tucker_poisson(U, methodData.F, A1);
    methodData.preconditioned_residual_function = @(U) ...
        preconditioned_residual_norm_tucker_poisson_fast_diagonalization( ...
            U, methodData.F, A1, preconditioner, ...
            config.diagnosticTolerance);

end

methodData.maximum_multilinear_rank = maximumMultilinearRank;

end


function info = execute_method( ...
    methodIndex, cycleLength, config, methodData, ...
    maximumMultilinearRank)
%EXECUTE_METHOD Run one untimed-history diagnostic cycle.

if methodIndex == 1

    [~, info] = gmres_left_preconditioned_full_fixed_cycle( ...
        methodData.operator_function, ...
        methodData.preconditioner_function, ...
        methodData.b, methodData.x0, ...
        cycleLength, config.originalTrueResidualTarget, false);

elseif methodIndex == 2

    [~, info] = tucker_gmres_left_preconditioned_fixed( ...
        methodData.operator_function, ...
        methodData.preconditioner_function, ...
        methodData.F, methodData.U0, ...
        cycleLength, ...
        config.internalPreconditionedStoppingTolerance, ...
        config.fixedRecompressionTolerance, ...
        methodData.original_residual_function, ...
        methodData.preconditioned_residual_function, ...
        false, config.maximumTuckerBasisEntries, ...
        maximumMultilinearRank);

else

    [~, info] = tucker_gmres_left_preconditioned_capped_relaxed( ...
        methodData.operator_function, ...
        methodData.preconditioner_function, ...
        methodData.F, methodData.U0, ...
        cycleLength, ...
        config.internalPreconditionedStoppingTolerance, ...
        config.relaxationErrorBudget, ...
        config.maximumRelaxedTolerance, ...
        methodData.original_residual_function, ...
        methodData.preconditioned_residual_function, ...
        false, config.maximumTuckerBasisEntries, ...
        maximumMultilinearRank);

end

end


function residual = initial_original_residual(methodIndex, methodData)
%INITIAL_ORIGINAL_RESIDUAL Evaluate the common initial residual.

if methodIndex == 1
    residual = norm( ...
        methodData.b - ...
        methodData.operator_function(methodData.x0)) / ...
        norm(methodData.b);
else
    residual = ...
        methodData.original_residual_function(methodData.U0);
end

end


function residual = final_original_residual(methodIndex, info)
%FINAL_ORIGINAL_RESIDUAL Extract the independent cycle-end value.

if methodIndex == 1
    residual = info.original_true_relative_residual;
else
    residual = info.true_relative_residual(end);
end

end


function outputTable = append_residual_row( ...
    outputTable, N, methodIndex, config, iteration, residual)
%APPEND_RESIDUAL_ROW Add one original-residual observation.

nextRow = table( ...
    N, methodIndex, config.methodIds(methodIndex), ...
    config.methodNames(methodIndex), iteration, residual, ...
    'VariableNames', { ...
        'N', 'method_index', 'method_id', 'method_name', ...
        'iteration', 'original_true_relative_residual'});

outputTable = append_table(outputTable, nextRow);

end


function outputTable = append_table(outputTable, nextRow)
%APPEND_TABLE Append while preserving the first row's variable types.

if isempty(outputTable)
    outputTable = nextRow;
else
    outputTable = [outputTable; nextRow];
end

end
