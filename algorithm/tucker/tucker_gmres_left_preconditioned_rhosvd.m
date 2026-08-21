function [U, info] = tucker_gmres_left_preconditioned_rhosvd( ...
    originalOperatorTermsFunction, fixedLinearPreconditionerFunction, ...
    F0, U0, maximumIteration, targetTolerance, roundSumSettings, ...
    solutionRoundSumSettings, originalTrueResidualFunction, ...
    preconditionedResidualNormFunction, displayProgress, ...
    maximumBasisStorageEntries)
%TUCKER_GMRES_LEFT_PRECONDITIONED_RHOSVD Full Arnoldi RHOSVD control.
%
% This is the full Arnoldi comparison for the paper-order RHOSVD Tucker
% sGMRES method. The complete left-preconditioned operator result is rounded
% once by adaptive RHOSVD. All current basis tensors are then subtracted in
% one further adaptive RHOSVD RoundSum call.

if nargin < 11, displayProgress = false; end
if nargin < 12, maximumBasisStorageEntries = Inf; end

if ~isa(F0, 'ttensor') || ~isa(U0, 'ttensor') || ...
        ~isequal(size(F0), size(U0))
    error('F0 and U0 must be same-sized Tensor Toolbox ttensors.');
end
if maximumIteration < 1 || maximumIteration ~= floor(maximumIteration)
    error('maximumIteration must be a positive integer.');
end
if targetTolerance <= 0 || targetTolerance >= 1
    error('targetTolerance must be between 0 and 1.');
end
if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end

tensorDimensions = double(size(F0));
numberOfModes = ndims(F0);
basisSettings = check_adaptive_settings( ...
    roundSumSettings, tensorDimensions, 'roundSumSettings');
solutionSettings = check_adaptive_settings( ...
    solutionRoundSumSettings, tensorDimensions, ...
    'solutionRoundSumSettings');

wallTimer = tic;
solverTimer = tic;
timing.operator_term_time_sec = 0;
timing.preconditioner_term_time_sec = 0;
timing.roundsum_time_sec = 0;
timing.orthogonalisation_inner_product_time_sec = 0;
timing.small_least_squares_time_sec = 0;
timing.basis_normalisation_time_sec = 0;
timing.solution_assembly_time_sec = 0;
timing.roundsum_error_diagnostic_time_sec = 0;

[initialOperatorTerms, initialOperatorCoefficients, initialOperatorInfo] = ...
    originalOperatorTermsFunction(U0);
check_unrounded_operator_info(initialOperatorInfo);
originalResidualTerms = [{F0}; initialOperatorTerms(:)];
originalResidualCoefficients = ...
    [1; -double(initialOperatorCoefficients(:))];
[initialOriginalResidualNorm, ~] = tucker_weighted_sum_norm( ...
    originalResidualTerms, originalResidualCoefficients);

if initialOriginalResidualNorm <= 100 * eps * max(1, norm(F0))
    U = U0;
    info.iterations = 0;
    info.stop_reason = "zero_initial_residual";
    info.true_relative_residual = originalTrueResidualFunction(U0);
    info.solver_time_sec = toc(solverTimer);
    info.wall_time_sec = toc(wallTimer);
    return
end

componentTimer = tic;
preconditionedResidualTerms = apply_fixed_linear_map_to_terms( ...
    originalResidualTerms, fixedLinearPreconditionerFunction);
timing.preconditioner_term_time_sec = toc(componentTimer);
[initialPreconditionedResidualNorm, ~] = tucker_weighted_sum_norm( ...
    preconditionedResidualTerms, originalResidualCoefficients);

roundSumCallCount = 1;
[initialDirection, initialRoundInfo] = apply_adaptive_roundsum( ...
    preconditionedResidualTerms, originalResidualCoefficients, ...
    basisSettings, basisSettings.random_seed);
timing.roundsum_time_sec = initialRoundInfo.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    initialRoundInfo.error_diagnostic_time_sec;
beta = norm(initialDirection);

if beta <= max(100 * eps, basisSettings.compression_tolerance) * ...
        max(1, initialPreconditionedResidualNorm)
    error('Adaptive RoundSum produced a zero initial basis tensor.');
end

V = cell(maximumIteration + 1, 1);
V{1} = (1 / beta) * initialDirection;
H = zeros(maximumIteration + 1, maximumIteration);
computedResidual = NaN(maximumIteration, 1);
basisRanks = NaN(maximumIteration + 1, numberOfModes);
basisStorageHistory = NaN(maximumIteration + 1, 1);
basisRanks(1, :) = size(V{1}.core);
basisStorageHistory(1) = tucker_storage_entries( ...
    tensorDimensions, basisRanks(1, :));
roundSumInfo = cell(1 + 2 * maximumIteration, 1);
roundSumSeeds = NaN(1 + 2 * maximumIteration, 1);
roundSumInfo{1} = initialRoundInfo;
roundSumSeeds(1) = basisSettings.random_seed;
roundSumInfoIndex = 1;
stopReason = "iteration_limit";
breakdown = false;
basisMemoryLimitReached = false;
y = zeros(0, 1);

for j = 1:maximumIteration
    componentTimer = tic;
    [operatorTerms, operatorCoefficients, operatorInfo] = ...
        originalOperatorTermsFunction(V{j});
    check_unrounded_operator_info(operatorInfo);
    timing.operator_term_time_sec = timing.operator_term_time_sec + ...
        toc(componentTimer);

    componentTimer = tic;
    preconditionedOperatorTerms = apply_fixed_linear_map_to_terms( ...
        operatorTerms(:), fixedLinearPreconditionerFunction);
    timing.preconditioner_term_time_sec = ...
        timing.preconditioner_term_time_sec + toc(componentTimer);

    roundSumCallCount = roundSumCallCount + 1;
    operatorRoundSeed = basisSettings.random_seed + roundSumCallCount - 1;
    [roundedOperator, operatorRoundInfo] = apply_adaptive_roundsum( ...
        preconditionedOperatorTerms, double(operatorCoefficients(:)), ...
        basisSettings, operatorRoundSeed);
    timing.roundsum_time_sec = timing.roundsum_time_sec + ...
        operatorRoundInfo.algorithm_time_sec;
    timing.roundsum_error_diagnostic_time_sec = ...
        timing.roundsum_error_diagnostic_time_sec + ...
        operatorRoundInfo.error_diagnostic_time_sec;
    roundSumInfoIndex = roundSumInfoIndex + 1;
    roundSumInfo{roundSumInfoIndex} = operatorRoundInfo;
    roundSumSeeds(roundSumInfoIndex) = operatorRoundSeed;

    componentTimer = tic;
    h = zeros(j, 1);
    for basisIndex = 1:j
        h(basisIndex) = innerprod(V{basisIndex}, roundedOperator);
    end
    timing.orthogonalisation_inner_product_time_sec = ...
        timing.orthogonalisation_inner_product_time_sec + ...
        toc(componentTimer);

    roundSumCallCount = roundSumCallCount + 1;
    basisRoundSeed = basisSettings.random_seed + roundSumCallCount - 1;
    [newDirection, basisRoundInfo] = apply_adaptive_roundsum( ...
        [{roundedOperator}; V(1:j)], [1; -h], ...
        basisSettings, basisRoundSeed);
    timing.roundsum_time_sec = timing.roundsum_time_sec + ...
        basisRoundInfo.algorithm_time_sec;
    timing.roundsum_error_diagnostic_time_sec = ...
        timing.roundsum_error_diagnostic_time_sec + ...
        basisRoundInfo.error_diagnostic_time_sec;
    roundSumInfoIndex = roundSumInfoIndex + 1;
    roundSumInfo{roundSumInfoIndex} = basisRoundInfo;
    roundSumSeeds(roundSumInfoIndex) = basisRoundSeed;

    nextNorm = norm(newDirection);
    H(1:j, j) = h;
    H(j + 1, j) = nextNorm;

    componentTimer = tic;
    reducedRightHandSide = zeros(j + 1, 1);
    reducedRightHandSide(1) = beta;
    currentH = H(1:j + 1, 1:j);
    y = lsqminnorm(currentH, reducedRightHandSide, sqrt(eps));
    computedResidual(j) = ...
        norm(reducedRightHandSide - currentH * y) / beta;
    timing.small_least_squares_time_sec = ...
        timing.small_least_squares_time_sec + toc(componentTimer);

    if displayProgress
        fprintf('RHOSVD Tucker GMRES iteration %3d, residual %.3e\n', ...
            j, computedResidual(j));
    end

    if computedResidual(j) <= targetTolerance
        stopReason = "projected_residual";
        break
    end
    if nextNorm <= max(100 * eps, ...
            basisSettings.compression_tolerance) * max(1, norm(roundedOperator))
        breakdown = true;
        stopReason = "arnoldi_breakdown";
        break
    end
    if j == maximumIteration
        break
    end

    componentTimer = tic;
    V{j + 1} = (1 / nextNorm) * newDirection;
    timing.basis_normalisation_time_sec = ...
        timing.basis_normalisation_time_sec + toc(componentTimer);
    basisRanks(j + 1, :) = size(V{j + 1}.core);
    basisStorageHistory(j + 1) = basisStorageHistory(j) + ...
        tucker_storage_entries(tensorDimensions, basisRanks(j + 1, :));

    if basisStorageHistory(j + 1) > maximumBasisStorageEntries
        basisMemoryLimitReached = true;
        stopReason = "basis_memory";
        break
    end
end

numberOfIterations = j;
solutionSeed = solutionSettings.random_seed;
[U, solutionRoundInfo] = apply_adaptive_roundsum( ...
    [{U0}; V(1:numberOfIterations)], ...
    [1; y(1:numberOfIterations)], solutionSettings, solutionSeed);
timing.solution_assembly_time_sec = solutionRoundInfo.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    timing.roundsum_error_diagnostic_time_sec + ...
    solutionRoundInfo.error_diagnostic_time_sec;
rawSolverTime = toc(solverTimer);
solverTime = rawSolverTime - timing.roundsum_error_diagnostic_time_sec;

diagnosticTimer = tic;
initialTrueResidual = originalTrueResidualFunction(U0);
finalTrueResidual = originalTrueResidualFunction(U);
finalPreconditionedResidual = preconditionedResidualNormFunction(U) / ...
    initialPreconditionedResidualNorm;
basisGram = zeros(numberOfIterations);
for rowIndex = 1:numberOfIterations
    for columnIndex = 1:numberOfIterations
        basisGram(rowIndex, columnIndex) = ...
            innerprod(V{rowIndex}, V{columnIndex});
    end
end
independentDiagnosticTime = toc(diagnosticTimer);

usedRows = 1:numberOfIterations;
info.iterations = numberOfIterations;
info.stop_reason = stopReason;
info.breakdown = breakdown;
info.stopped_for_basis_memory = basisMemoryLimitReached;
info.computed_relative_residual = computedResidual(usedRows);
info.true_relative_residual = [initialTrueResidual; finalTrueResidual];
info.true_residual_iteration = [0; numberOfIterations];
info.independent_preconditioned_relative_residual = ...
    finalPreconditionedResidual;
info.initial_preconditioned_residual_norm = ...
    initialPreconditionedResidualNorm;
info.basis_ranks = basisRanks(usedRows, :);
info.basis_storage_history_entries = basisStorageHistory(usedRows);
info.solution_ranks = size(U.core);
info.basis_gram = basisGram;
info.basis_orthogonality_error_fro = ...
    norm(basisGram - eye(numberOfIterations), 'fro');
info.H = H(1:numberOfIterations + 1, 1:numberOfIterations);
info.y = y(1:numberOfIterations);
info.roundsum_seeds = roundSumSeeds(1:roundSumInfoIndex);
info.roundsum_info = roundSumInfo(1:roundSumInfoIndex);
info.solution_roundsum_info = solutionRoundInfo;
info.solver_time_sec = solverTime;
info.diagnostic_time_sec = independentDiagnosticTime + ...
    timing.roundsum_error_diagnostic_time_sec;
info.wall_time_sec = toc(wallTimer);
timing.solver_time_sec = solverTime;
timing.diagnostic_time_sec = info.diagnostic_time_sec;
timing.wall_time_sec = info.wall_time_sec;
timing.accounted_solver_time_sec = timing.operator_term_time_sec + ...
    timing.preconditioner_term_time_sec + timing.roundsum_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solverTime - timing.accounted_solver_time_sec;
info.timing = timing;

end


function [Z, info] = apply_adaptive_roundsum( ...
    terms, coefficients, settings, randomSeed)

[Z, info] = tucker_roundsum_rhosvd_adaptive( ...
    terms, coefficients, settings.oversampling, ...
    settings.compression_tolerance, settings.maximum_rank, randomSeed, ...
    settings.measure_error_diagnostics);

end


function settings = check_adaptive_settings( ...
    settings, tensorDimensions, argumentName)

requiredFields = {'oversampling', 'compression_tolerance', ...
    'maximum_rank', 'random_seed'};
if ~isstruct(settings)
    error('%s must be a structure.', argumentName);
end
for fieldIndex = 1:numel(requiredFields)
    if ~isfield(settings, requiredFields{fieldIndex})
        error('%s is missing field %s.', argumentName, ...
            requiredFields{fieldIndex});
    end
end
if ~isfield(settings, 'measure_error_diagnostics')
    settings.measure_error_diagnostics = true;
end
if settings.oversampling < 0 || ...
        settings.oversampling ~= floor(settings.oversampling)
    error('%s oversampling is invalid.', argumentName);
end
if settings.maximum_rank < 1 || ...
        settings.maximum_rank ~= floor(settings.maximum_rank) || ...
        settings.maximum_rank > min(tensorDimensions)
    error('%s maximum_rank is invalid.', argumentName);
end
if settings.compression_tolerance <= 0 || ...
        settings.compression_tolerance >= 1
    error('%s compression_tolerance is invalid.', argumentName);
end
if settings.random_seed < 0 || ...
        settings.random_seed ~= floor(settings.random_seed)
    error('%s random_seed is invalid.', argumentName);
end

end


function check_unrounded_operator_info(operationInfo)

if ~isstruct(operationInfo) || ...
        ~isfield(operationInfo, 'rounding_performed') || ...
        logical(operationInfo.rounding_performed)
    error('The operator callback must return an unrounded formal sum.');
end

end


function mappedTerms = apply_fixed_linear_map_to_terms(terms, mapFunction)

mappedTerms = cell(size(terms));
for termIndex = 1:numel(terms)
    [mappedTerms{termIndex}, operationInfo] = mapFunction(terms{termIndex});
    if ~isstruct(operationInfo) || ...
            ~isfield(operationInfo, 'rounding_performed') || ...
            logical(operationInfo.rounding_performed)
        error('The preconditioner must be fixed and linear.');
    end
end

end


function numberOfEntries = tucker_storage_entries( ...
    tensorDimensions, rankVector)

numberOfEntries = prod(rankVector) + ...
    sum(tensorDimensions .* rankVector);

end
