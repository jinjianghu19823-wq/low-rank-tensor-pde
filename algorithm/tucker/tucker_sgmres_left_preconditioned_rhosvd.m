function [U, info] = tucker_sgmres_left_preconditioned_rhosvd( ...
    originalOperatorTermsFunction, fixedLinearPreconditionerFunction, ...
    F0, U0, maximumIteration, residualSketch, ...
    sketchStoppingTolerance, orthogonalisationWindow, ...
    basisRoundSumSettings, solutionRoundSumSettings, ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, diagnosticIterations)
%TUCKER_SGMRES_LEFT_PRECONDITIONED_RHOSVD Run Algorithm 5.8.
%
% The operator callback returns the unrounded action A0(X) as a formal
% collection of Tucker terms. The fixed linear preconditioner is then
% applied to every term without recompression. The default paper_order
% variant applies RoundSum before forming each residual sketch column and
% uses batch local projection coefficients. The
% exact_sketch_before_roundsum variant is a legacy option retained only for
% reproducing historical dissertation experiments.
%
% Fixed RoundSum settings use range_sketch_sizes, compression_tolerance,
% maximum_ranks, and random_seed. Adaptive settings use oversampling,
% compression_tolerance, maximum_rank, and random_seed.
% The optional logical field measure_error_diagnostics is true by default.
% The output ranks must satisfy R_n <= k_n. A distinct seed is used for
% every basis RoundSum call.


%% 1. Check the inputs and randomized settings

if nargin < 13
    displayProgress = false;
end
if nargin < 14
    maximumBasisStorageEntries = Inf;
end
if nargin < 15
    diagnosticIterations = [];
end

if ~isa(F0, 'ttensor') || ~isa(U0, 'ttensor')
    error('F0 and U0 must be Tensor Toolbox ttensor objects.');
end
if ~isequal(size(F0), size(U0))
    error('F0 and U0 must have the same tensor dimensions.');
end
if maximumIteration < 1 || maximumIteration ~= floor(maximumIteration)
    error('maximumIteration must be a positive integer.');
end
if orthogonalisationWindow < 1 || ...
        orthogonalisationWindow ~= floor(orthogonalisationWindow) || ...
        orthogonalisationWindow > maximumIteration
    error('orthogonalisationWindow must be between 1 and maximumIteration.');
end
if sketchStoppingTolerance <= 0 || sketchStoppingTolerance >= 1
    error('sketchStoppingTolerance must be between 0 and 1.');
end
if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end
if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end

tensorDimensions = double(size(F0));
numberOfModes = ndims(F0);
normF0 = norm(F0);
diagnosticIterations = normalise_tucker_diagnostic_iterations( ...
    diagnosticIterations, maximumIteration);

if normF0 == 0
    error('The original right hand side F0 must be nonzero.');
end
if ~isequal(tensorDimensions, double(residualSketch.tensor_dimensions))
    error('The residual sketch dimensions do not match F0.');
end

basisSettings = check_roundsum_settings( ...
    basisRoundSumSettings, tensorDimensions, 'basisRoundSumSettings');
solutionSettings = check_roundsum_settings( ...
    solutionRoundSumSettings, tensorDimensions, ...
    'solutionRoundSumSettings');
paperOrder = basisSettings.algorithm_variant == "paper_order";

if basisSettings.random_seed == residualSketch.random_seed || ...
        solutionSettings.random_seed == residualSketch.random_seed
    error(['The residual sketch and RoundSum maps must use ', ...
           'independent random seeds.']);
end


%% 2. Form the exact initial residual as a formal Tucker sum

wallTimer = tic;
solverTimer = tic;
timing.initial_residual_time_sec = 0;
timing.operator_term_time_sec = 0;
timing.preconditioner_term_time_sec = 0;
timing.residual_sketch_time_sec = 0;
timing.roundsum_time_sec = 0;
timing.small_least_squares_time_sec = 0;
timing.orthogonalisation_inner_product_time_sec = 0;
timing.basis_normalisation_time_sec = 0;
timing.solution_assembly_time_sec = 0;
timing.roundsum_error_diagnostic_time_sec = 0;

componentTimer = tic;
[initialOperatorTerms, initialOperatorCoefficients, initialOperatorInfo] = ...
    originalOperatorTermsFunction(U0);
check_unrounded_operator_info(initialOperatorInfo);
initialOperatorTerms = initialOperatorTerms(:);
initialOperatorCoefficients = double(initialOperatorCoefficients(:));
timing.operator_term_time_sec = toc(componentTimer);

originalResidualTerms = [{F0}; initialOperatorTerms];
originalResidualCoefficients = ...
    [1; -initialOperatorCoefficients];
[initialOriginalResidualNorm, ~] = tucker_weighted_sum_norm( ...
    originalResidualTerms, originalResidualCoefficients);
originalResidualScale = formal_sum_scale( ...
    originalResidualTerms, originalResidualCoefficients);

if initialOriginalResidualNorm <= ...
        100 * eps * max(1, originalResidualScale)
    U = U0;
    info = zero_initial_residual_info( ...
        U0, originalTrueResidualFunction, residualSketch, ...
        basisSettings, solutionSettings, wallTimer, solverTimer);
    return
end

componentTimer = tic;
[preconditionedResidualTerms, initialPreconditionerInfo] = ...
    apply_fixed_linear_map_to_terms( ...
    originalResidualTerms, fixedLinearPreconditionerFunction);
timing.preconditioner_term_time_sec = toc(componentTimer);
preconditionedResidualCoefficients = originalResidualCoefficients;

[initialPreconditionedResidualNorm, ~] = tucker_weighted_sum_norm( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients);
preconditionedResidualScale = formal_sum_scale( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients);

if initialPreconditionedResidualNorm <= ...
        100 * eps * max(1, preconditionedResidualScale)
    error(['The fixed left preconditioner annihilated the nonzero ', ...
           'initial residual.']);
end

%% 3. Create the first basis tensor with a fresh RoundSum map

basisRoundSumCallCount = 1;
initialRoundSeed = basisSettings.random_seed;
[R0P, initialRoundInfo] = apply_configured_roundsum( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients, ...
    basisSettings, initialRoundSeed);
timing.roundsum_time_sec = initialRoundInfo.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    initialRoundInfo.error_diagnostic_time_sec;
beta = norm(R0P);

if beta <= max(100 * eps, basisSettings.compression_tolerance) * ...
        max(1, initialPreconditionedResidualNorm)
    error(['RoundSum produced a numerically zero first basis tensor ', ...
           'from a nonzero initial residual.']);
end

componentTimer = tic;
V = cell(maximumIteration, 1);
V{1} = (1 / beta) * R0P;
timing.basis_normalisation_time_sec = toc(componentTimer);

componentTimer = tic;
if paperOrder
    g = apply_tucker_row_khatri_rao_sketch(R0P, residualSketch);
else
    g = apply_tucker_row_khatri_rao_sketch_sum( ...
        preconditionedResidualTerms, ...
        preconditionedResidualCoefficients, residualSketch);
end
timing.residual_sketch_time_sec = toc(componentTimer);
normG = norm(g);

if normG <= 100 * eps * max(1, initialPreconditionedResidualNorm)
    error(['The residual sketch g is numerically zero although the ', ...
           'preconditioned residual is nonzero. Draw a new sketch.']);
end


%% 4. Build the local basis and the exact sketched residual problem

C = zeros(residualSketch.sketch_size, maximumIteration);
computedSketchRelativeResidual = NaN(maximumIteration, 1);
sketchMatrixCondition = NaN(maximumIteration, 1);
sketchMatrixRank = NaN(maximumIteration, 1);
basisRanks = NaN(maximumIteration, numberOfModes);
basisStorageHistory = NaN(maximumIteration, 1);
basisRanks(1, :) = size(V{1}.core);
basisStorageHistory(1) = tucker_storage_entries( ...
    tensorDimensions, basisRanks(1, :));

roundSumSeeds = NaN(1 + 2 * maximumIteration, 1);
roundSumSeeds(1) = initialRoundSeed;
roundSumInfo = cell(size(roundSumSeeds));
roundSumInfo{1} = initialRoundInfo;
roundSumInfoIndex = 1;

stopReason = "iteration_limit";
breakdown = false;
basisMemoryLimitReached = false;
y = zeros(0, 1);
orthogonalisationInnerProductCount = 0;
exactSketchBeforeRoundSum = ~paperOrder;
roundedOperatorSketchAfterRoundSum = paperOrder;
checkpointDiagnosticTimeSec = 0;
checkpointTrueResidual = NaN(numel(diagnosticIterations), 1);
checkpointTrueResidualIteration = NaN(numel(diagnosticIterations), 1);
checkpointDiagnosticIndex = 0;

for j = 1:maximumIteration

    componentTimer = tic;
    [operatorTerms, operatorCoefficients, operatorInfo] = ...
        originalOperatorTermsFunction(V{j});
    check_unrounded_operator_info(operatorInfo);
    operatorTerms = operatorTerms(:);
    operatorCoefficients = double(operatorCoefficients(:));
    timing.operator_term_time_sec = ...
        timing.operator_term_time_sec + toc(componentTimer);

    componentTimer = tic;
    [preconditionedOperatorTerms, ~] = ...
        apply_fixed_linear_map_to_terms( ...
        operatorTerms, fixedLinearPreconditionerFunction);
    timing.preconditioner_term_time_sec = ...
        timing.preconditioner_term_time_sec + toc(componentTimer);

    if paperOrder
        basisRoundSumCallCount = basisRoundSumCallCount + 1;
        operatorRoundSeed = basisSettings.random_seed + ...
            basisRoundSumCallCount - 1;
        [Zhat, operatorRoundInfo] = apply_configured_roundsum( ...
            preconditionedOperatorTerms, operatorCoefficients, ...
            basisSettings, operatorRoundSeed);
        timing.roundsum_time_sec = timing.roundsum_time_sec + ...
            operatorRoundInfo.algorithm_time_sec;
        timing.roundsum_error_diagnostic_time_sec = ...
            timing.roundsum_error_diagnostic_time_sec + ...
            operatorRoundInfo.error_diagnostic_time_sec;
        roundSumInfoIndex = roundSumInfoIndex + 1;
        roundSumSeeds(roundSumInfoIndex) = operatorRoundSeed;
        roundSumInfo{roundSumInfoIndex} = operatorRoundInfo;

        componentTimer = tic;
        C(:, j) = apply_tucker_row_khatri_rao_sketch( ...
            Zhat, residualSketch);
    else
        componentTimer = tic;
        C(:, j) = apply_tucker_row_khatri_rao_sketch_sum( ...
            preconditionedOperatorTerms, operatorCoefficients, ...
            residualSketch);
    end
    timing.residual_sketch_time_sec = ...
        timing.residual_sketch_time_sec + toc(componentTimer);

    componentTimer = tic;
    currentC = C(:, 1:j);
    y = lsqminnorm(currentC, g, sqrt(eps));
    sketchResidual = g - currentC * y;
    computedSketchRelativeResidual(j) = norm(sketchResidual) / normG;
    timing.small_least_squares_time_sec = ...
        timing.small_least_squares_time_sec + toc(componentTimer);

    [sketchMatrixRank(j), sketchMatrixCondition(j)] = ...
        small_matrix_diagnostics(currentC);

    if displayProgress
        fprintf([ ...
            'RHOSVD Tucker sGMRES iteration %2d, ', ...
            'sketched residual %.3e\n'], ...
            j, computedSketchRelativeResidual(j));
    end

    if any(diagnosticIterations == j)
        diagnosticTimer = tic;
        diagnosticTerms = [{U0}; V(1:j)];
        diagnosticCoefficients = [1; y(1:j)];
        [diagnosticSolution, ~] = apply_configured_roundsum( ...
            diagnosticTerms, diagnosticCoefficients, ...
            solutionSettings, solutionSettings.random_seed);
        checkpointDiagnosticIndex = checkpointDiagnosticIndex + 1;
        checkpointTrueResidual(checkpointDiagnosticIndex) = ...
            originalTrueResidualFunction(diagnosticSolution);
        checkpointTrueResidualIteration(checkpointDiagnosticIndex) = j;
        checkpointDiagnosticTimeSec = checkpointDiagnosticTimeSec + ...
            toc(diagnosticTimer);
        clear diagnosticSolution diagnosticTerms diagnosticCoefficients
    end

    if computedSketchRelativeResidual(j) <= sketchStoppingTolerance
        stopReason = "sketched_residual";
        break
    end

    if j == maximumIteration
        break
    end

    if ~paperOrder
        basisRoundSumCallCount = basisRoundSumCallCount + 1;
        operatorRoundSeed = basisSettings.random_seed + ...
            basisRoundSumCallCount - 1;
        [Zhat, operatorRoundInfo] = apply_configured_roundsum( ...
            preconditionedOperatorTerms, operatorCoefficients, ...
            basisSettings, operatorRoundSeed);
        timing.roundsum_time_sec = timing.roundsum_time_sec + ...
            operatorRoundInfo.algorithm_time_sec;
        timing.roundsum_error_diagnostic_time_sec = ...
            timing.roundsum_error_diagnostic_time_sec + ...
            operatorRoundInfo.error_diagnostic_time_sec;
        roundSumInfoIndex = roundSumInfoIndex + 1;
        roundSumSeeds(roundSumInfoIndex) = operatorRoundSeed;
        roundSumInfo{roundSumInfoIndex} = operatorRoundInfo;
    end

    firstIndex = max(1, j - orthogonalisationWindow + 1);
    localCoefficients = zeros(j - firstIndex + 1, 1);

    for i = firstIndex:j
        componentTimer = tic;
        coefficient = innerprod(Zhat, V{i});

        if ~paperOrder
            for earlierIndex = firstIndex:i - 1
                earlierLocalIndex = earlierIndex - firstIndex + 1;
                coefficient = coefficient - ...
                    localCoefficients(earlierLocalIndex) * ...
                    innerprod(V{earlierIndex}, V{i});
                orthogonalisationInnerProductCount = ...
                    orthogonalisationInnerProductCount + 1;
            end
        end

        localCoefficients(i - firstIndex + 1) = coefficient;
        orthogonalisationInnerProductCount = ...
            orthogonalisationInnerProductCount + 1;
        timing.orthogonalisation_inner_product_time_sec = ...
            timing.orthogonalisation_inner_product_time_sec + ...
            toc(componentTimer);
    end

    Wterms = [{Zhat}; V(firstIndex:j)];
    Wcoefficients = [1; -localCoefficients];
    basisRoundSumCallCount = basisRoundSumCallCount + 1;
    basisRoundSeed = basisSettings.random_seed + ...
        basisRoundSumCallCount - 1;
    [Wtilde, oneBasisRoundInfo] = apply_configured_roundsum( ...
        Wterms, Wcoefficients, basisSettings, basisRoundSeed);
    timing.roundsum_time_sec = timing.roundsum_time_sec + ...
        oneBasisRoundInfo.algorithm_time_sec;
    timing.roundsum_error_diagnostic_time_sec = ...
        timing.roundsum_error_diagnostic_time_sec + ...
        oneBasisRoundInfo.error_diagnostic_time_sec;
    roundSumInfoIndex = roundSumInfoIndex + 1;
    roundSumSeeds(roundSumInfoIndex) = basisRoundSeed;
    roundSumInfo{roundSumInfoIndex} = oneBasisRoundInfo;

    Wnorm = norm(Wtilde);
    breakdownThreshold = max( ...
        100 * eps, basisSettings.compression_tolerance) * ...
        max(1, norm(Zhat));

    if Wnorm <= breakdownThreshold
        breakdown = true;
        stopReason = "arnoldi_breakdown";
        break
    end

    componentTimer = tic;
    V{j + 1} = (1 / Wnorm) * Wtilde;
    timing.basis_normalisation_time_sec = ...
        timing.basis_normalisation_time_sec + toc(componentTimer);

    basisRanks(j + 1, :) = size(V{j + 1}.core);
    basisStorageHistory(j + 1) = basisStorageHistory(j) + ...
        tucker_storage_entries( ...
        tensorDimensions, basisRanks(j + 1, :));

    if basisStorageHistory(j + 1) > maximumBasisStorageEntries
        basisMemoryLimitReached = true;
        stopReason = "basis_memory";
        break
    end
end

numberOfIterations = j;
checkpointTrueResidual = ...
    checkpointTrueResidual(1:checkpointDiagnosticIndex);
checkpointTrueResidualIteration = ...
    checkpointTrueResidualIteration(1:checkpointDiagnosticIndex);
roundSumSeeds = roundSumSeeds(1:roundSumInfoIndex);
roundSumInfo = roundSumInfo(1:roundSumInfoIndex);


%% 5. Form the final solution with one independent RoundSum call

solutionTerms = [{U0}; V(1:numberOfIterations)];
solutionCoefficients = [1; y(1:numberOfIterations)];
solutionRoundSeed = solutionSettings.random_seed;

if any(roundSumSeeds == solutionRoundSeed)
    error(['The solution RoundSum seed must differ from every basis ', ...
           'RoundSum seed used in this cycle.']);
end

[U, solutionRoundInfo] = apply_configured_roundsum( ...
    solutionTerms, solutionCoefficients, solutionSettings, ...
    solutionRoundSeed);
timing.solution_assembly_time_sec = ...
    solutionRoundInfo.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    timing.roundsum_error_diagnostic_time_sec + ...
    solutionRoundInfo.error_diagnostic_time_sec;
rawSolverTimeSec = toc(solverTimer);
solverTimeSec = rawSolverTimeSec - ...
    timing.roundsum_error_diagnostic_time_sec - ...
    checkpointDiagnosticTimeSec;


%% 6. Evaluate the two independent residual diagnostics

diagnosticTimer = tic;
initialOriginalTrueResidual = originalTrueResidualFunction(U0);
finalOriginalTrueResidual = originalTrueResidualFunction(U);
initialPreconditionedRelativeResidual = ...
    preconditionedResidualNormFunction(U0) / ...
    initialPreconditionedResidualNorm;
finalPreconditionedRelativeResidual = ...
    preconditionedResidualNormFunction(U) / ...
    initialPreconditionedResidualNorm;

basisGram = zeros(numberOfIterations);

for rowIndex = 1:numberOfIterations
    for columnIndex = 1:numberOfIterations
        basisGram(rowIndex, columnIndex) = ...
            innerprod(V{rowIndex}, V{columnIndex});
    end
end

basisOrthogonalityError = ...
    norm(basisGram - eye(numberOfIterations), 'fro');
independentDiagnosticTimeSec = toc(diagnosticTimer);
diagnosticTimeSec = independentDiagnosticTimeSec + ...
    timing.roundsum_error_diagnostic_time_sec + ...
    checkpointDiagnosticTimeSec;
wallTimeSec = toc(wallTimer);

if ~isempty(checkpointTrueResidualIteration) && ...
        checkpointTrueResidualIteration(end) == numberOfIterations
    checkpointTrueResidual(end) = finalOriginalTrueResidual;
else
    checkpointTrueResidual(end + 1, 1) = finalOriginalTrueResidual;
    checkpointTrueResidualIteration(end + 1, 1) = numberOfIterations;
end


%% 7. Return the numerical evidence and implementation contract

usedBasisRows = 1:numberOfIterations;
info.iterations = numberOfIterations;
info.stop_reason = stopReason;
info.breakdown = breakdown;
info.stopped_for_basis_memory = basisMemoryLimitReached;
info.sketch_size = residualSketch.sketch_size;
info.sketch_seed = residualSketch.random_seed;
info.orthogonalisation_window = orthogonalisationWindow;
info.computed_sketch_relative_residual = ...
    computedSketchRelativeResidual(usedBasisRows);
info.sketch_matrix_condition = ...
    sketchMatrixCondition(usedBasisRows);
info.sketch_matrix_rank = sketchMatrixRank(usedBasisRows);
info.independent_preconditioned_relative_residual = ...
    finalPreconditionedRelativeResidual;
info.true_relative_residual = ...
    [initialOriginalTrueResidual; checkpointTrueResidual];
info.true_residual_iteration = [0; checkpointTrueResidualIteration];
info.initial_preconditioned_relative_residual = ...
    initialPreconditionedRelativeResidual;
info.initial_original_residual_norm = initialOriginalResidualNorm;
info.initial_preconditioned_residual_norm = ...
    initialPreconditionedResidualNorm;
info.basis_ranks = basisRanks(usedBasisRows, :);
info.basis_storage_history_entries = ...
    basisStorageHistory(usedBasisRows);
info.solution_ranks = size(U.core);
info.basis_gram = basisGram;
info.basis_orthogonality_error_fro = basisOrthogonalityError;
info.C = C(:, usedBasisRows);
info.g = g;
info.y = y(1:numberOfIterations);
info.orthogonalisation_inner_product_count = ...
    orthogonalisationInnerProductCount;
info.exact_fixed_linear_sketch_before_roundsum = ...
    exactSketchBeforeRoundSum;
info.rounded_operator_sketch_after_roundsum = ...
    roundedOperatorSketchAfterRoundSum;
info.fixed_linear_preconditioner_rounding_performed = false;
info.local_mgs_is_batch_formula_adaptation = ~paperOrder;
info.batch_local_projection_coefficients = paperOrder;
info.algorithm_variant = basisSettings.algorithm_variant;
info.basis_roundsum_settings = basisSettings;
info.solution_roundsum_settings = solutionSettings;
info.basis_roundsum_seeds = roundSumSeeds;
info.basis_roundsum_info = roundSumInfo;
info.solution_roundsum_seed = solutionRoundSeed;
info.solution_roundsum_info = solutionRoundInfo;
info.original_true_residual_evaluation_count = ...
    numel(info.true_relative_residual);
info.initial_operator_info = initialOperatorInfo;
info.initial_preconditioner_info = initialPreconditionerInfo;
info.solver_time_sec = solverTimeSec;
info.diagnostic_time_sec = diagnosticTimeSec;
info.wall_time_sec = wallTimeSec;
timing.solver_time_sec = solverTimeSec;
timing.diagnostic_time_sec = diagnosticTimeSec;
timing.wall_time_sec = wallTimeSec;
timing.independent_residual_diagnostic_time_sec = ...
    independentDiagnosticTimeSec;
timing.checkpoint_diagnostic_time_sec = checkpointDiagnosticTimeSec;
timing.accounted_diagnostic_time_sec = ...
    timing.independent_residual_diagnostic_time_sec + ...
    timing.roundsum_error_diagnostic_time_sec + ...
    timing.checkpoint_diagnostic_time_sec;
timing.unclassified_diagnostic_time_sec = ...
    diagnosticTimeSec - timing.accounted_diagnostic_time_sec;
timing.accounted_solver_time_sec = ...
    timing.initial_residual_time_sec + ...
    timing.operator_term_time_sec + ...
    timing.preconditioner_term_time_sec + ...
    timing.residual_sketch_time_sec + ...
    timing.roundsum_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solverTimeSec - timing.accounted_solver_time_sec;
info.timing = timing;

end


function [Z, info] = apply_configured_roundsum( ...
    terms, coefficients, settings, randomSeed)
%APPLY_CONFIGURED_ROUNDSUM Apply fixed or adaptive RHOSVD RoundSum.

if settings.adaptive_rank_selection
    [Z, info] = tucker_roundsum_rhosvd_adaptive( ...
        terms, coefficients, settings.oversampling, ...
        settings.compression_tolerance, settings.maximum_rank, ...
        randomSeed, settings.measure_error_diagnostics);
else
    [Z, info] = tucker_roundsum_rhosvd( ...
        terms, coefficients, settings.range_sketch_sizes, ...
        settings.compression_tolerance, settings.maximum_ranks, ...
        randomSeed, settings.measure_error_diagnostics);
    info.adaptive_rank_selection = false;
end

end


function settings = check_roundsum_settings( ...
    settings, tensorDimensions, argumentName)
%CHECK_ROUNDSUM_SETTINGS Validate one basis or solution setting group.

if ~isstruct(settings)
    error('%s must be a structure.', argumentName);
end

requiredFields = {'compression_tolerance', 'random_seed'};
for fieldIndex = 1:numel(requiredFields)
    if ~isfield(settings, requiredFields{fieldIndex})
        error('%s is missing field %s.', ...
            argumentName, requiredFields{fieldIndex});
    end
end

if ~isfield(settings, 'measure_error_diagnostics')
    settings.measure_error_diagnostics = true;
end
if ~isfield(settings, 'algorithm_variant')
    settings.algorithm_variant = "paper_order";
end
if ~isfield(settings, 'adaptive_rank_selection')
    settings.adaptive_rank_selection = false;
end

if ~islogical(settings.measure_error_diagnostics) || ...
        ~isscalar(settings.measure_error_diagnostics)
    error('%s measure_error_diagnostics must be one logical value.', ...
        argumentName);
end
if ~islogical(settings.adaptive_rank_selection) || ...
        ~isscalar(settings.adaptive_rank_selection)
    error('%s adaptive_rank_selection must be one logical value.', ...
        argumentName);
end

settings.algorithm_variant = string(settings.algorithm_variant);
validVariants = ["exact_sketch_before_roundsum", "paper_order"];
if ~isscalar(settings.algorithm_variant) || ...
        ~any(settings.algorithm_variant == validVariants)
    error('%s contains an invalid algorithm_variant.', argumentName);
end

numberOfModes = numel(tensorDimensions);

if settings.adaptive_rank_selection
    adaptiveFields = {'oversampling', 'maximum_rank'};
    for fieldIndex = 1:numel(adaptiveFields)
        if ~isfield(settings, adaptiveFields{fieldIndex})
            error('%s is missing field %s.', ...
                argumentName, adaptiveFields{fieldIndex});
        end
    end

    if ~isscalar(settings.oversampling) || ...
            settings.oversampling < 0 || ...
            settings.oversampling ~= floor(settings.oversampling)
        error('%s oversampling must be a nonnegative integer.', ...
            argumentName);
    end
    if ~isscalar(settings.maximum_rank) || ...
            settings.maximum_rank < 1 || ...
            settings.maximum_rank ~= floor(settings.maximum_rank) || ...
            settings.maximum_rank > min(tensorDimensions)
        error('%s maximum_rank is invalid.', argumentName);
    end
    settings.range_sketch_sizes = [];
    settings.maximum_ranks = [];
else
    fixedFields = {'range_sketch_sizes', 'maximum_ranks'};
    for fieldIndex = 1:numel(fixedFields)
        if ~isfield(settings, fixedFields{fieldIndex})
            error('%s is missing field %s.', ...
                argumentName, fixedFields{fieldIndex});
        end
    end

    settings.range_sketch_sizes = ...
        double(settings.range_sketch_sizes(:).');

    if isscalar(settings.range_sketch_sizes)
        settings.range_sketch_sizes = repmat( ...
            settings.range_sketch_sizes, 1, numberOfModes);
    elseif numel(settings.range_sketch_sizes) ~= numberOfModes
        error('%s range_sketch_sizes must have one value per mode.', ...
            argumentName);
    end

    if any(settings.range_sketch_sizes < 1) || ...
            any(settings.range_sketch_sizes ~= ...
            floor(settings.range_sketch_sizes)) || ...
            any(settings.range_sketch_sizes > tensorDimensions)
        error('%s contains an invalid range sketch size.', argumentName);
    end

    if isempty(settings.maximum_ranks)
        settings.maximum_ranks = settings.range_sketch_sizes;
    else
        settings.maximum_ranks = normalise_tucker_rank_cap( ...
            settings.maximum_ranks, tensorDimensions);
    end

    if any(settings.maximum_ranks > settings.range_sketch_sizes)
        error('%s must satisfy R_n <= k_n in every mode.', argumentName);
    end
end

if ~isscalar(settings.compression_tolerance) || ...
        settings.compression_tolerance <= 0 || ...
        settings.compression_tolerance >= 1
    error('%s compression_tolerance must be between 0 and 1.', ...
        argumentName);
end

if ~isscalar(settings.random_seed) || ...
        ~isfinite(settings.random_seed) || ...
        settings.random_seed < 0 || ...
        settings.random_seed ~= floor(settings.random_seed)
    error('%s random_seed must be a nonnegative integer.', argumentName);
end

end


function check_unrounded_operator_info(operationInfo)
%CHECK_UNROUNDED_OPERATOR_INFO Enforce the fixed linear operator contract.

if ~isstruct(operationInfo) || ...
        ~isfield(operationInfo, 'rounding_performed') || ...
        logical(operationInfo.rounding_performed)
    error(['The operator callback must return an unrounded formal sum ', ...
           'and report rounding_performed=false.']);
end

end


function [mappedTerms, operationInfo] = ...
    apply_fixed_linear_map_to_terms(terms, mapFunction)
%APPLY_FIXED_LINEAR_MAP_TO_TERMS Apply P to each exact operator term.

mappedTerms = cell(size(terms));
operationInfo = cell(size(terms));

for termIndex = 1:numel(terms)
    [mappedTerms{termIndex}, operationInfo{termIndex}] = ...
        mapFunction(terms{termIndex});

    if ~isstruct(operationInfo{termIndex}) || ...
            ~isfield(operationInfo{termIndex}, 'rounding_performed') || ...
            logical(operationInfo{termIndex}.rounding_performed)
        error(['The preconditioner callback must be fixed and linear. ', ...
               'It must report rounding_performed=false.']);
    end
end

end


function scale = formal_sum_scale(terms, coefficients)
%FORMAL_SUM_SCALE A cancellation-free norm scale for zero checks.

scale = 0;

for termIndex = 1:numel(terms)
    scale = scale + abs(coefficients(termIndex)) * norm(terms{termIndex});
end

end


function [numericalRank, conditionEstimate] = ...
    small_matrix_diagnostics(matrix)
%SMALL_MATRIX_DIAGNOSTICS Rank and condition of the residual matrix.

singularValues = svd(matrix, 'econ');

if isempty(singularValues) || singularValues(1) == 0
    numericalRank = 0;
    conditionEstimate = Inf;
else
    threshold = max(size(matrix)) * eps(singularValues(1));
    numericalRank = sum(singularValues > threshold);
    if singularValues(end) <= threshold
        conditionEstimate = Inf;
    else
        conditionEstimate = singularValues(1) / singularValues(end);
    end
end

end


function numberOfEntries = tucker_storage_entries( ...
    tensorDimensions, rankVector)
%TUCKER_STORAGE_ENTRIES Count core and factor entries.

numberOfEntries = prod(rankVector) + ...
    sum(tensorDimensions .* rankVector);

end


function info = zero_initial_residual_info( ...
    U0, originalTrueResidualFunction, residualSketch, ...
    basisSettings, solutionSettings, wallTimer, solverTimer)
%ZERO_INITIAL_RESIDUAL_INFO Return safely without a Krylov division.

solverTimeSec = toc(solverTimer);
trueResidual = originalTrueResidualFunction(U0);

info.iterations = 0;
info.stop_reason = "zero_initial_residual";
info.breakdown = false;
info.stopped_for_basis_memory = false;
info.sketch_size = residualSketch.sketch_size;
info.sketch_seed = residualSketch.random_seed;
info.computed_sketch_relative_residual = zeros(0, 1);
info.independent_preconditioned_relative_residual = 0;
info.true_relative_residual = trueResidual;
info.true_residual_iteration = 0;
info.initial_original_residual_norm = 0;
info.basis_ranks = zeros(0, ndims(U0));
info.solution_ranks = size(U0.core);
info.C = zeros(residualSketch.sketch_size, 0);
info.g = zeros(residualSketch.sketch_size, 1);
info.y = zeros(0, 1);
info.exact_fixed_linear_sketch_before_roundsum = true;
info.fixed_linear_preconditioner_rounding_performed = false;
info.basis_roundsum_settings = basisSettings;
info.solution_roundsum_settings = solutionSettings;
info.basis_roundsum_seeds = zeros(0, 1);
info.solution_roundsum_seed = NaN;
info.solver_time_sec = solverTimeSec;
info.diagnostic_time_sec = 0;
info.wall_time_sec = toc(wallTimer);

end
