function [U, info] = tucker_sgmres_left_preconditioned_plain( ...
    originalOperatorTermsFunction, fixedLinearPreconditionerFunction, ...
    F0, U0, ...
    maximumIteration, sketch, sketchStoppingTolerance, ...
    orthogonalisationWindow, basisCompressionTolerance, ...
    solutionCompressionTolerance, originalTrueResidualFunction, ...
    preconditionedResidualNormFunction, displayProgress, ...
    maximumBasisStorageEntries, maximumMultilinearRank, ...
    diagnosticIterations)
%TUCKER_SGMRES_LEFT_PRECONDITIONED_PLAIN Run Algorithm 5.7.
%
% This is the plain Tucker sGMRES method. The operator callback returns the
% unrounded action A0(X) as a formal Tucker sum. The fixed linear left
% preconditioner is applied term by term. Each residual sketch column is
% therefore computed before deterministic Tucker rounding. The method uses
% a local q-vector Gram Schmidt window and one deterministic Tucker round
% after the local subtractions. Independent residual diagnostics are
% evaluated only after the solver timer stops.


%% 1. Check the inputs

if nargin < 13
    displayProgress = false;
end
if nargin < 14
    maximumBasisStorageEntries = Inf;
end
if nargin < 15
    maximumMultilinearRank = [];
end
if nargin < 16
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
if basisCompressionTolerance <= 0 || basisCompressionTolerance >= 1
    error('basisCompressionTolerance must be between 0 and 1.');
end
if solutionCompressionTolerance <= 0 || ...
        solutionCompressionTolerance >= 1
    error('solutionCompressionTolerance must be between 0 and 1.');
end
if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end
if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end
if ~isequal(double(size(F0)), double(sketch.tensor_dimensions))
    error('The residual sketch dimensions do not match F0.');
end

tensorDimensions = double(size(F0));
numberOfModes = ndims(F0);
maximumRanks = normalise_tucker_rank_cap( ...
    maximumMultilinearRank, tensorDimensions);
diagnosticIterations = normalise_tucker_diagnostic_iterations( ...
    diagnosticIterations, maximumIteration);
normF0 = norm(F0);

if normF0 == 0
    error('The original right hand side F0 must be nonzero.');
end


%% 2. Initialise timing and form the starting residual

wallTimer = tic;
solverTimer = tic;
timing.initial_residual_time_sec = 0;
timing.operator_term_time_sec = 0;
timing.preconditioner_term_time_sec = 0;
timing.operator_exact_sum_time_sec = 0;
timing.residual_sketch_time_sec = 0;
timing.small_least_squares_time_sec = 0;
timing.orthogonalisation_inner_product_time_sec = 0;
timing.orthogonalisation_exact_sum_time_sec = 0;
timing.basis_round_time_sec = 0;
timing.basis_normalisation_time_sec = 0;
timing.solution_assembly_time_sec = 0;

componentTimer = tic;
[initialOperatorTerms, initialOperatorCoefficients, ...
    initialOperatorInfo] = originalOperatorTermsFunction(U0);
check_unrounded_operator_info(initialOperatorInfo);
initialOperatorTerms = initialOperatorTerms(:);
initialOperatorCoefficients = double(initialOperatorCoefficients(:));
timing.operator_term_time_sec = toc(componentTimer);

originalResidualTerms = [{F0}; initialOperatorTerms];
originalResidualCoefficients = [1; -initialOperatorCoefficients];
componentTimer = tic;
[initialOriginalResidualNorm, ~] = tucker_weighted_sum_norm( ...
    originalResidualTerms, originalResidualCoefficients);
initialOriginalResidualScale = formal_sum_scale( ...
    originalResidualTerms, originalResidualCoefficients);
timing.initial_residual_time_sec = toc(componentTimer);

if initialOriginalResidualNorm <= ...
        100 * eps * max(1, initialOriginalResidualScale)
    error(['The initial residual is numerically zero. Return U0 ', ...
           'without starting the Tucker sGMRES cycle.']);
end

componentTimer = tic;
[preconditionedResidualTerms, ~] = ...
    apply_fixed_linear_map_to_terms( ...
        originalResidualTerms, fixedLinearPreconditionerFunction);
timing.preconditioner_term_time_sec = toc(componentTimer);
preconditionedResidualCoefficients = originalResidualCoefficients;
[initialPreconditionedResidualNorm, ~] = tucker_weighted_sum_norm( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients);
initialPreconditionedResidualScale = formal_sum_scale( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients);

if initialPreconditionedResidualNorm <= ...
        100 * eps * max(1, initialPreconditionedResidualScale)
    error(['The fixed left preconditioner annihilated the nonzero ', ...
           'initial residual.']);
end

componentTimer = tic;
g = apply_tucker_row_khatri_rao_sketch_sum( ...
    preconditionedResidualTerms, ...
    preconditionedResidualCoefficients, sketch);
timing.residual_sketch_time_sec = ...
    timing.residual_sketch_time_sec + toc(componentTimer);
normG = norm(g);

if normG <= 100 * eps * max(1, initialPreconditionedResidualNorm)
    error('The starting residual sketch is numerically zero.');
end

componentTimer = tic;
[preconditionedResidual] = tucker_exact_sum_from_terms( ...
    preconditionedResidualTerms, preconditionedResidualCoefficients);
timing.operator_exact_sum_time_sec = ...
    timing.operator_exact_sum_time_sec + toc(componentTimer);

componentTimer = tic;
[R0P, initialRoundInfo] = tucker_round( ...
    preconditionedResidual, basisCompressionTolerance, maximumRanks);
timing.basis_round_time_sec = ...
    timing.basis_round_time_sec + toc(componentTimer);
beta = norm(R0P);

if beta <= 1e-14 * normF0
    error('The rounded starting preconditioned residual is nearly zero.');
end

componentTimer = tic;
V = cell(maximumIteration, 1);
V{1} = (1 / beta) * R0P;
timing.basis_normalisation_time_sec = ...
    timing.basis_normalisation_time_sec + toc(componentTimer);


%% 3. Prepare the stored basis and residual problem

C = zeros(sketch.sketch_size, maximumIteration);
computedSketchRelativeResidual = NaN(maximumIteration, 1);
sketchMatrixCondition = NaN(maximumIteration, 1);
sketchMatrixRank = NaN(maximumIteration, 1);
basisRanks = NaN(maximumIteration, numberOfModes);
basisStorageHistory = NaN(maximumIteration, 1);
basisRanks(1, :) = size(V{1}.core);
basisStorageHistory(1) = tucker_storage_entries( ...
    tensorDimensions, basisRanks(1, :));
rankCapActive = operation_cap_active(initialRoundInfo);
basisMemoryLimitReached = ...
    basisStorageHistory(1) > maximumBasisStorageEntries;
breakdown = false;
stopReason = "iteration_limit";
y = zeros(0, 1);
orthogonalisationInnerProductCount = 0;
orthogonalisationSubtractionCount = 0;
basisRoundCount = 1;
checkpointDiagnosticTimeSec = 0;
checkpointTrueResidual = NaN(numel(diagnosticIterations), 1);
checkpointTrueResidualIteration = NaN(numel(diagnosticIterations), 1);
checkpointDiagnosticIndex = 0;


%% 4. Build the local Arnoldi basis and sketched residual problem

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

    componentTimer = tic;
    C(:, j) = apply_tucker_row_khatri_rao_sketch_sum( ...
        preconditionedOperatorTerms, operatorCoefficients, sketch);
    timing.residual_sketch_time_sec = ...
        timing.residual_sketch_time_sec + toc(componentTimer);

    componentTimer = tic;
    currentC = C(:, 1:j);
    y = lsqminnorm(currentC, g, sqrt(eps));
    sketchResidual = g - currentC * y;
    computedSketchRelativeResidual(j) = norm(sketchResidual) / normG;
    timing.small_least_squares_time_sec = ...
        timing.small_least_squares_time_sec + toc(componentTimer);

    singularValues = svd(currentC, 'econ');
    if isempty(singularValues) || singularValues(1) == 0
        sketchMatrixCondition(j) = Inf;
        sketchMatrixRank(j) = 0;
    else
        numericalThreshold = max(size(currentC)) * ...
            eps(singularValues(1));
        sketchMatrixRank(j) = sum(singularValues > numericalThreshold);
        sketchMatrixCondition(j) = ...
            singularValues(1) / singularValues(end);
    end

    if displayProgress
        fprintf([ ...
            'Plain Tucker sGMRES iteration %2d, ', ...
            'sketched residual %.3e\n'], ...
            j, computedSketchRelativeResidual(j));
    end

    if any(diagnosticIterations == j)
        diagnosticTimer = tic;
        diagnosticSolution = U0;
        for diagnosticBasisIndex = 1:j
            diagnosticSolution = tucker_axpby_round( ...
                diagnosticSolution, 1, V{diagnosticBasisIndex}, ...
                y(diagnosticBasisIndex), ...
                solutionCompressionTolerance, maximumRanks);
        end
        checkpointDiagnosticIndex = checkpointDiagnosticIndex + 1;
        checkpointTrueResidual(checkpointDiagnosticIndex) = ...
            originalTrueResidualFunction(diagnosticSolution);
        checkpointTrueResidualIteration(checkpointDiagnosticIndex) = j;
        checkpointDiagnosticTimeSec = checkpointDiagnosticTimeSec + ...
            toc(diagnosticTimer);
        clear diagnosticSolution
    end

    if computedSketchRelativeResidual(j) <= sketchStoppingTolerance
        stopReason = "sketched_residual";
        break
    end

    if j == maximumIteration
        break
    end

    componentTimer = tic;
    Z = tucker_exact_sum_from_terms( ...
        preconditionedOperatorTerms, operatorCoefficients);
    timing.operator_exact_sum_time_sec = ...
        timing.operator_exact_sum_time_sec + toc(componentTimer);

    firstIndex = max(1, j - orthogonalisationWindow + 1);
    W = Z;
    operatorProductNorm = norm(Z);

    for i = firstIndex:j
        componentTimer = tic;
        coefficient = innerprod(W, V{i});
        timing.orthogonalisation_inner_product_time_sec = ...
            timing.orthogonalisation_inner_product_time_sec + ...
            toc(componentTimer);
        orthogonalisationInnerProductCount = ...
            orthogonalisationInnerProductCount + 1;

        componentTimer = tic;
        W = tucker_axpby_exact(W, 1, V{i}, -coefficient);
        timing.orthogonalisation_exact_sum_time_sec = ...
            timing.orthogonalisation_exact_sum_time_sec + ...
            toc(componentTimer);
        orthogonalisationSubtractionCount = ...
            orthogonalisationSubtractionCount + 1;
    end

    componentTimer = tic;
    [W, roundInfo] = tucker_round( ...
        W, basisCompressionTolerance, maximumRanks);
    timing.basis_round_time_sec = ...
        timing.basis_round_time_sec + toc(componentTimer);
    basisRoundCount = basisRoundCount + 1;
    rankCapActive = rankCapActive || operation_cap_active(roundInfo);

    Wnorm = norm(W);
    breakdownThreshold = max( ...
        100 * eps, basisCompressionTolerance) * ...
        max(1, operatorProductNorm);

    if Wnorm <= breakdownThreshold
        breakdown = true;
        stopReason = "arnoldi_breakdown";
        break
    end

    componentTimer = tic;
    V{j + 1} = (1 / Wnorm) * W;
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


%% 5. Assemble the rounded Tucker solution

componentTimer = tic;
U = U0;
solutionRankCapActive = false;
solutionMaximumLocalError = 0;

for i = 1:numberOfIterations
    [U, additionInfo] = tucker_axpby_round( ...
        U, 1, V{i}, y(i), solutionCompressionTolerance, ...
        maximumRanks);
    solutionRankCapActive = solutionRankCapActive || ...
        operation_cap_active(additionInfo);
    if isfield(additionInfo, 'relative_error_estimate')
        solutionMaximumLocalError = max( ...
            solutionMaximumLocalError, ...
            additionInfo.relative_error_estimate);
    end
end
timing.solution_assembly_time_sec = toc(componentTimer);
rankCapActive = rankCapActive || solutionRankCapActive;
rawSolverTimeSec = toc(solverTimer);
solverTimeSec = rawSolverTimeSec - checkpointDiagnosticTimeSec;


%% 6. Evaluate independent diagnostics after the solver timer

diagnosticTimer = tic;
initialOriginalTrueResidual = originalTrueResidualFunction(U0);
initialPreconditionedRelativeResidual = ...
    preconditionedResidualNormFunction(U0) / ...
    initialPreconditionedResidualNorm;
finalOriginalTrueResidual = originalTrueResidualFunction(U);
finalPreconditionedRelativeResidual = ...
    preconditionedResidualNormFunction(U) / ...
    initialPreconditionedResidualNorm;

% Form the unrounded linear candidate only as a small-problem diagnostic.
% It is not part of the timed Algorithm 5.7 solve.
linearCandidate = U0;
for i = 1:numberOfIterations
    linearCandidate = tucker_axpby_exact( ...
        linearCandidate, 1, V{i}, y(i));
end
linearCandidatePreconditionedRelativeResidual = ...
    preconditionedResidualNormFunction(linearCandidate) / ...
    initialPreconditionedResidualNorm;
linearCandidateOriginalTrueResidual = ...
    originalTrueResidualFunction(linearCandidate);

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
    checkpointDiagnosticTimeSec;
wallTimeSec = toc(wallTimer);

if ~isempty(checkpointTrueResidualIteration) && ...
        checkpointTrueResidualIteration(end) == numberOfIterations
    checkpointTrueResidual(end) = finalOriginalTrueResidual;
else
    checkpointTrueResidual(end + 1, 1) = finalOriginalTrueResidual;
    checkpointTrueResidualIteration(end + 1, 1) = numberOfIterations;
end


%% 7. Return the evidence

usedBasisRows = 1:numberOfIterations;
info.iterations = numberOfIterations;
info.stop_reason = stopReason;
info.breakdown = breakdown;
info.stopped_for_basis_memory = basisMemoryLimitReached;
info.rank_cap_active = rankCapActive;
info.exact_fixed_linear_sketch_before_rounding = true;
info.fixed_linear_preconditioner_rounding_performed = false;
info.sketch_size = sketch.sketch_size;
info.sketch_seed = sketch.random_seed;
info.orthogonalisation_window = orthogonalisationWindow;
info.basis_compression_tolerance = basisCompressionTolerance;
info.solution_compression_tolerance = solutionCompressionTolerance;
info.sketch_stopping_tolerance = sketchStoppingTolerance;
info.cycle_starting_preconditioned_residual_norm = beta;
info.computed_sketch_relative_residual = ...
    computedSketchRelativeResidual(usedBasisRows);
info.sketch_matrix_condition = ...
    sketchMatrixCondition(usedBasisRows);
info.sketch_matrix_rank = sketchMatrixRank(usedBasisRows);
info.independent_preconditioned_relative_residual = ...
    finalPreconditionedRelativeResidual;
info.linear_candidate_preconditioned_relative_residual = ...
    linearCandidatePreconditionedRelativeResidual;
info.linear_candidate_original_true_relative_residual = ...
    linearCandidateOriginalTrueResidual;
info.true_relative_residual = ...
    [initialOriginalTrueResidual; checkpointTrueResidual];
info.true_residual_iteration = ...
    [0; checkpointTrueResidualIteration];
info.initial_preconditioned_relative_residual = ...
    initialPreconditionedRelativeResidual;
info.sketch_to_linear_residual_ratio = ...
    computedSketchRelativeResidual(numberOfIterations) / ...
    linearCandidatePreconditionedRelativeResidual;
info.rounded_solution_residual_gap = abs( ...
    finalPreconditionedRelativeResidual - ...
    linearCandidatePreconditionedRelativeResidual);
info.basis_ranks = basisRanks(usedBasisRows, :);
info.basis_storage_history_entries = ...
    basisStorageHistory(usedBasisRows);
info.peak_basis_storage_entries = ...
    max(info.basis_storage_history_entries);
info.solution_ranks = size(U.core);
info.basis_gram = basisGram;
info.basis_orthogonality_error_fro = basisOrthogonalityError;
info.C = C(:, usedBasisRows);
info.y = y;
info.orthogonalisation_inner_product_count = ...
    orthogonalisationInnerProductCount;
info.orthogonalisation_subtraction_count = ...
    orthogonalisationSubtractionCount;
info.basis_round_count = basisRoundCount;
info.solution_assembly_count = 1;
info.original_true_residual_evaluation_count = ...
    numel(info.true_relative_residual);
info.solution_maximum_local_recompression_error = ...
    solutionMaximumLocalError;
info.solver_time_sec = solverTimeSec;
info.diagnostic_time_sec = diagnosticTimeSec;
info.wall_time_sec = wallTimeSec;
timing.solver_time_sec = solverTimeSec;
timing.diagnostic_time_sec = diagnosticTimeSec;
timing.wall_time_sec = wallTimeSec;
timing.checkpoint_diagnostic_time_sec = checkpointDiagnosticTimeSec;
timing.independent_residual_diagnostic_time_sec = ...
    independentDiagnosticTimeSec;
timing.accounted_solver_time_sec = ...
    timing.initial_residual_time_sec + ...
    timing.operator_term_time_sec + ...
    timing.preconditioner_term_time_sec + ...
    timing.operator_exact_sum_time_sec + ...
    timing.residual_sketch_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.orthogonalisation_exact_sum_time_sec + ...
    timing.basis_round_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solverTimeSec - timing.accounted_solver_time_sec;
info.timing = timing;

end


function active = operation_cap_active(operationInfo)
%OPERATION_CAP_ACTIVE Read an optional rank cap diagnostic.

active = isstruct(operationInfo) && ...
    isfield(operationInfo, 'rank_cap_active') && ...
    logical(operationInfo.rank_cap_active);

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
%APPLY_FIXED_LINEAR_MAP_TO_TERMS Apply P to every exact operator term.

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
%FORMAL_SUM_SCALE Return a cancellation free scale for zero checks.

scale = 0;

for termIndex = 1:numel(terms)
    scale = scale + abs(coefficients(termIndex)) * norm(terms{termIndex});
end

end


function numberOfEntries = tucker_storage_entries( ...
    tensorDimensions, rankVector)
%TUCKER_STORAGE_ENTRIES Count core and factor entries.

numberOfEntries = prod(rankVector) + ...
    sum(tensorDimensions .* rankVector);

end
