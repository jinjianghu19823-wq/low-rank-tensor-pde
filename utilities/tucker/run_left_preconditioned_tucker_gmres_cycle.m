function [U, info] = run_left_preconditioned_tucker_gmres_cycle(originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, targetTolerance, fixedCompressionTolerance, ...
    toleranceMode, originalTrueResidualFunction, ...
    preconditionedResidualNormFunction, displayProgress, ...
    maximumBasisStorageEntries, maximumMultilinearRank, ...
    relaxationErrorBudget, maximumRelaxedTolerance, ...
    diagnosticIterations)
%RUN_LEFT_PRECONDITIONED_TUCKER_GMRES_CYCLE Run one Tucker-GMRES cycle.
%
% This shared cycle implements the same left-preconditioned Arnoldi process
% for two controlled methods:
%
%   toleranceMode = "fixed"
%       Use fixedCompressionTolerance in every recompression.
%
%   toleranceMode = "relaxed"
%       Use the residual-based relaxation rule from Section 5.5. At step j,
%       use
%
%           eta_j = relaxationErrorBudget ...
%                   / (computedResidualNorm_{j-1} / beta).
%
%       The optional maximumRelaxedTolerance limits eta_j. Existing callers
%       recover the uncapped thesis rule because relaxationErrorBudget
%       defaults to targetTolerance and maximumRelaxedTolerance to 1-eps.
%
% The small Hessenberg residual controls the Arnoldi loop. After that loop
% ends, the Tucker solution is assembled once. The original true residual
% then decides convergence. The independently recomputed preconditioned
% residual is a cycle-end diagnostic used to measure the residual gap.
%
% info.timing contains two levels. Its top-level phase fields are
% non-overlapping and close back to solver_time_sec after adding
% unclassified_solver_time_sec. info.timing.kernel contains nested leaf
% timings such as QR, STHOSVD SVD, exact-core construction, Hadamard rank
% multiplication, and DST transforms. info.timing.kernel_by_phase attributes
% the same leaf timings to the operator, preconditioner, outer round,
% Arnoldi subtraction, and solution assembly callers. The nested fields
% explain a phase but must not be added to the phase timings again.
%
% Function-handle interfaces:
%
%   [Y, operationInfo] = originalOperatorFunction(X, tolerance, maxRanks)
%   [Z, operationInfo] = preconditionerFunction(Y, tolerance, maxRanks)
%
% Both operations return Tucker tensors. Their operationInfo structures may
% record local sequential-rounding errors and active rank caps.
%
% Thesis notation (Section 5.5, left-preconditioned relaxed Tucker-GMRES):
%   originalOperatorFunction       <->  \mathcal A_0
%   preconditionerFunction         <->  \mathcal P
%   F0, U0, U                      <->  \mathcal B_0, \mathcal X_0, \mathcal X_j
%   maximumIteration               <->  \ell_max (or m in the error bound)
%   targetTolerance                <->  \varepsilon (internal residual threshold)
%   fixedCompressionTolerance      <->  \eta in the fixed method
%   relaxationErrorBudget          <->  \varepsilon in the \eta_j numerator
%   maximumRelaxedTolerance        <->  \eta_max
%   maximumMultilinearRank         <->  \boldsymbol R=(R_1,...,R_d)
%   R0P, beta                      <->  \widetilde{\mathcal R}_0^P,
%                                       ||\widetilde{\mathcal R}_0^P||_F
%   V{j}, W                        <->  \mathcal V_j, recompressed A_P(V_j)
%   eta                            <->  \eta_j
%   H(1:j+1,1:j), y                <->  \bar H_j, y_j
%   computedResidualNorm           <->  ||\widetilde r_j^P||_2
%   originalTrueResidualFunction   <->  original true relative residual
%   preconditionedResidualNormFunction
%                                   <->  ||\mathcal P(\mathcal B_0-
%                                            \mathcal A_0(\mathcal X))||_F
% Here \mathcal A_P=\mathcal P\circ\mathcal A_0. The implementation normalises
% computed and diagnostic preconditioned residuals by beta, the starting
% preconditioned-residual norm.


%% 1. Check the inputs

if ~isa(F0, 'ttensor') || ~isa(U0, 'ttensor')
    error('F0 and U0 must be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(F0), size(U0))
    error('F0 and U0 must have the same tensor dimensions.');
end

if maximumIteration < 1 || maximumIteration ~= floor(maximumIteration)
    error('maximumIteration must be a positive integer.');
end

if targetTolerance <= 0 || targetTolerance >= 1
    error('targetTolerance must be between 0 and 1.');
end

if fixedCompressionTolerance <= 0 || fixedCompressionTolerance >= 1
    error('fixedCompressionTolerance must be between 0 and 1.');
end

if nargin < 14
    relaxationErrorBudget = targetTolerance;
end

if nargin < 15
    maximumRelaxedTolerance = 1 - eps;
end
if nargin < 16
    diagnosticIterations = [];
end
diagnosticIterations = normalise_tucker_diagnostic_iterations( ...
    diagnosticIterations, maximumIteration);

if relaxationErrorBudget <= 0 || relaxationErrorBudget >= 1
    error('relaxationErrorBudget must be between 0 and 1.');
end

if maximumRelaxedTolerance <= 0 || maximumRelaxedTolerance >= 1
    error('maximumRelaxedTolerance must be between 0 and 1.');
end

toleranceMode = lower(string(toleranceMode));

if toleranceMode ~= "fixed" && toleranceMode ~= "relaxed"
    error('toleranceMode must be "fixed" or "relaxed".');
end

if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end

if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end

maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, size(F0));

normF0 = norm(F0);

if normF0 == 0
    error('The original right-hand side F0 must be nonzero.');
end

if toleranceMode == "relaxed"
    startingCompressionTolerance = min(relaxationErrorBudget, maximumRelaxedTolerance);
    solutionCompressionTolerance = startingCompressionTolerance;
else
    solutionCompressionTolerance = fixedCompressionTolerance;
    startingCompressionTolerance = fixedCompressionTolerance;
end


%% 2. Form the preconditioned starting residual

wallTimer = tic;
solverElapsed = 0;

phaseTiming.operator_application_time_sec = 0;
phaseTiming.preconditioner_application_time_sec = 0;
phaseTiming.outer_round_time_sec = 0;
phaseTiming.initial_residual_formation_time_sec = 0;
phaseTiming.orthogonalisation_inner_product_time_sec = 0;
phaseTiming.orthogonalisation_subtraction_round_time_sec = 0;
phaseTiming.small_least_squares_time_sec = 0;
phaseTiming.basis_normalisation_time_sec = 0;
phaseTiming.initial_basis_normalisation_time_sec = 0;
phaseTiming.solution_assembly_time_sec = 0;
phaseTiming.original_true_residual_time_sec = 0;
phaseTiming.preconditioned_residual_diagnostic_time_sec = 0;

phaseTiming.operator_application_history_sec = ...
    zeros(maximumIteration, 1);
phaseTiming.preconditioner_application_history_sec = ...
    zeros(maximumIteration, 1);
phaseTiming.outer_round_history_sec = zeros(maximumIteration, 1);
phaseTiming.orthogonalisation_inner_product_history_sec = ...
    zeros(maximumIteration, 1);
phaseTiming.orthogonalisation_subtraction_round_history_sec = ...
    zeros(maximumIteration, 1);
phaseTiming.small_least_squares_history_sec = ...
    zeros(maximumIteration, 1);
phaseTiming.basis_normalisation_history_sec = ...
    zeros(maximumIteration, 1);

kernelTiming = empty_tucker_kernel_timing();
kernelTimingByPhase = empty_phase_kernel_timing();

solverTimer = tic;

componentTimer = tic;
[AU0, initialOperatorInfo] = originalOperatorFunction(U0, startingCompressionTolerance, maximumRanks);
initialOperatorApplicationTime = toc(componentTimer);
phaseTiming.operator_application_time_sec = ...
    phaseTiming.operator_application_time_sec + ...
    initialOperatorApplicationTime;
kernelTiming = add_operation_kernel_timing( ...
    kernelTiming, initialOperatorInfo);
kernelTimingByPhase.operator_application = ...
    add_operation_kernel_timing( ...
        kernelTimingByPhase.operator_application, initialOperatorInfo);

% Form B0-A0(U0) exactly in the represented Tucker subspaces. The
% preconditioner action then performs its own controlled recompressions.
componentTimer = tic;
originalResidual = tucker_axpby_exact(F0, 1, AU0, -1);
phaseTiming.initial_residual_formation_time_sec = toc(componentTimer);

componentTimer = tic;
[preconditionedResidual, initialPreconditionerInfo] = preconditionerFunction(originalResidual, startingCompressionTolerance, maximumRanks);
initialPreconditionerApplicationTime = toc(componentTimer);
phaseTiming.preconditioner_application_time_sec = ...
    phaseTiming.preconditioner_application_time_sec + ...
    initialPreconditionerApplicationTime;
kernelTiming = add_operation_kernel_timing( ...
    kernelTiming, initialPreconditionerInfo);
kernelTimingByPhase.preconditioner_application = ...
    add_operation_kernel_timing( ...
        kernelTimingByPhase.preconditioner_application, ...
        initialPreconditionerInfo);

componentTimer = tic;
[R0P, initialOuterRoundInfo] = tucker_round(preconditionedResidual, startingCompressionTolerance, maximumRanks);
initialOuterRoundTime = toc(componentTimer);
phaseTiming.outer_round_time_sec = ...
    phaseTiming.outer_round_time_sec + initialOuterRoundTime;
kernelTiming = add_operation_kernel_timing( ...
    kernelTiming, initialOuterRoundInfo);
kernelTimingByPhase.outer_round = add_operation_kernel_timing( ...
    kernelTimingByPhase.outer_round, initialOuterRoundInfo);

beta = norm(R0P);
solverElapsed = solverElapsed + toc(solverTimer);

if beta <= 1e-14 * normF0
    error(['The preconditioned starting residual is nearly zero. ', 'Use a more accurate preconditioner application or ', 'recompression tolerance.']);
end


%% 3. Evaluate the initial residuals independently

diagnosticTimer = tic;
initialOriginalTrueResidual = originalTrueResidualFunction(U0);
initialOriginalTrueResidualTime = toc(diagnosticTimer);
phaseTiming.original_true_residual_time_sec = ...
    phaseTiming.original_true_residual_time_sec + ...
    initialOriginalTrueResidualTime;

diagnosticTimer = tic;
initialPreconditionedResidualNorm = preconditionedResidualNormFunction(U0);
initialPreconditionedResidualDiagnosticTime = toc(diagnosticTimer);
phaseTiming.preconditioned_residual_diagnostic_time_sec = ...
    phaseTiming.preconditioned_residual_diagnostic_time_sec + ...
    initialPreconditionedResidualDiagnosticTime;
diagnosticElapsed = phaseTiming.original_true_residual_time_sec + ...
    phaseTiming.preconditioned_residual_diagnostic_time_sec;

phaseTiming.initial_operator_application_time_sec = ...
    initialOperatorApplicationTime;
phaseTiming.initial_preconditioner_application_time_sec = ...
    initialPreconditionerApplicationTime;
phaseTiming.initial_outer_round_time_sec = initialOuterRoundTime;
phaseTiming.initial_original_true_residual_time_sec = ...
    initialOriginalTrueResidualTime;
phaseTiming.initial_preconditioned_residual_diagnostic_time_sec = ...
    initialPreconditionedResidualDiagnosticTime;

if initialOriginalTrueResidual <= targetTolerance

    U = U0;
    wallElapsed = toc(wallTimer);
    info = empty_cycle_info(toleranceMode, maximumRanks, beta, initialOriginalTrueResidual, initialPreconditionedResidualNorm / beta, ...
        solverElapsed, diagnosticElapsed, wallElapsed, ...
        targetTolerance, fixedCompressionTolerance, ...
        relaxationErrorBudget, maximumRelaxedTolerance);
    info.timing = finalise_cycle_timing( ...
        phaseTiming, 0, solverElapsed, diagnosticElapsed, ...
        wallElapsed, kernelTiming, kernelTimingByPhase);
    return

end


%% 4. Prepare the Arnoldi basis and histories

numberOfModes = ndims(F0);

V = cell(maximumIteration + 1, 1);
H = zeros(maximumIteration + 1, maximumIteration);

originalTrueResidualHistory = initialOriginalTrueResidual;
originalTrueResidualIteration = 0;
truePreconditionedRelativeResidualHistory = initialPreconditionedResidualNorm / beta;
truePreconditionedResidualIteration = 0;
computedPreconditionedRelativeResidualHistory = zeros(maximumIteration, 1);
etaHistory = zeros(maximumIteration, 1);

basisRanks = zeros(maximumIteration + 1, numberOfModes);
basisStorageHistory = zeros(maximumIteration + 1, 1);

solverTimeHistory = zeros(maximumIteration, 1);
diagnosticTimeHistory = zeros(maximumIteration, 1);
wallTimeHistory = zeros(maximumIteration, 1);

rankCapActiveHistory = false(maximumIteration, 1);
operatorProductMaximumLocalErrorHistory = zeros(maximumIteration, 1);
orthogonalisationMaximumLocalErrorHistory = zeros(maximumIteration, 1);
solutionMaximumLocalErrorHistory = zeros(maximumIteration, 1);

solverTimer = tic;
V{1} = (1 / beta) * R0P;
phaseTiming.initial_basis_normalisation_time_sec = toc(solverTimer);
phaseTiming.basis_normalisation_time_sec = ...
    phaseTiming.basis_normalisation_time_sec + ...
    phaseTiming.initial_basis_normalisation_time_sec;
solverElapsed = solverElapsed + ...
    phaseTiming.initial_basis_normalisation_time_sec;
basisRanks(1, :) = size(V{1}.core);
basisStorageHistory(1) = tucker_entries_from_rank_local(size(F0), basisRanks(1, :));

if basisStorageHistory(1) > maximumBasisStorageEntries
    error('The initial Tucker basis exceeds the storage safety limit.');
end

computedResidualNormPrevious = beta;
basisMemoryLimitReached = false;
cycleStoppedOnComputedResidual = false;
stopReason = "iteration_limit";

[initialCapActive, initialLocalError] = summarise_operation_info(initialOperatorInfo, initialPreconditionerInfo, initialOuterRoundInfo);


%% 5. Run the left-preconditioned Arnoldi cycle

for j = 1:maximumIteration

    if toleranceMode == "relaxed"

        previousRelativeResidual = computedResidualNormPrevious / beta;

        eta = relaxationErrorBudget / previousRelativeResidual;

        eta = min([eta, maximumRelaxedTolerance, 1 - eps]);

    else
        eta = fixedCompressionTolerance;
    end

    etaHistory(j) = eta;
    solverTimer = tic;


    %% 5a. Apply the recompressed preconditioned operator product

    componentTimer = tic;
    [A0V, operatorInfo] = originalOperatorFunction(V{j}, eta, maximumRanks);
    oneOperatorApplicationTime = toc(componentTimer);
    phaseTiming.operator_application_time_sec = ...
        phaseTiming.operator_application_time_sec + ...
        oneOperatorApplicationTime;
    phaseTiming.operator_application_history_sec(j) = ...
        oneOperatorApplicationTime;
    kernelTiming = add_operation_kernel_timing( ...
        kernelTiming, operatorInfo);
    kernelTimingByPhase.operator_application = ...
        add_operation_kernel_timing( ...
            kernelTimingByPhase.operator_application, operatorInfo);

    componentTimer = tic;
    [preconditionedProduct, preconditionerInfo] = preconditionerFunction(A0V, eta, maximumRanks);
    onePreconditionerApplicationTime = toc(componentTimer);
    phaseTiming.preconditioner_application_time_sec = ...
        phaseTiming.preconditioner_application_time_sec + ...
        onePreconditionerApplicationTime;
    phaseTiming.preconditioner_application_history_sec(j) = ...
        onePreconditionerApplicationTime;
    kernelTiming = add_operation_kernel_timing( ...
        kernelTiming, preconditionerInfo);
    kernelTimingByPhase.preconditioner_application = ...
        add_operation_kernel_timing( ...
            kernelTimingByPhase.preconditioner_application, ...
            preconditionerInfo);

    componentTimer = tic;
    [W, productRoundInfo] = tucker_round(preconditionedProduct, eta, maximumRanks);
    oneOuterRoundTime = toc(componentTimer);
    phaseTiming.outer_round_time_sec = ...
        phaseTiming.outer_round_time_sec + oneOuterRoundTime;
    phaseTiming.outer_round_history_sec(j) = oneOuterRoundTime;
    kernelTiming = add_operation_kernel_timing( ...
        kernelTiming, productRoundInfo);
    kernelTimingByPhase.outer_round = add_operation_kernel_timing( ...
        kernelTimingByPhase.outer_round, productRoundInfo);

    arnoldiProductNorm = norm(W);

    [productCapActive, productLocalError] = summarise_operation_info(operatorInfo, preconditionerInfo, productRoundInfo);

    rankCapActiveHistory(j) = productCapActive;
    operatorProductMaximumLocalErrorHistory(j) = productLocalError;


    %% 5b. Perform one modified Gram--Schmidt pass

    for i = 1:j

        componentTimer = tic;
        correction = innerprod(V{i}, W);
        oneInnerProductTime = toc(componentTimer);
        phaseTiming.orthogonalisation_inner_product_time_sec = ...
            phaseTiming.orthogonalisation_inner_product_time_sec + ...
            oneInnerProductTime;
        phaseTiming.orthogonalisation_inner_product_history_sec(j) = ...
            phaseTiming.orthogonalisation_inner_product_history_sec(j) + ...
            oneInnerProductTime;

        H(i, j) = correction;

        componentTimer = tic;
        [W, subtractionInfo] = tucker_axpby_round(W, 1, V{i}, -correction, eta, maximumRanks);
        oneSubtractionRoundTime = toc(componentTimer);
        phaseTiming.orthogonalisation_subtraction_round_time_sec = ...
            phaseTiming.orthogonalisation_subtraction_round_time_sec + ...
            oneSubtractionRoundTime;
        phaseTiming.orthogonalisation_subtraction_round_history_sec(j) = ...
            phaseTiming.orthogonalisation_subtraction_round_history_sec(j) + ...
            oneSubtractionRoundTime;
        kernelTiming = add_operation_kernel_timing( ...
            kernelTiming, subtractionInfo);
        kernelTimingByPhase.orthogonalisation_subtraction_and_round = ...
            add_operation_kernel_timing( ...
                kernelTimingByPhase.orthogonalisation_subtraction_and_round, ...
                subtractionInfo);

        rankCapActiveHistory(j) = rankCapActiveHistory(j) || subtractionInfo.rank_cap_active;

        orthogonalisationMaximumLocalErrorHistory(j) = max(orthogonalisationMaximumLocalErrorHistory(j), subtractionInfo.relative_error_estimate);

    end


    %% 5c. Form the Hessenberg problem

    H(j + 1, j) = norm(W);

    % Implementation safeguard, not a theorem in Section 5.5: if the
    % remaining direction is no larger than the requested recompression
    % accuracy, normalising it could turn truncation noise into a new basis
    % tensor. The factors 100 and max(1,arnoldiProductNorm) are conservative
    % numerical scaling choices, not constants from the inexact-GMRES bound.
    breakdownThreshold = max(100 * eps, eta) * max(1, arnoldiProductNorm);
    numericalArnoldiBreakdown = H(j + 1, j) <= breakdownThreshold;

    if numericalArnoldiBreakdown
        H(j + 1, j) = 0;
    end

    basisRanks(j + 1, :) = size(W.core);
    basisStorageHistory(j + 1) = basisStorageHistory(j) + tucker_entries_from_rank_local(size(F0), basisRanks(j + 1, :));

    if basisStorageHistory(j + 1) > maximumBasisStorageEntries
        basisMemoryLimitReached = true;
    end

    componentTimer = tic;
    smallRightHandSide = zeros(j + 1, 1);
    smallRightHandSide(1) = beta;

    % The algorithm asks for a least-squares minimiser. Lsqminnorm also
    % returns a well-defined minimum-norm solution when the small
    % Hessenberg matrix is numerically rank deficient.
    y = lsqminnorm(H(1:j + 1, 1:j), smallRightHandSide, sqrt(eps));
    smallResidual = smallRightHandSide - H(1:j + 1, 1:j) * y;
    computedResidualNorm = norm(smallResidual);

    oneSmallLeastSquaresTime = toc(componentTimer);
    phaseTiming.small_least_squares_time_sec = ...
        phaseTiming.small_least_squares_time_sec + ...
        oneSmallLeastSquaresTime;
    phaseTiming.small_least_squares_history_sec(j) = ...
        oneSmallLeastSquaresTime;

    computedPreconditionedRelativeResidualHistory(j) = computedResidualNorm / beta;


    solverElapsed = solverElapsed + toc(solverTimer);

    if any(diagnosticIterations == j)
        diagnosticTimer = tic;
        diagnosticSolution = U0;
        for diagnosticBasisIndex = 1:j
            diagnosticSolution = tucker_axpby_round( ...
                diagnosticSolution, 1, V{diagnosticBasisIndex}, ...
                y(diagnosticBasisIndex), ...
                solutionCompressionTolerance, maximumRanks);
        end
        checkpointTrueResidual = ...
            originalTrueResidualFunction(diagnosticSolution);
        diagnosticElapsed = diagnosticElapsed + toc(diagnosticTimer);
        originalTrueResidualHistory = ...
            [originalTrueResidualHistory; checkpointTrueResidual]; %#ok<AGROW>
        originalTrueResidualIteration = ...
            [originalTrueResidualIteration; j]; %#ok<AGROW>
        clear diagnosticSolution
    end


    solverTimeHistory(j) = solverElapsed;
    diagnosticTimeHistory(j) = diagnosticElapsed;
    wallTimeHistory(j) = toc(wallTimer);

    if displayProgress
        fprintf([ ...
            'Iteration %3d: eta = %.3e, computed preconditioned ', ...
            'residual = %.3e\n'], ...
            j, eta, ...
            computedPreconditionedRelativeResidualHistory(j));
    end


    %% 5d. Apply the stopping and safety tests

    if computedPreconditionedRelativeResidualHistory(j) <= targetTolerance
        cycleStoppedOnComputedResidual = true;
        stopReason = "computed_preconditioned_target";
        break
    end

    if computedPreconditionedRelativeResidualHistory(j) <= 100 * eps
        cycleStoppedOnComputedResidual = true;
        stopReason = "computed_preconditioned_floor";
        break
    end

    if basisMemoryLimitReached
        stopReason = "basis_memory";
        break
    end

    if numericalArnoldiBreakdown
        stopReason = "arnoldi_breakdown";
        break
    end

    solverTimer = tic;
    V{j + 1} = (1 / H(j + 1, j)) * W;
    oneBasisNormalisationTime = toc(solverTimer);
    phaseTiming.basis_normalisation_time_sec = ...
        phaseTiming.basis_normalisation_time_sec + ...
        oneBasisNormalisationTime;
    phaseTiming.basis_normalisation_history_sec(j) = ...
        oneBasisNormalisationTime;
    solverElapsed = solverElapsed + oneBasisNormalisationTime;
    solverTimeHistory(j) = solverElapsed;
    wallTimeHistory(j) = toc(wallTimer);
    computedResidualNormPrevious = computedResidualNorm;

end


%% 6. Assemble the Tucker solution once

numberOfCompletedIterations = j;
solverTimer = tic;
U = U0;

for i = 1:numberOfCompletedIterations

    [U, solutionAdditionInfo] = tucker_axpby_round(U, 1, V{i}, y(i), solutionCompressionTolerance, maximumRanks);
    kernelTiming = add_operation_kernel_timing( ...
        kernelTiming, solutionAdditionInfo);
    kernelTimingByPhase.solution_assembly = ...
        add_operation_kernel_timing( ...
            kernelTimingByPhase.solution_assembly, solutionAdditionInfo);

    rankCapActiveHistory(numberOfCompletedIterations) = rankCapActiveHistory(numberOfCompletedIterations) || solutionAdditionInfo.rank_cap_active;

    solutionMaximumLocalErrorHistory(numberOfCompletedIterations) = max(solutionMaximumLocalErrorHistory(numberOfCompletedIterations), solutionAdditionInfo.relative_error_estimate);

end

solutionRanks = size(U.core);
solutionRankIteration = numberOfCompletedIterations;
phaseTiming.solution_assembly_time_sec = toc(solverTimer);
solverElapsed = solverElapsed + phaseTiming.solution_assembly_time_sec;


%% 7. Recompute both cycle-end residuals independently

diagnosticTimer = tic;
finalOriginalTrueResidual = originalTrueResidualFunction(U);
finalOriginalTrueResidualTime = toc(diagnosticTimer);
phaseTiming.original_true_residual_time_sec = ...
    phaseTiming.original_true_residual_time_sec + ...
    finalOriginalTrueResidualTime;

diagnosticTimer = tic;
finalPreconditionedResidualNorm = preconditionedResidualNormFunction(U);
finalPreconditionedResidualDiagnosticTime = toc(diagnosticTimer);
phaseTiming.preconditioned_residual_diagnostic_time_sec = ...
    phaseTiming.preconditioned_residual_diagnostic_time_sec + ...
    finalPreconditionedResidualDiagnosticTime;

finalTruePreconditionedRelativeResidual = finalPreconditionedResidualNorm / beta;

diagnosticElapsed = phaseTiming.original_true_residual_time_sec + ...
    phaseTiming.preconditioned_residual_diagnostic_time_sec;

if originalTrueResidualIteration(end) == numberOfCompletedIterations
    originalTrueResidualHistory(end) = finalOriginalTrueResidual;
else
    originalTrueResidualHistory = ...
        [originalTrueResidualHistory; finalOriginalTrueResidual];
    originalTrueResidualIteration = ...
        [originalTrueResidualIteration; numberOfCompletedIterations];
end
truePreconditionedRelativeResidualHistory = [truePreconditionedRelativeResidualHistory; finalTruePreconditionedRelativeResidual];
truePreconditionedResidualIteration = [truePreconditionedResidualIteration; numberOfCompletedIterations];

preconditionedResidualGap = abs(finalTruePreconditionedRelativeResidual - computedPreconditionedRelativeResidualHistory(numberOfCompletedIterations));

solverTimeHistory(numberOfCompletedIterations) = solverElapsed;
diagnosticTimeHistory(numberOfCompletedIterations) = diagnosticElapsed;
wallTimeHistory(numberOfCompletedIterations) = toc(wallTimer);

if displayProgress
    fprintf([ ...
        'Cycle complete at iteration %d: computed preconditioned ', ...
        'residual = %.3e, original true residual = %.3e\n'], ...
        numberOfCompletedIterations, ...
        computedPreconditionedRelativeResidualHistory( ...
            numberOfCompletedIterations), ...
        finalOriginalTrueResidual);
end


%% 8. Collect the cycle diagnostics

info.iterations = numberOfCompletedIterations;
info.converged = finalOriginalTrueResidual <= targetTolerance;
info.stop_reason = stopReason;
info.tolerance_mode = toleranceMode;
info.target_tolerance = targetTolerance;
info.fixed_compression_tolerance = fixedCompressionTolerance;
info.relaxation_error_budget = relaxationErrorBudget;
info.maximum_relaxed_tolerance = maximumRelaxedTolerance;
info.maximum_multilinear_rank = maximumRanks;
info.cycle_starting_preconditioned_residual_norm = beta;

info.true_relative_residual = ...
    originalTrueResidualHistory;
info.true_residual_iteration = originalTrueResidualIteration;
info.true_preconditioned_relative_residual = ...
    truePreconditionedRelativeResidualHistory;
info.true_preconditioned_residual_iteration = ...
    truePreconditionedResidualIteration;
info.computed_preconditioned_relative_residual = ...
    computedPreconditionedRelativeResidualHistory( ...
        1:numberOfCompletedIterations);
info.preconditioned_residual_gap = preconditionedResidualGap;
info.preconditioned_residual_gap_iteration = ...
    numberOfCompletedIterations;
info.eta = etaHistory(1:numberOfCompletedIterations);

info.solution_ranks = solutionRanks;
info.solution_rank_iteration = solutionRankIteration;
info.basis_ranks = ...
    basisRanks(1:numberOfCompletedIterations + 1, :);
info.basis_storage_history_entries = ...
    basisStorageHistory(1:numberOfCompletedIterations + 1);
info.peak_basis_storage_entries = ...
    max(info.basis_storage_history_entries);

info.H = H(1:numberOfCompletedIterations + 1, ...
    1:numberOfCompletedIterations);
info.y = y;

info.rank_cap_active_history = ...
    rankCapActiveHistory(1:numberOfCompletedIterations);
info.rank_cap_active = initialCapActive || ...
    any(info.rank_cap_active_history);
info.initial_maximum_local_recompression_error = initialLocalError;
info.operator_product_maximum_local_recompression_error = ...
    operatorProductMaximumLocalErrorHistory( ...
        1:numberOfCompletedIterations);
info.orthogonalisation_maximum_local_recompression_error = ...
    orthogonalisationMaximumLocalErrorHistory( ...
        1:numberOfCompletedIterations);
info.solution_maximum_local_recompression_error = ...
    solutionMaximumLocalErrorHistory( ...
        1:numberOfCompletedIterations);

info.stopped_for_basis_memory = basisMemoryLimitReached;
info.cycle_stopped_on_computed_residual = ...
    cycleStoppedOnComputedResidual;
info.orthogonalisation_passes = 1;

info.solver_time_history_sec = ...
    solverTimeHistory(1:numberOfCompletedIterations);
info.diagnostic_time_history_sec = ...
    diagnosticTimeHistory(1:numberOfCompletedIterations);
info.wall_time_history_sec = ...
    wallTimeHistory(1:numberOfCompletedIterations);
info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
wallElapsed = toc(wallTimer);
info.wall_time_sec = wallElapsed;
info.solution_assembly_count = 1;
info.original_true_residual_evaluation_count = ...
    numel(originalTrueResidualHistory);
info.timing = finalise_cycle_timing( ...
    phaseTiming, numberOfCompletedIterations, solverElapsed, ...
    diagnosticElapsed, wallElapsed, kernelTiming, kernelTimingByPhase);

end


function timingByPhase = empty_phase_kernel_timing()
%EMPTY_PHASE_KERNEL_TIMING Initialise caller-attributed leaf timings.

timingByPhase.operator_application = empty_tucker_kernel_timing();
timingByPhase.preconditioner_application = empty_tucker_kernel_timing();
timingByPhase.outer_round = empty_tucker_kernel_timing();
timingByPhase.orthogonalisation_subtraction_and_round = ...
    empty_tucker_kernel_timing();
timingByPhase.solution_assembly = empty_tucker_kernel_timing();

end


function totalTiming = add_operation_kernel_timing(totalTiming, operationInfo)
%ADD_OPERATION_KERNEL_TIMING Add leaf timings returned by one operation.

if isstruct(operationInfo) && isfield(operationInfo, 'kernel_timing')
    totalTiming = add_tucker_kernel_timing( ...
        totalTiming, operationInfo.kernel_timing);
end

end


function timing = finalise_cycle_timing( ...
    timing, completedIterations, solverElapsed, diagnosticElapsed, ...
    wallElapsed, kernelTiming, kernelTimingByPhase)
%FINALISE_CYCLE_TIMING Trim histories and close the timing accounts.

historyFields = { ...
    'operator_application_history_sec', ...
    'preconditioner_application_history_sec', ...
    'outer_round_history_sec', ...
    'orthogonalisation_inner_product_history_sec', ...
    'orthogonalisation_subtraction_round_history_sec', ...
    'small_least_squares_history_sec', ...
    'basis_normalisation_history_sec'};

for fieldIndex = 1:numel(historyFields)
    fieldName = historyFields{fieldIndex};
    timing.(fieldName) = timing.(fieldName)(1:completedIterations);
end

timing.accounted_arnoldi_iteration_history_sec = ...
    timing.operator_application_history_sec + ...
    timing.preconditioner_application_history_sec + ...
    timing.outer_round_history_sec + ...
    timing.orthogonalisation_inner_product_history_sec + ...
    timing.orthogonalisation_subtraction_round_history_sec + ...
    timing.small_least_squares_history_sec + ...
    timing.basis_normalisation_history_sec;

timing.accounted_solver_time_sec = ...
    timing.operator_application_time_sec + ...
    timing.preconditioner_application_time_sec + ...
    timing.outer_round_time_sec + ...
    timing.initial_residual_formation_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.orthogonalisation_subtraction_round_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solverElapsed - timing.accounted_solver_time_sec;

timing.accounted_diagnostic_time_sec = ...
    timing.original_true_residual_time_sec + ...
    timing.preconditioned_residual_diagnostic_time_sec;
timing.unclassified_diagnostic_time_sec = ...
    diagnosticElapsed - timing.accounted_diagnostic_time_sec;

timing.solver_time_sec = solverElapsed;
timing.diagnostic_time_sec = diagnosticElapsed;
timing.wall_time_sec = wallElapsed;
timing.kernel = kernelTiming;
timing.kernel_by_phase = kernelTimingByPhase;

end


function [rankCapActive, maximumLocalError] = summarise_operation_info(varargin)
%SUMMARISE_OPERATION_INFO Combine diagnostics from nested operations.

rankCapActive = false;
maximumLocalError = 0;

for argumentIndex = 1:nargin

    operationInfo = varargin{argumentIndex};

    if isempty(operationInfo)
        continue
    end

    if isfield(operationInfo, 'rank_cap_active')
        rankCapActive = rankCapActive || operationInfo.rank_cap_active;
    end

    scalarFields = { ...
        'relative_error_estimate', ...
        'maximum_local_recompression_error', ...
        'final_relative_error_estimate'};

    for fieldIndex = 1:numel(scalarFields)

        fieldName = scalarFields{fieldIndex};

        if isfield(operationInfo, fieldName)
            maximumLocalError = max( ...
                maximumLocalError, max(operationInfo.(fieldName), [], 'all'));
        end

    end

    vectorFields = { ...
        'addition_relative_error_estimate'};

    for fieldIndex = 1:numel(vectorFields)

        fieldName = vectorFields{fieldIndex};

        if isfield(operationInfo, fieldName) && ...
                ~isempty(operationInfo.(fieldName))
            maximumLocalError = max( ...
                maximumLocalError, ...
                max(operationInfo.(fieldName), [], 'all'));
        end

    end

end

end


function numberOfEntries = tucker_entries_from_rank_local(tensorDimensions, rankVector)
%TUCKER_ENTRIES_FROM_RANK_LOCAL Count Tucker core and factor entries.

numberOfEntries = prod(rankVector) + ...
    sum(tensorDimensions .* rankVector);

end


function info = empty_cycle_info( ...
    toleranceMode, maximumRanks, beta, originalResidual, ...
    preconditionedResidual, solverTime, diagnosticTime, wallTime, ...
    targetTolerance, fixedCompressionTolerance, ...
    relaxationErrorBudget, maximumRelaxedTolerance)
%EMPTY_CYCLE_INFO Return consistent fields for an accurate initial guess.

info.iterations = 0;
info.converged = true;
info.stop_reason = "initial_guess";
info.tolerance_mode = toleranceMode;
info.target_tolerance = targetTolerance;
info.fixed_compression_tolerance = fixedCompressionTolerance;
info.relaxation_error_budget = relaxationErrorBudget;
info.maximum_relaxed_tolerance = maximumRelaxedTolerance;
info.maximum_multilinear_rank = maximumRanks;
info.cycle_starting_preconditioned_residual_norm = beta;
info.true_relative_residual = originalResidual;
info.true_residual_iteration = 0;
info.true_preconditioned_relative_residual = preconditionedResidual;
info.true_preconditioned_residual_iteration = 0;
info.computed_preconditioned_relative_residual = zeros(0, 1);
info.preconditioned_residual_gap = zeros(0, 1);
info.preconditioned_residual_gap_iteration = zeros(0, 1);
info.eta = zeros(0, 1);
info.solution_ranks = zeros(0, numel(maximumRanks));
info.solution_rank_iteration = zeros(0, 1);
info.basis_ranks = zeros(0, numel(maximumRanks));
info.basis_storage_history_entries = zeros(0, 1);
info.peak_basis_storage_entries = 0;
info.H = [];
info.y = [];
info.rank_cap_active_history = false(0, 1);
info.rank_cap_active = false;
info.initial_maximum_local_recompression_error = 0;
info.operator_product_maximum_local_recompression_error = zeros(0, 1);
info.orthogonalisation_maximum_local_recompression_error = zeros(0, 1);
info.solution_maximum_local_recompression_error = zeros(0, 1);
info.stopped_for_basis_memory = false;
info.cycle_stopped_on_computed_residual = false;
info.orthogonalisation_passes = 1;
info.solver_time_history_sec = zeros(0, 1);
info.diagnostic_time_history_sec = zeros(0, 1);
info.wall_time_history_sec = zeros(0, 1);
info.solver_time_sec = solverTime;
info.diagnostic_time_sec = diagnosticTime;
info.wall_time_sec = wallTime;
info.solution_assembly_count = 0;
info.original_true_residual_evaluation_count = 1;

end
