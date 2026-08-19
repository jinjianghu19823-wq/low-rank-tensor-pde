function [U, info] = run_restarted_left_preconditioned_tucker_gmres( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    restartLength, maximumCycles, targetTolerance, ...
    fixedCompressionTolerance, toleranceMode, ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank)
%RUN_RESTARTED_LEFT_PRECONDITIONED_TUCKER_GMRES Run repeated local cycles.
%
% Each cycle recomputes the preconditioned starting residual, constructs a
% new local Tucker-Arnoldi basis, and discards that basis at the cycle end.
% In relaxed mode, eta starts again from targetTolerance in every cycle.


%% 1. Check the restart inputs

if restartLength < 1 || restartLength ~= floor(restartLength)
    error('restartLength must be a positive integer.');
end

if maximumCycles < 1 || maximumCycles ~= floor(maximumCycles)
    error('maximumCycles must be a positive integer.');
end

toleranceMode = lower(string(toleranceMode));

if toleranceMode ~= "fixed" && toleranceMode ~= "relaxed"
    error('toleranceMode must be "fixed" or "relaxed".');
end

maximumRanks = normalise_tucker_rank_cap( ...
    maximumMultilinearRank, size(F0));


%% 2. Prepare global histories

wrapperWallTimer = tic;
U = U0;

globalIteration = 0;
solverElapsed = 0;
diagnosticTimer = tic;
initialOriginalTrueResidual = originalTrueResidualFunction(U0);
diagnosticElapsed = toc(diagnosticTimer);

iterationHistory = zeros(0, 1);
cycleHistory = zeros(0, 1);
localIterationHistory = zeros(0, 1);

originalTrueResidualHistory = initialOriginalTrueResidual;
originalTrueResidualIteration = 0;
truePreconditionedRelativeResidualHistory = zeros(0, 1);
truePreconditionedResidualIteration = zeros(0, 1);
computedPreconditionedRelativeResidualHistory = zeros(0, 1);
preconditionedResidualGapHistory = zeros(0, 1);
preconditionedResidualGapIteration = zeros(0, 1);
etaHistory = zeros(0, 1);
solverTimeHistory = zeros(0, 1);
diagnosticTimeHistory = zeros(0, 1);
wallTimeHistory = zeros(0, 1);

numberOfModes = ndims(F0);
solutionRanks = zeros(0, numberOfModes);
solutionRankIteration = zeros(0, 1);
solutionRankCycle = zeros(0, 1);

basisRanks = zeros(0, numberOfModes);
basisRankGlobalIteration = zeros(0, 1);
basisRankLocalIndex = zeros(0, 1);
basisRankCycle = zeros(0, 1);

cycleIterations = zeros(maximumCycles, 1);
cycleSolverTimes = zeros(maximumCycles, 1);
cycleDiagnosticTimes = zeros(maximumCycles, 1);
cyclePeakBasisStorageEntries = zeros(maximumCycles, 1);
cycleStartingPreconditionedResidualNorms = ...
    zeros(maximumCycles, 1);
cycleStopReasons = strings(maximumCycles, 1);

rankCapActiveHistory = false(0, 1);
operatorProductErrorHistory = zeros(0, 1);
orthogonalisationErrorHistory = zeros(0, 1);
solutionErrorHistory = zeros(0, 1);

basisMemoryLimitReached = false;
rankCapActive = false;
stopReason = "cycle_limit";
cycleWallOffset = 0;


%% 3. Run the restart cycles

for cycle = 1:maximumCycles

    currentOriginalTrueResidual = originalTrueResidualHistory(end);

    if currentOriginalTrueResidual <= targetTolerance
        stopReason = "original_true_residual";
        break
    end

    cycleWallOffset = toc(wrapperWallTimer);
    solverOffset = solverElapsed;
    diagnosticOffset = diagnosticElapsed;

    [Unew, cycleInfo] = run_left_preconditioned_tucker_gmres_cycle( ...
        originalOperatorFunction, preconditionerFunction, F0, U, ...
        restartLength, targetTolerance, fixedCompressionTolerance, ...
        toleranceMode, originalTrueResidualFunction, ...
        preconditionedResidualNormFunction, false, ...
        maximumBasisStorageEntries, maximumRanks);

    numberOfLocalIterations = cycleInfo.iterations;
    cycleIterations(cycle) = numberOfLocalIterations;
    cycleSolverTimes(cycle) = cycleInfo.solver_time_sec;
    cycleDiagnosticTimes(cycle) = cycleInfo.diagnostic_time_sec;
    cyclePeakBasisStorageEntries(cycle) = ...
        cycleInfo.peak_basis_storage_entries;
    cycleStartingPreconditionedResidualNorms(cycle) = ...
        cycleInfo.cycle_starting_preconditioned_residual_norm;
    cycleStopReasons(cycle) = cycleInfo.stop_reason;

    localIterations = (1:numberOfLocalIterations).';
    globalIterations = globalIteration + localIterations;

    iterationHistory = [iterationHistory; globalIterations]; %#ok<AGROW>
    cycleHistory = [cycleHistory; ...
        cycle * ones(numberOfLocalIterations, 1)]; %#ok<AGROW>
    localIterationHistory = [localIterationHistory; ...
        localIterations]; %#ok<AGROW>

    originalTrueResidualHistory = [originalTrueResidualHistory; ...
        cycleInfo.true_relative_residual(end)]; %#ok<AGROW>
    originalTrueResidualIteration = [originalTrueResidualIteration; ...
        globalIteration + numberOfLocalIterations]; %#ok<AGROW>
    truePreconditionedRelativeResidualHistory = ...
        [truePreconditionedRelativeResidualHistory; ...
         cycleInfo.true_preconditioned_relative_residual(end)]; %#ok<AGROW>
    truePreconditionedResidualIteration = ...
        [truePreconditionedResidualIteration; ...
         globalIteration + numberOfLocalIterations]; %#ok<AGROW>
    computedPreconditionedRelativeResidualHistory = ...
        [computedPreconditionedRelativeResidualHistory; ...
         cycleInfo.computed_preconditioned_relative_residual]; %#ok<AGROW>
    preconditionedResidualGapHistory = ...
        [preconditionedResidualGapHistory; ...
         cycleInfo.preconditioned_residual_gap]; %#ok<AGROW>
    preconditionedResidualGapIteration = ...
        [preconditionedResidualGapIteration; ...
         globalIteration + ...
         cycleInfo.preconditioned_residual_gap_iteration]; %#ok<AGROW>
    etaHistory = [etaHistory; cycleInfo.eta]; %#ok<AGROW>
    solverTimeHistory = [solverTimeHistory; ...
        solverOffset + cycleInfo.solver_time_history_sec]; %#ok<AGROW>
    diagnosticTimeHistory = [diagnosticTimeHistory; ...
        diagnosticOffset + ...
        cycleInfo.diagnostic_time_history_sec]; %#ok<AGROW>
    wallTimeHistory = [wallTimeHistory; ...
        cycleWallOffset + cycleInfo.wall_time_history_sec]; %#ok<AGROW>

    solutionRanks = [solutionRanks; ...
        cycleInfo.solution_ranks]; %#ok<AGROW>
    solutionRankIteration = [solutionRankIteration; ...
        globalIteration + cycleInfo.solution_rank_iteration]; %#ok<AGROW>
    solutionRankCycle = [solutionRankCycle; ...
        cycle * ones(size(cycleInfo.solution_ranks, 1), 1)]; %#ok<AGROW>

    numberOfBasisTensors = size(cycleInfo.basis_ranks, 1);
    localBasisIndices = (0:numberOfBasisTensors - 1).';

    basisRanks = [basisRanks; cycleInfo.basis_ranks]; %#ok<AGROW>
    basisRankGlobalIteration = [basisRankGlobalIteration; ...
        globalIteration + localBasisIndices]; %#ok<AGROW>
    basisRankLocalIndex = [basisRankLocalIndex; ...
        localBasisIndices]; %#ok<AGROW>
    basisRankCycle = [basisRankCycle; ...
        cycle * ones(numberOfBasisTensors, 1)]; %#ok<AGROW>

    rankCapActive = rankCapActive || cycleInfo.rank_cap_active;
    rankCapActiveHistory = [rankCapActiveHistory; ...
        cycleInfo.rank_cap_active_history]; %#ok<AGROW>
    operatorProductErrorHistory = [operatorProductErrorHistory; ...
        cycleInfo.operator_product_maximum_local_recompression_error]; ...
        %#ok<AGROW>
    orthogonalisationErrorHistory = ...
        [orthogonalisationErrorHistory; ...
         cycleInfo.orthogonalisation_maximum_local_recompression_error]; ...
        %#ok<AGROW>
    solutionErrorHistory = [solutionErrorHistory; ...
        cycleInfo.solution_maximum_local_recompression_error]; %#ok<AGROW>

    globalIteration = globalIteration + numberOfLocalIterations;
    solverElapsed = solverOffset + cycleInfo.solver_time_sec;
    diagnosticElapsed = diagnosticOffset + cycleInfo.diagnostic_time_sec;
    U = Unew;

    if displayProgress
        fprintf([ ...
            'Restarted %s Tucker-GMRES(%d), cycle %d: ', ...
            'cumulative iteration %d, original true residual = %.3e\n'], ...
            char(toleranceMode), restartLength, cycle, ...
            globalIteration, originalTrueResidualHistory(end));
    end

    if cycleInfo.stopped_for_basis_memory
        basisMemoryLimitReached = true;
        stopReason = "basis_memory";
        break
    end

    if cycleInfo.converged
        stopReason = "original_true_residual";
        break
    end

    if cycleInfo.stop_reason == "arnoldi_breakdown"
        stopReason = "arnoldi_breakdown";
        break
    end

    % A zero-step cycle would repeat the same approximation forever.
    if numberOfLocalIterations == 0
        stopReason = "no_cycle_progress";
        break
    end

    stopReason = cycleInfo.stop_reason;

end


%% 4. Collect the restarted diagnostics

numberOfCycles = find(cycleIterations > 0, 1, 'last');

if isempty(numberOfCycles)
    numberOfCycles = 0;
end

if numberOfCycles == maximumCycles && ...
        originalTrueResidualHistory(end) > targetTolerance && ...
        ~basisMemoryLimitReached
    stopReason = "cycle_limit";
end

info.iterations = globalIteration;
info.number_of_cycles = numberOfCycles;
info.converged = originalTrueResidualHistory(end) <= targetTolerance;
info.stop_reason = stopReason;
info.tolerance_mode = toleranceMode;
info.restart_length = restartLength;
info.maximum_cycles = maximumCycles;
info.target_tolerance = targetTolerance;
info.fixed_compression_tolerance = fixedCompressionTolerance;
info.maximum_multilinear_rank = maximumRanks;

info.iteration = iterationHistory;
info.cycle = cycleHistory;
info.local_iteration = localIterationHistory;
info.true_relative_residual = originalTrueResidualHistory;
info.true_residual_iteration = originalTrueResidualIteration;
info.true_preconditioned_relative_residual = ...
    truePreconditionedRelativeResidualHistory;
info.true_preconditioned_residual_iteration = ...
    truePreconditionedResidualIteration;
info.computed_preconditioned_relative_residual = ...
    computedPreconditionedRelativeResidualHistory;
info.preconditioned_residual_gap = preconditionedResidualGapHistory;
info.preconditioned_residual_gap_iteration = ...
    preconditionedResidualGapIteration;
info.eta = etaHistory;
info.solver_time_history_sec = solverTimeHistory;
info.diagnostic_time_history_sec = diagnosticTimeHistory;
info.wall_time_history_sec = wallTimeHistory;

info.solution_ranks = solutionRanks;
info.solution_rank_iteration = solutionRankIteration;
info.solution_rank_cycle = solutionRankCycle;
info.basis_ranks = basisRanks;
info.basis_rank_global_iteration = basisRankGlobalIteration;
info.basis_rank_local_index = basisRankLocalIndex;
info.basis_rank_cycle = basisRankCycle;

info.cycle_iterations = cycleIterations(1:numberOfCycles);
info.cycle_solver_time_sec = cycleSolverTimes(1:numberOfCycles);
info.cycle_diagnostic_time_sec = ...
    cycleDiagnosticTimes(1:numberOfCycles);
info.cycle_peak_basis_storage_entries = ...
    cyclePeakBasisStorageEntries(1:numberOfCycles);
info.cycle_starting_preconditioned_residual_norm = ...
    cycleStartingPreconditionedResidualNorms(1:numberOfCycles);
info.cycle_stop_reason = cycleStopReasons(1:numberOfCycles);
info.peak_basis_storage_entries = ...
    max([0; info.cycle_peak_basis_storage_entries]);

info.rank_cap_active = rankCapActive;
info.rank_cap_active_history = rankCapActiveHistory;
info.operator_product_maximum_local_recompression_error = ...
    operatorProductErrorHistory;
info.orthogonalisation_maximum_local_recompression_error = ...
    orthogonalisationErrorHistory;
info.solution_maximum_local_recompression_error = ...
    solutionErrorHistory;
info.stopped_for_basis_memory = basisMemoryLimitReached;
info.orthogonalisation_passes = 1;

info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
info.wall_time_sec = toc(wrapperWallTimer);
info.last_cycle_wall_offset_sec = cycleWallOffset;
info.solution_assembly_count = numberOfCycles;

end
