function [U, info] = tucker_gmres_restarted( ...
    Afun, F, U0, restartLength, maximumCycles, ...
    targetTolerance, compressionTolerance, ...
    trueResidualFunction, displayProgress, ...
    maximumBasisStorageEntries, maximumMultilinearRank)
%TUCKER_GMRES_RESTARTED Solve with restarted Tucker-GMRES.
%
% The inner TUCKER_GMRES call builds one local Tucker-Arnoldi basis. At the
% end of a cycle, this wrapper keeps the newest approximation, discards the
% local basis, and begins again from the newly computed residual.
%
% Inputs:
%   Afun, F, U0
%       Tucker operator, right-hand side, and initial guess.
%
%   restartLength
%       Maximum number of Tucker-Arnoldi steps in one cycle.
%
%   maximumCycles
%       Maximum number of restart cycles.
%
%   targetTolerance
%       Stopping tolerance for the independently evaluated true residual.
%
%   compressionTolerance
%       Relative STHOSVD recompression tolerance.
%
%   trueResidualFunction
%       Function handle that evaluates the true relative residual.
%
%   displayProgress (optional)
%       Set to true to print one summary per cycle. The default is false.
%
%   maximumBasisStorageEntries (optional)
%       Safety limit for the active basis inside one restart cycle. The
%       default is Inf.
%
%   maximumMultilinearRank (optional)
%       Scalar or mode-wise hard Tucker rank cap. Use [] for no additional
%       cap.
%
% Outputs:
%   U
%       Final Tucker approximation.
%
%   info
%       Combined residual, timing, cycle, and Tucker-rank histories.


%% 1. Check the inputs

if nargin < 9
    displayProgress = false;
end

if nargin < 10
    maximumBasisStorageEntries = Inf;
end

if nargin < 11
    maximumMultilinearRank = [];
end

if restartLength < 1
    error('restartLength must be at least 1.');
end

if maximumCycles < 1
    error('maximumCycles must be at least 1.');
end

if targetTolerance <= 0
    error('targetTolerance must be positive.');
end

if compressionTolerance <= 0 || compressionTolerance >= 1
    error('compressionTolerance must be between 0 and 1.');
end

if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end

if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end

maximumRanks = normalise_tucker_rank_cap( ...
    maximumMultilinearRank, size(F));


%% 2. Evaluate the initial true residual

wrapperWallTimer = tic;
diagnosticTimer = tic;

U = U0;
initialTrueResidual = trueResidualFunction(U);

outerDiagnosticTime = toc(diagnosticTimer);

globalIteration = 0;
solverElapsed = 0;
diagnosticElapsed = outerDiagnosticTime;

iterationHistory = zeros(0, 1);
cycleHistory = zeros(0, 1);
localIterationHistory = zeros(0, 1);
computedResidualHistory = zeros(0, 1);
trueResidualHistory = initialTrueResidual;
trueResidualIteration = 0;
solverTimeHistory = zeros(0, 1);
diagnosticTimeHistory = outerDiagnosticTime;
wallTimeHistory = toc(wrapperWallTimer);

numberOfModes = ndims(F);
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
cycleStopReasons = strings(maximumCycles, 1);
basisMemoryLimitReached = false;
rankCapActive = false;
rankCapActiveHistory = zeros(0, 1);
maximumLocalRecompressionErrorHistory = zeros(0, 1);


%% 3. Run the restart cycles

for cycle = 1:maximumCycles

    if trueResidualHistory(end) <= targetTolerance
        break
    end

    cycleWallOffset = toc(wrapperWallTimer);
    solverOffset = solverElapsed;
    diagnosticOffset = diagnosticElapsed;

    [U, cycleInfo] = tucker_gmres( ...
        Afun, F, U, restartLength, targetTolerance, ...
        compressionTolerance, trueResidualFunction, false, ...
        maximumBasisStorageEntries, maximumRanks);

    numberOfLocalIterations = cycleInfo.iterations;
    cycleIterations(cycle) = numberOfLocalIterations;
    cycleSolverTimes(cycle) = cycleInfo.solver_time_sec;
    cycleDiagnosticTimes(cycle) = cycleInfo.diagnostic_time_sec;
    cyclePeakBasisStorageEntries(cycle) = ...
        cycleInfo.peak_basis_storage_entries;
    cycleStopReasons(cycle) = cycleInfo.stop_reason;

    localIterations = (1:numberOfLocalIterations).';
    globalIterations = globalIteration + localIterations;

    iterationHistory = [iterationHistory; globalIterations]; %#ok<AGROW>
    cycleHistory = [cycleHistory; ...
        cycle * ones(numberOfLocalIterations, 1)]; %#ok<AGROW>
    localIterationHistory = ...
        [localIterationHistory; localIterations]; %#ok<AGROW>

    computedResidualHistory = [computedResidualHistory; ...
        cycleInfo.computed_relative_residual]; %#ok<AGROW>

    % Only the cycle endpoint is new. The cycle's initial true residual is
    % already the previous endpoint in the combined history.
    trueResidualHistory = [trueResidualHistory; ...
        cycleInfo.true_relative_residual(end)]; %#ok<AGROW>
    trueResidualIteration = [trueResidualIteration; ...
        globalIteration + numberOfLocalIterations]; %#ok<AGROW>

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
    localBasisIndices = (0:numberOfBasisTensors-1).';

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
    maximumLocalRecompressionErrorHistory = ...
        [maximumLocalRecompressionErrorHistory; ...
         cycleInfo.maximum_local_recompression_error_history]; %#ok<AGROW>

    globalIteration = globalIteration + numberOfLocalIterations;
    solverElapsed = solverOffset + cycleInfo.solver_time_sec;
    diagnosticElapsed = ...
        diagnosticOffset + cycleInfo.diagnostic_time_sec;

    if displayProgress
        fprintf(['Tucker-GMRES(%d), cycle %d: cumulative iteration %d, ', ...
                 'true residual = %.3e\n'], ...
                 restartLength, cycle, globalIteration, ...
                 trueResidualHistory(end));
    end

    if cycleInfo.stopped_for_basis_memory
        basisMemoryLimitReached = true;
        break
    end

    if cycleInfo.converged
        break
    end

    if cycleInfo.stop_reason == "arnoldi_breakdown"
        break
    end

    if numberOfLocalIterations == 0
        break
    end

end


%% 4. Collect the combined information

numberOfCycles = find(cycleIterations > 0, 1, 'last');

if isempty(numberOfCycles)
    numberOfCycles = 0;
end

info.iterations = globalIteration;
info.number_of_cycles = numberOfCycles;
info.converged = trueResidualHistory(end) <= targetTolerance;
info.initial_true_relative_residual = initialTrueResidual;
info.iteration = iterationHistory;
info.cycle = cycleHistory;
info.local_iteration = localIterationHistory;
info.computed_relative_residual = computedResidualHistory;
info.true_relative_residual = trueResidualHistory;
info.true_residual_iteration = trueResidualIteration;
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
info.cycle_stop_reason = cycleStopReasons(1:numberOfCycles);
info.peak_basis_storage_entries = ...
    max([0; info.cycle_peak_basis_storage_entries]);
info.stopped_for_basis_memory = basisMemoryLimitReached;
info.orthogonalisation_passes = 1;
info.maximum_multilinear_rank = maximumRanks;
info.rank_cap_active = rankCapActive;
info.rank_cap_active_history = rankCapActiveHistory;
info.maximum_local_recompression_error_history = ...
    maximumLocalRecompressionErrorHistory;
info.solver_time_history_sec = solverTimeHistory;
info.diagnostic_time_history_sec = diagnosticTimeHistory;
info.wall_time_history_sec = wallTimeHistory;
info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
info.wall_time_sec = toc(wrapperWallTimer);
info.solution_assembly_count = numberOfCycles;

end
