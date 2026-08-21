function [x, info] = gmres_restarted_custom( ...
    Afun, b, x0, restartLength, maximumCycles, ...
    targetTolerance, displayProgress)
%GMRES_RESTARTED_CUSTOM Educational restarted full GMRES.
%
% One call to GMRES_UNRESTARTED_CUSTOM builds one local Krylov space. After
% at most restartLength steps, this wrapper keeps the newest approximation,
% discards the local basis, recomputes the residual, and starts a new cycle.
%
% Inputs:
%   Afun, b, x0
%       Matrix-free operator, right-hand side, and initial guess.
%
%   restartLength
%       Maximum number of Arnoldi steps in one cycle.
%
%   maximumCycles
%       Maximum number of restart cycles.
%
%   targetTolerance
%       Stopping tolerance for the true relative residual.
%
%   displayProgress (optional)
%       Set to true to print cycle summaries. The default is false.
%
% Outputs:
%   x
%       Final approximate solution.
%
%   info
%       Combined histories indexed by cumulative Arnoldi iteration.


%% 1. Check the inputs

if nargin < 7
    displayProgress = false;
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

if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end

b = b(:);
x = x0(:);
normB = norm(b);

if normB == 0
    error('The right-hand side b must be nonzero.');
end


%% 2. Evaluate the initial true residual

wrapperWallTimer = tic;
diagnosticTimer = tic;

initialTrueResidual = norm(b - Afun(x)) / normB;

outerDiagnosticTime = toc(diagnosticTimer);

globalIteration = 0;
solverElapsed = 0;
diagnosticElapsed = outerDiagnosticTime;

iterationHistory = zeros(0, 1);
cycleHistory = zeros(0, 1);
localIterationHistory = zeros(0, 1);
computedResidualHistory = zeros(0, 1);
trueResidualHistory = initialTrueResidual;
solverTimeHistory = zeros(0, 1);
diagnosticTimeHistory = outerDiagnosticTime;
wallTimeHistory = toc(wrapperWallTimer);

cycleIterations = zeros(maximumCycles, 1);
cycleBasisVectors = zeros(maximumCycles, 1);
breakdown = false;


%% 3. Run the restart cycles

for cycle = 1:maximumCycles

    if trueResidualHistory(end) <= targetTolerance
        break
    end

    cycleWallOffset = toc(wrapperWallTimer);
    solverOffset = solverElapsed;
    diagnosticOffset = diagnosticElapsed;

    [x, cycleInfo] = gmres_unrestarted_custom( ...
        Afun, b, x, restartLength, targetTolerance, false);

    numberOfLocalIterations = cycleInfo.iterations;
    cycleIterations(cycle) = numberOfLocalIterations;
    cycleBasisVectors(cycle) = cycleInfo.number_of_basis_vectors;

    localIterations = (1:numberOfLocalIterations).';
    globalIterations = globalIteration + localIterations;

    iterationHistory = [iterationHistory; globalIterations]; %#ok<AGROW>
    cycleHistory = [cycleHistory; ...
        cycle * ones(numberOfLocalIterations, 1)]; %#ok<AGROW>
    localIterationHistory = ...
        [localIterationHistory; localIterations]; %#ok<AGROW>

    computedResidualHistory = [computedResidualHistory; ...
        cycleInfo.computed_relative_residual]; %#ok<AGROW>
    trueResidualHistory = [trueResidualHistory; ...
        cycleInfo.true_relative_residual]; %#ok<AGROW>

    solverTimeHistory = [solverTimeHistory; ...
        solverOffset + cycleInfo.solver_time_history_sec]; %#ok<AGROW>
    diagnosticTimeHistory = [diagnosticTimeHistory; ...
        diagnosticOffset + ...
        cycleInfo.diagnostic_time_history_sec]; %#ok<AGROW>
    wallTimeHistory = [wallTimeHistory; ...
        cycleWallOffset + cycleInfo.wall_time_history_sec]; %#ok<AGROW>

    globalIteration = globalIteration + numberOfLocalIterations;
    solverElapsed = solverOffset + cycleInfo.solver_time_sec;
    diagnosticElapsed = ...
        diagnosticOffset + cycleInfo.diagnostic_time_sec;
    breakdown = breakdown || cycleInfo.breakdown;

    if displayProgress
        fprintf(['Full GMRES(%d), cycle %d: cumulative iteration %d, ', ...
                 'true residual = %.3e\n'], ...
                 restartLength, cycle, globalIteration, ...
                 trueResidualHistory(end));
    end

    if numberOfLocalIterations < restartLength || cycleInfo.converged
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
info.breakdown = breakdown;
info.initial_true_relative_residual = initialTrueResidual;
info.iteration = iterationHistory;
info.cycle = cycleHistory;
info.local_iteration = localIterationHistory;
info.computed_relative_residual = computedResidualHistory;
info.true_relative_residual = trueResidualHistory;
info.cycle_iterations = cycleIterations(1:numberOfCycles);
info.cycle_basis_vectors = cycleBasisVectors(1:numberOfCycles);
info.maximum_basis_vectors = max([0; info.cycle_basis_vectors]);
info.orthogonalisation_passes = 1;
info.solver_time_history_sec = solverTimeHistory;
info.diagnostic_time_history_sec = diagnosticTimeHistory;
info.wall_time_history_sec = wallTimeHistory;
info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
info.wall_time_sec = toc(wrapperWallTimer);

end
