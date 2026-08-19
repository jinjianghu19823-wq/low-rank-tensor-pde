function [U, info] = tucker_gmres(Afun, F, U0, maximumIteration, targetTolerance, compressionTolerance, trueResidualFunction, displayProgress, maximumBasisStorageEntries, maximumMultilinearRank)
%TUCKER_GMRES Solve a tensor linear system using Tucker-GMRES.
%
% Inputs:
%   Afun
%       Applies the linear operator to a Tucker tensor.
%
%   F
%       Right-hand-side tensor, stored as a ttensor.
%
%   U0
%       Initial guess, stored as a ttensor.
%
%   maximumIteration
%       Maximum number of Krylov iterations.
%
%   targetTolerance
%       Required true relative residual.
%
%   compressionTolerance
%       Relative tolerance used when recompressing Tucker tensors.
%
%   trueResidualFunction
%       Calculates the true relative residual of an approximate solution.
%
%   displayProgress (optional)
%       Set this to true to print one line per iteration. The default is
%       true so that existing teaching examples keep their output.
%
%   maximumBasisStorageEntries (optional)
%       Safety limit for the active Tucker-Arnoldi basis, counted as core
%       plus factor entries. The default is Inf. This is a memory stop, not
%       a target Tucker rank.
%
%   maximumMultilinearRank (optional)
%       Scalar or mode-wise hard Tucker rank cap used in every residual,
%       orthogonalisation, and solution recompression. Use [] for no
%       additional cap.
%
% Outputs:
%   U
%       Final approximate solution, stored as a ttensor.
%
%   info
%       Structure containing convergence information. The small
%       Hessenberg residual is stored at every Arnoldi step. The Tucker
%       solution and independently evaluated true residual are formed only
%       once, after the Arnoldi loop. Core solver time and diagnostic time
%       are recorded separately.
%
% Thesis notation (Section 5.5, fixed-tolerance Tucker-GMRES):
%   Afun, F                    <->  \mathcal A, \mathcal B
%   U0, U                      <->  \mathcal X_0, \mathcal X_j
%   maximumIteration           <->  \ell_max
%   targetTolerance            <->  \varepsilon
%   compressionTolerance       <->  \eta
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
%   R0                         <->  \widetilde{\mathcal R}_0
%   beta                       <->  ||\widetilde{\mathcal R}_0||_F
%   V{j}, W                    <->  \mathcal V_j, \mathcal W
%   H(1:j+1,1:j), y            <->  \bar H_j, y_j
%   smallResidual              <->  \beta e_1-\bar H_j y_j
%   trueResidualFunction(U)    <->  ||\mathcal B-\mathcal A(U)||_F/||\mathcal B||_F


%% 1. Check some basic inputs

if nargin < 8
    displayProgress = true;
end

if nargin < 9
    maximumBasisStorageEntries = Inf;
end

if nargin < 10
    maximumMultilinearRank = [];
end

if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end

if maximumIteration < 1
    error('maximumIteration must be at least 1.');
end

if targetTolerance <= 0
    error('targetTolerance must be positive.');
end

if compressionTolerance <= 0 || compressionTolerance >= 1
    error('compressionTolerance must be between 0 and 1.');
end

if maximumBasisStorageEntries <= 0
    error('maximumBasisStorageEntries must be positive.');
end

normF = norm(F);
maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, size(F));

if normF == 0
    error('The right-hand side F must be nonzero.');
end

% These timers distinguish the reported core solver kernels from the
% independently evaluated true residual used for stopping after every
% iteration. Therefore solver_time_sec is not complete end-to-end time.
wallTimer = tic;
solverElapsed = 0;
diagnosticElapsed = 0;


%% 2. Calculate the initial residual

solverTimer = tic;

% Apply the operator to the initial guess.
AU0 = Afun(U0);

% R0 = F - A(U0), followed by Tucker recompression.
[R0, initialRoundInfo] = tucker_axpby_round(F, 1, AU0, -1, compressionTolerance, maximumRanks);

% beta is the Frobenius norm of the compressed initial residual.
beta = norm(R0);

solverElapsed = solverElapsed + toc(solverTimer);

% Calculate the true initial residual independently.
diagnosticTimer = tic;
initialTrueResidual = trueResidualFunction(U0);
diagnosticElapsed = diagnosticElapsed + toc(diagnosticTimer);

% If the initial guess is already accurate enough, stop here.
if initialTrueResidual <= targetTolerance

    U = U0;

    info.iterations = 0;
    info.converged = true;
    info.true_relative_residual = initialTrueResidual;
    info.true_residual_iteration = 0;
    info.computed_relative_residual = zeros(0, 1);
    info.solution_ranks = [];
    info.solution_rank_iteration = zeros(0, 1);
    info.basis_ranks = [];
    info.H = [];
    info.y = [];
    info.solver_time_sec = solverElapsed;
    info.diagnostic_time_sec = diagnosticElapsed;
    info.wall_time_sec = toc(wallTimer);
    info.solver_time_history_sec = [];
    info.diagnostic_time_history_sec = [];
    info.wall_time_history_sec = [];
    info.basis_storage_history_entries = [];
    info.peak_basis_storage_entries = 0;
    info.stopped_for_basis_memory = false;
    info.orthogonalisation_passes = 1;
    info.maximum_multilinear_rank = maximumRanks;
    info.rank_cap_active = initialRoundInfo.rank_cap_active;
    info.rank_cap_active_history = false(0, 1);
    info.initial_recompression_relative_error = ...
        initialRoundInfo.relative_error_estimate;
    info.maximum_local_recompression_error_history = zeros(0, 1);
    info.stop_reason = "initial_guess";
    info.cycle_stopped_on_computed_residual = false;
    info.solution_assembly_count = 0;
    info.true_residual_evaluation_count = 1;

    return
end

% If compression has accidentally removed the whole initial residual,
% the Krylov basis cannot be started.
if beta <= 1e-14 * normF
    error(['The compressed initial residual is nearly zero. ', 'Use a smaller compression tolerance.']);
end


%% 3. Prepare storage for the Arnoldi iteration

% V is a cell array because each basis tensor may have different ranks.
V = cell(maximumIteration + 1, 1);

% H is the small upper-Hessenberg matrix.
H = zeros(maximumIteration + 1, maximumIteration);

% The reduced residual is available at every Arnoldi step. The true
% residual is evaluated only for the initial guess and the final assembled
% approximation.
trueResidualHistory = initialTrueResidual;
trueResidualIteration = 0;
computedResidualHistory = zeros(maximumIteration, 1);

% Store cumulative timings at the end of each iteration.
solverTimeHistory = zeros(maximumIteration, 1);
diagnosticTimeHistory = zeros(maximumIteration, 1);
wallTimeHistory = zeros(maximumIteration, 1);

% The solution is assembled once after Arnoldi. Basis ranks are still
% recorded at every Arnoldi step.
numberOfModes = ndims(F);

basisRanks = zeros(maximumIteration + 1, numberOfModes);
basisStorageHistory = zeros(maximumIteration + 1, 1);
rankCapActiveHistory = false(maximumIteration, 1);
maximumLocalRecompressionErrorHistory = zeros(maximumIteration, 1);

% Normalise the initial residual to obtain the first basis tensor.
V{1} = (1 / beta) * R0;

% Store the rank of the first basis tensor. At iteration j, GMRES has
% generated j+1 Arnoldi tensors, so the final storage estimate should use
% all rows of basisRanks returned below.
basisRanks(1,:) = size(V{1}.core);
basisStorageHistory(1) = tucker_entries_from_rank(size(F), basisRanks(1,:));

if basisStorageHistory(1) > maximumBasisStorageEntries
    error('The initial Tucker basis exceeds the storage safety limit.');
end

basisMemoryLimitReached = false;
cycleStoppedOnComputedResidual = false;
stopReason = "iteration_limit";


%% 4. Start the Tucker-Arnoldi iteration

for j = 1:maximumIteration

    solverTimer = tic;

    % Apply the tensor operator:
    %
    %     W = A(V_j).
    %
    % Afun should return a recompressed ttensor.
    W = Afun(V{j});


    %% 5. Orthogonalise W with one modified Gram--Schmidt pass

    for i = 1:j

        % Calculate the Frobenius inner product
        %
        %     correction = <V_i, W>.
        correction = innerprod(V{i}, W);
        H(i,j) = correction;

        % Perform
        %
        %     W = W - correction * V_i,
        %
        % and recompress the result.
        [W, subtractionInfo] = tucker_axpby_round(W, 1, V{i}, -correction, compressionTolerance, maximumRanks);

        rankCapActiveHistory(j) = rankCapActiveHistory(j) || subtractionInfo.rank_cap_active;
        maximumLocalRecompressionErrorHistory(j) = max(maximumLocalRecompressionErrorHistory(j), subtractionInfo.relative_error_estimate);

    end


    %% 6. Calculate the next Arnoldi coefficient

    H(j+1,j) = norm(W);

    % W becomes the next basis tensor after normalisation. Scaling does not
    % change its Tucker ranks, so its ranks can already be recorded here.
    basisRanks(j+1,:) = size(W.core);
    basisStorageHistory(j+1) = basisStorageHistory(j) + tucker_entries_from_rank(size(F), basisRanks(j+1,:));

    if basisStorageHistory(j+1) > maximumBasisStorageEntries
        basisMemoryLimitReached = true;
    end


    %% 7. Solve the small least-squares problem

    % Construct beta * e_1.
    smallRightHandSide = zeros(j+1, 1);
    smallRightHandSide(1) = beta;

    % MATLAB backslash solves the rectangular least-squares problem
    %
    %     min ||beta*e_1 - H*y||_2.
    y = H(1:j+1, 1:j) \ smallRightHandSide;


    %% 8. Calculate the reduced residual

    % This is the residual from the small Hessenberg problem. It controls
    % the inner Arnoldi loop without constructing the Tucker solution.
    smallResidual = smallRightHandSide - H(1:j+1, 1:j) * y;

    computedResidualHistory(j) = norm(smallResidual) / normF;

    solverElapsed = solverElapsed + toc(solverTimer);

    solverTimeHistory(j) = solverElapsed;
    diagnosticTimeHistory(j) = diagnosticElapsed;
    wallTimeHistory(j) = toc(wallTimer);


    %% 9. Display the Arnoldi progress

    if displayProgress
        fprintf('Iteration %3d: computed residual = %.3e\n', j, computedResidualHistory(j));
    end


    %% 10. Stop the Arnoldi loop when the reduced residual is small

    if computedResidualHistory(j) <= targetTolerance
        cycleStoppedOnComputedResidual = true;
        stopReason = "computed_residual";
        break
    end


    %% 11. Check the active-basis storage safety limit

    if basisMemoryLimitReached
        stopReason = "basis_memory";
        if displayProgress
            fprintf(['Tucker Arnoldi stopped because the active basis ', 'exceeded the storage safety limit.\n']);
        end
        break
    end


    %% 12. Check for Arnoldi breakdown

    if H(j+1,j) <= 1e-14
        stopReason = "arnoldi_breakdown";
        if displayProgress
            fprintf(['Arnoldi stopped because the next basis norm is ', 'nearly zero.\n']);
        end
        break
    end


    %% 13. Normalise W to obtain the next basis tensor

    solverTimer = tic;
    V{j+1} = (1 / H(j+1,j)) * W;
    solverElapsed = solverElapsed + toc(solverTimer);

end


%% 14. Assemble the Tucker solution once

solverTimer = tic;
U = U0;

for i = 1:j

    [U, solutionAdditionInfo] = tucker_axpby_round(U, 1, V{i}, y(i), compressionTolerance, maximumRanks);

    rankCapActiveHistory(j) = rankCapActiveHistory(j) || solutionAdditionInfo.rank_cap_active;
    maximumLocalRecompressionErrorHistory(j) = max(maximumLocalRecompressionErrorHistory(j), solutionAdditionInfo.relative_error_estimate);

end

solutionRanks = size(U.core);
solutionRankIteration = j;
solverElapsed = solverElapsed + toc(solverTimer);


%% 15. Evaluate the final true residual independently

diagnosticTimer = tic;
finalTrueResidual = trueResidualFunction(U);
diagnosticElapsed = diagnosticElapsed + toc(diagnosticTimer);

trueResidualHistory = [trueResidualHistory; finalTrueResidual];
trueResidualIteration = [trueResidualIteration; j];

% The final timing sample includes the delayed solution assembly and the
% cycle-end true-residual evaluation.
solverTimeHistory(j) = solverElapsed;
diagnosticTimeHistory(j) = diagnosticElapsed;
wallTimeHistory(j) = toc(wallTimer);

if displayProgress
    fprintf(['Cycle complete at iteration %d: computed residual = %.3e, ', 'true residual = %.3e\n'], j, computedResidualHistory(j), finalTrueResidual);
end


%% 16. Collect the final diagnostic information

info.iterations = j;

info.converged = finalTrueResidual <= targetTolerance;

info.true_relative_residual = trueResidualHistory;

info.true_residual_iteration = trueResidualIteration;

info.computed_relative_residual = computedResidualHistory(1:j);

info.solution_ranks = solutionRanks;

info.solution_rank_iteration = solutionRankIteration;

info.basis_ranks = basisRanks(1:j+1,:);

info.H = H(1:j+1, 1:j);

info.y = y;

info.solver_time_sec = solverElapsed;

info.diagnostic_time_sec = diagnosticElapsed;

info.wall_time_sec = toc(wallTimer);

info.solver_time_history_sec = solverTimeHistory(1:j);

info.diagnostic_time_history_sec = diagnosticTimeHistory(1:j);

info.wall_time_history_sec = wallTimeHistory(1:j);

info.basis_storage_history_entries = basisStorageHistory(1:j+1);

info.peak_basis_storage_entries = ...
    max(info.basis_storage_history_entries);

info.stopped_for_basis_memory = basisMemoryLimitReached;

info.orthogonalisation_passes = 1;

info.maximum_multilinear_rank = maximumRanks;

info.rank_cap_active_history = rankCapActiveHistory(1:j);

info.rank_cap_active = initialRoundInfo.rank_cap_active || any(info.rank_cap_active_history);

info.initial_recompression_relative_error = initialRoundInfo.relative_error_estimate;

info.maximum_local_recompression_error_history = maximumLocalRecompressionErrorHistory(1:j);

info.stop_reason = stopReason;

info.cycle_stopped_on_computed_residual = cycleStoppedOnComputedResidual;

info.solution_assembly_count = 1;

info.true_residual_evaluation_count = 2;

end


function numberOfEntries = tucker_entries_from_rank(tensorDimensions, rankVector)
%TUCKER_ENTRIES_FROM_RANK Count core and factor entries.

numberOfEntries = prod(rankVector) + sum(tensorDimensions .* rankVector);

end
