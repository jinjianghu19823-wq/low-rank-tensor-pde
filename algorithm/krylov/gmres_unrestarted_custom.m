function [x, info] = gmres_unrestarted_custom( ...
    Afun, b, x0, maximumIteration, targetTolerance, displayProgress)
%GMRES_UNRESTARTED_CUSTOM Educational implementation of full GMRES.
%
% This routine follows the same mathematical steps as the Tucker-GMRES
% routine, but every Krylov vector is stored as a full MATLAB vector. It is
% intentionally written with descriptive variables and explicit sections.
%
% Inputs:
%   Afun
%       Function handle that calculates A*x without requiring A explicitly.
%
%   b
%       Right-hand-side vector.
%
%   x0
%       Initial guess.
%
%   maximumIteration
%       Maximum number of Arnoldi steps in this cycle.
%
%   targetTolerance
%       Stopping tolerance for the independently evaluated true residual.
%
%   displayProgress (optional)
%       Set to true to print one line per iteration. The default is false.
%
% Outputs:
%   x
%       Final approximate solution.
%
%   info
%       Residual histories, timings, iteration count, and breakdown status.
%       solver_time_sec is core solver time excluding the independent
%       true-residual calculations used for stopping. It is not complete
%       end-to-end time.


%% 1. Check the inputs

if nargin < 6
    displayProgress = false;
end

if maximumIteration < 1
    error('maximumIteration must be at least 1.');
end

if targetTolerance <= 0
    error('targetTolerance must be positive.');
end

if ~islogical(displayProgress) || ~isscalar(displayProgress)
    error('displayProgress must be one logical value.');
end

b = b(:);
x0 = x0(:);

if length(b) ~= length(x0)
    error('b and x0 must have the same number of entries.');
end

normB = norm(b);

if normB == 0
    error('The right-hand side b must be nonzero.');
end


%% 2. Calculate the initial residual

wallTimer = tic;
solverElapsed = 0;
diagnosticElapsed = 0;

solverTimer = tic;

initialResidual = b - Afun(x0);
beta = norm(initialResidual);
initialTrueResidual = beta / normB;

solverElapsed = solverElapsed + toc(solverTimer);

if initialTrueResidual <= targetTolerance
    x = x0;
    info = empty_information(initialTrueResidual, solverElapsed, ...
        diagnosticElapsed, toc(wallTimer));
    return
end

if beta <= 1e-14 * normB
    error('The initial residual is too small to start Arnoldi iteration.');
end


%% 3. Prepare the Arnoldi basis and histories

numberOfUnknowns = length(b);

% Column j of V is the j-th orthonormal Arnoldi vector.
V = zeros(numberOfUnknowns, maximumIteration + 1);
V(:,1) = initialResidual / beta;

% H is the small upper-Hessenberg matrix from the Arnoldi relation.
H = zeros(maximumIteration + 1, maximumIteration);

computedResidualHistory = zeros(maximumIteration, 1);
trueResidualHistory = zeros(maximumIteration, 1);
solverTimeHistory = zeros(maximumIteration, 1);
diagnosticTimeHistory = zeros(maximumIteration, 1);
wallTimeHistory = zeros(maximumIteration, 1);

x = x0;
y = [];
breakdown = false;


%% 4. Run the Arnoldi and GMRES iterations

for j = 1:maximumIteration

    solverTimer = tic;

    % Apply the matrix-free operator to the newest basis vector.
    w = Afun(V(:,j));


    %% 4a. Orthogonalise with one modified Gram--Schmidt pass

    for i = 1:j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end

    H(j+1,j) = norm(w);

    if H(j+1,j) > 1e-14 && j < maximumIteration
        V(:,j+1) = w / H(j+1,j);
    end


    %% 4b. Solve the small GMRES least-squares problem

    smallRightHandSide = zeros(j+1, 1);
    smallRightHandSide(1) = beta;

    currentHessenberg = H(1:j+1, 1:j);
    y = currentHessenberg \ smallRightHandSide;

    smallResidual = ...
        smallRightHandSide - currentHessenberg * y;

    computedResidualHistory(j) = norm(smallResidual) / normB;


    %% 4c. Construct the current approximate solution

    x = x0 + V(:,1:j) * y;

    solverElapsed = solverElapsed + toc(solverTimer);


    %% 4d. Evaluate the true residual independently

    diagnosticTimer = tic;

    trueResidualHistory(j) = norm(b - Afun(x)) / normB;

    diagnosticElapsed = ...
        diagnosticElapsed + toc(diagnosticTimer);

    solverTimeHistory(j) = solverElapsed;
    diagnosticTimeHistory(j) = diagnosticElapsed;
    wallTimeHistory(j) = toc(wallTimer);

    if displayProgress
        fprintf(['Full GMRES iteration %3d: computed residual = %.3e, ', ...
                 'true residual = %.3e\n'], ...
                 j, computedResidualHistory(j), trueResidualHistory(j));
    end


    %% 4e. Check stopping conditions

    if trueResidualHistory(j) <= targetTolerance
        break
    end

    if H(j+1,j) <= 1e-14
        breakdown = true;
        if displayProgress
            fprintf('Full Arnoldi stopped because the next norm is tiny.\n');
        end
        break
    end

end


%% 5. Collect the final information

numberOfIterations = j;

info.iterations = numberOfIterations;
info.iter = (1:numberOfIterations).';
info.converged = ...
    trueResidualHistory(numberOfIterations) <= targetTolerance;
info.breakdown = breakdown;
info.initial_true_relative_residual = initialTrueResidual;
info.computed_relative_residual = ...
    computedResidualHistory(1:numberOfIterations);
info.true_relative_residual = ...
    trueResidualHistory(1:numberOfIterations);

% Keep the historical names used by earlier learning material.
info.relres = info.computed_relative_residual;
info.ls_relres = info.computed_relative_residual;
info.true_relres = info.true_relative_residual;

info.H = H(1:numberOfIterations+1, 1:numberOfIterations);
info.y = y;
info.number_of_basis_vectors = numberOfIterations + 1;
info.orthogonalisation_passes = 1;

info.solver_time_history_sec = ...
    solverTimeHistory(1:numberOfIterations);
info.diagnostic_time_history_sec = ...
    diagnosticTimeHistory(1:numberOfIterations);
info.wall_time_history_sec = ...
    wallTimeHistory(1:numberOfIterations);

info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
info.wall_time_sec = toc(wallTimer);

% Keep the earlier timing field names for compatibility.
info.time = info.solver_time_history_sec;
info.final_true_relres = ...
    info.true_relative_residual(end);
info.final_solver_time_sec = info.solver_time_sec;
info.final_diagnostic_time_sec = info.diagnostic_time_sec;
info.final_wall_time_sec = info.wall_time_sec;

end


function info = empty_information( ...
    initialTrueResidual, solverElapsed, diagnosticElapsed, wallElapsed)
%EMPTY_INFORMATION Create a consistent output for a zero-iteration solve.

info.iterations = 0;
info.iter = zeros(0, 1);
info.converged = true;
info.breakdown = false;
info.initial_true_relative_residual = initialTrueResidual;
info.computed_relative_residual = zeros(0, 1);
info.true_relative_residual = zeros(0, 1);
info.relres = zeros(0, 1);
info.ls_relres = zeros(0, 1);
info.true_relres = zeros(0, 1);
info.H = zeros(1, 0);
info.y = zeros(0, 1);
info.number_of_basis_vectors = 0;
info.orthogonalisation_passes = 1;
info.solver_time_history_sec = zeros(0, 1);
info.diagnostic_time_history_sec = zeros(0, 1);
info.wall_time_history_sec = zeros(0, 1);
info.solver_time_sec = solverElapsed;
info.diagnostic_time_sec = diagnosticElapsed;
info.wall_time_sec = wallElapsed;
info.time = zeros(0, 1);
info.final_true_relres = initialTrueResidual;
info.final_solver_time_sec = solverElapsed;
info.final_diagnostic_time_sec = diagnosticElapsed;
info.final_wall_time_sec = wallElapsed;

end
