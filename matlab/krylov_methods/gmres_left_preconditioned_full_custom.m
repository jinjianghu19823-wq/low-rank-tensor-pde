function [x, info] = gmres_left_preconditioned_full_custom( ...
    operatorFunction, preconditionerFunction, b, x0, ...
    maximumIteration, targetOriginalTolerance, displayProgress)
%GMRES_LEFT_PRECONDITIONED_FULL_CUSTOM Full GMRES for M*A*x=M*b.
%
% The Arnoldi process is built for the left-preconditioned operator M*A.
% The reduced and independently recomputed preconditioned residuals are
% normalized by ||M*b||_2. Stopping uses the independently evaluated
% original residual ||b-A*x||_2/||b||_2.


%% 1. Check the inputs

if nargin < 7
    displayProgress = false;
end

if maximumIteration < 1
    error('maximumIteration must be at least 1.');
end

if targetOriginalTolerance <= 0
    error('targetOriginalTolerance must be positive.');
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


%% 2. Form the initial original and preconditioned residuals

wallTimer = tic;
solverElapsed = 0;
diagnosticElapsed = 0;

solverTimer = tic;

preconditionedRightHandSide = preconditionerFunction(b);
normPreconditionedRightHandSide = ...
    norm(preconditionedRightHandSide);

if normPreconditionedRightHandSide == 0
    error('The preconditioned right-hand side must be nonzero.');
end

initialOriginalResidualVector = b - operatorFunction(x0);
initialPreconditionedResidualVector = ...
    preconditionerFunction(initialOriginalResidualVector);
beta = norm(initialPreconditionedResidualVector);

initialOriginalResidual = ...
    norm(initialOriginalResidualVector) / normB;
initialPreconditionedResidual = ...
    beta / normPreconditionedRightHandSide;

solverElapsed = solverElapsed + toc(solverTimer);

if initialOriginalResidual <= targetOriginalTolerance
    x = x0;
    info = empty_information( ...
        initialOriginalResidual, initialPreconditionedResidual, ...
        normPreconditionedRightHandSide, solverElapsed, ...
        diagnosticElapsed, toc(wallTimer));
    return
end

if beta <= 1e-14 * normPreconditionedRightHandSide
    error('The preconditioned starting residual is too small.');
end


%% 3. Allocate the Arnoldi basis and diagnostic histories

numberOfUnknowns = length(b);
V = zeros(numberOfUnknowns, maximumIteration + 1);
V(:,1) = initialPreconditionedResidualVector / beta;
H = zeros(maximumIteration + 1, maximumIteration);

computedPreconditionedResidualHistory = ...
    zeros(maximumIteration, 1);
truePreconditionedResidualHistory = ...
    zeros(maximumIteration, 1);
originalResidualHistory = zeros(maximumIteration, 1);
solverTimeHistory = zeros(maximumIteration, 1);
diagnosticTimeHistory = zeros(maximumIteration, 1);
wallTimeHistory = zeros(maximumIteration, 1);

x = x0;
y = [];
breakdown = false;


%% 4. Run left-preconditioned Arnoldi and GMRES

for j = 1:maximumIteration

    solverTimer = tic;

    w = preconditionerFunction(operatorFunction(V(:,j)));


    %% 4a. Orthogonalise with one modified Gram--Schmidt pass

    for i = 1:j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end

    H(j+1,j) = norm(w);

    if H(j+1,j) > 1e-14 && j < maximumIteration
        V(:,j+1) = w / H(j+1,j);
    end


    %% 4b. Solve the reduced least-squares problem

    smallRightHandSide = zeros(j+1, 1);
    smallRightHandSide(1) = beta;
    currentHessenberg = H(1:j+1, 1:j);
    y = currentHessenberg \ smallRightHandSide;
    smallResidual = ...
        smallRightHandSide - currentHessenberg * y;

    computedPreconditionedResidualHistory(j) = ...
        norm(smallResidual) / normPreconditionedRightHandSide;


    %% 4c. Form the current full solution

    x = x0 + V(:,1:j) * y;
    solverElapsed = solverElapsed + toc(solverTimer);


    %% 4d. Recompute both residuals independently

    diagnosticTimer = tic;

    originalResidualVector = b - operatorFunction(x);
    originalResidualHistory(j) = ...
        norm(originalResidualVector) / normB;

    truePreconditionedResidualVector = ...
        preconditionerFunction(originalResidualVector);
    truePreconditionedResidualHistory(j) = ...
        norm(truePreconditionedResidualVector) / ...
        normPreconditionedRightHandSide;

    diagnosticElapsed = ...
        diagnosticElapsed + toc(diagnosticTimer);

    solverTimeHistory(j) = solverElapsed;
    diagnosticTimeHistory(j) = diagnosticElapsed;
    wallTimeHistory(j) = toc(wallTimer);

    if displayProgress
        fprintf([ ...
            'Preconditioned full GMRES iteration %2d: ', ...
            'reduced %.3e, independent preconditioned %.3e, ', ...
            'original %.3e\n'], ...
            j, computedPreconditionedResidualHistory(j), ...
            truePreconditionedResidualHistory(j), ...
            originalResidualHistory(j));
    end


    %% 4e. Stop only from the original true residual

    if originalResidualHistory(j) <= targetOriginalTolerance
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


%% 5. Return histories and timings

numberOfIterations = j;

info.iterations = numberOfIterations;
info.iter = (1:numberOfIterations).';
info.converged = ...
    originalResidualHistory(numberOfIterations) <= ...
    targetOriginalTolerance;
info.breakdown = breakdown;
info.initial_original_true_relative_residual = ...
    initialOriginalResidual;
info.initial_preconditioned_true_relative_residual = ...
    initialPreconditionedResidual;
info.computed_preconditioned_relative_residual = ...
    computedPreconditionedResidualHistory(1:numberOfIterations);
info.true_preconditioned_relative_residual = ...
    truePreconditionedResidualHistory(1:numberOfIterations);
info.original_true_relative_residual = ...
    originalResidualHistory(1:numberOfIterations);
info.norm_preconditioned_right_hand_side = ...
    normPreconditionedRightHandSide;
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

end


function info = empty_information( ...
    initialOriginalResidual, initialPreconditionedResidual, ...
    normPreconditionedRightHandSide, solverElapsed, ...
    diagnosticElapsed, wallElapsed)
%EMPTY_INFORMATION Return a consistent zero-iteration result.

info.iterations = 0;
info.iter = zeros(0, 1);
info.converged = true;
info.breakdown = false;
info.initial_original_true_relative_residual = ...
    initialOriginalResidual;
info.initial_preconditioned_true_relative_residual = ...
    initialPreconditionedResidual;
info.computed_preconditioned_relative_residual = zeros(0, 1);
info.true_preconditioned_relative_residual = zeros(0, 1);
info.original_true_relative_residual = zeros(0, 1);
info.norm_preconditioned_right_hand_side = ...
    normPreconditionedRightHandSide;
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

end
