function test_tucker_gmres_foundation
%TEST_TUCKER_GMRES_FOUNDATION Correctness tests for Tucker-GMRES routines.
%
% This file contains five small tests:
%
%   1. Tucker addition and recompression.
%   2. Four-dimensional Poisson operator consistency.
%   3. Tucker-GMRES with the identity operator.
%   4. Tucker-GMRES and full GMRES on a tiny 4D Poisson problem.
%   5. Restarted full and Tucker-GMRES on the same tiny problem.
%
% These are correctness tests, not performance experiments.


%% 0. Prepare MATLAB and Tensor Toolbox

add_toolboxes();
rng(1, 'twister');

fprintf('\nStarting Tucker-GMRES foundation tests.\n\n');


%% Test 1. Tucker addition and recompression

fprintf('Test 1: Tucker addition and recompression...\n');

tensorDimension = 4;

coreX = tensor(randn(2, 2, 2));
coreY = tensor(randn(2, 1, 2));

factorsX = cell(3, 1);
factorsY = cell(3, 1);

factorsX{1} = randn(tensorDimension, 2);
factorsX{2} = randn(tensorDimension, 2);
factorsX{3} = randn(tensorDimension, 2);

factorsY{1} = randn(tensorDimension, 2);
factorsY{2} = randn(tensorDimension, 1);
factorsY{3} = randn(tensorDimension, 2);

X = ttensor(coreX, factorsX);
Y = ttensor(coreY, factorsY);

alpha = 1.3;
beta = -0.7;
compressionTolerance = 1e-12;

Z = tucker_axpby_round( ...
    X, alpha, Y, beta, compressionTolerance);

Zexact = tucker_axpby_exact(X, alpha, Y, beta);

exactSum = alpha * full(X) + beta * full(Y);

additionError = norm(exactSum - full(Z)) / norm(exactSum);

exactAdditionError = ...
    norm(exactSum - full(Zexact)) / norm(exactSum);

fprintf('  Relative addition error: %.3e\n', additionError);
fprintf('  Exact Tucker addition error: %.3e\n', ...
    exactAdditionError);

assert(additionError < 1e-10, ...
    'The Tucker addition test did not reach the expected accuracy.');

assert(exactAdditionError < 1e-12, ...
    'The exact Tucker addition test did not reach the expected accuracy.');

% Check the important cancellation case X - X = 0.
zeroTensor = tucker_axpby_round( ...
    X, 1, X, -1, compressionTolerance);

assert(norm(zeroTensor) < 1e-12, ...
    'The cancellation test X-X did not produce zero.');

fprintf('  Test 1 passed.\n\n');


%% Test 2. Four-dimensional Poisson operator consistency

fprintf('Test 2: Four-dimensional Poisson operator consistency...\n');

N = 3;
numberOfModes = 4;

h = 1 / (N + 1);
onesVector = ones(N, 1);

A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

% Tensor Toolbox's dense tensor/ttm reference calculation requires a dense
% matrix. The actual Tucker routine above still uses the sparse A1.
A1full = full(A1);

fullInput = tensor(randn(N, N, N, N));

% At this tiny size, a very accurate HOSVD representation is affordable.
tuckerInput = hosvd(fullInput, 1e-12, 'verbosity', 0);

tuckerOutput = poisson_action_tucker( ...
    tuckerInput, A1, compressionTolerance);

% Calculate the same action directly on the small full tensor.
fullOutput = ttm(fullInput, A1full, 1);

for mode = 2:numberOfModes
    fullOutput = fullOutput + ttm(fullInput, A1full, mode);
end

tensorActionError = ...
    norm(fullOutput - full(tuckerOutput)) / norm(fullOutput);

fprintf('  Tucker action relative error: %.3e\n', ...
    tensorActionError);

assert(tensorActionError < 1e-10, ...
    'The Tucker Poisson action does not match the full tensor action.');

% Build the small explicit Kronecker-sum matrix as an independent check.
identityMatrix = speye(N);
matrixDimension = N^numberOfModes;
poissonMatrix = sparse(matrixDimension, matrixDimension);

for activeMode = 1:numberOfModes

    oneKroneckerTerm = 1;

    % MATLAB vectorisation places mode 1 at the right of the Kronecker
    % product, so the loop runs from the final mode down to mode 1.
    for mode = numberOfModes:-1:1

        if mode == activeMode
            nextMatrix = A1;
        else
            nextMatrix = identityMatrix;
        end

        oneKroneckerTerm = kron(oneKroneckerTerm, nextMatrix);

    end

    poissonMatrix = poissonMatrix + oneKroneckerTerm;

end


fullInputValues = double(fullInput);
fullOutputValues = double(fullOutput);

matrixOutputVector = poissonMatrix * fullInputValues(:);

kroneckerError = ...
    norm(matrixOutputVector - fullOutputValues(:)) / ...
    norm(fullOutputValues(:));

fprintf('  Kronecker consistency error: %.3e\n', kroneckerError);

assert(kroneckerError < 1e-12, ...
    'The tensor Poisson action does not match the Kronecker matrix.');

% Check the true-residual formula against a directly formed full residual.
testRightHandSide = hosvd( ...
    tensor(randn(N, N, N, N)), 1e-12, 'verbosity', 0);

trueResidualFromInnerProducts = ...
    true_residual_tucker_poisson( ...
    tuckerInput, testRightHandSide, A1);

fullTuckerInput = full(tuckerInput);
fullPoissonAction = ttm(fullTuckerInput, A1full, 1);

for mode = 2:numberOfModes
    fullPoissonAction = ...
        fullPoissonAction + ttm(fullTuckerInput, A1full, mode);
end

trueResidualFromFullTensor = ...
    norm(full(testRightHandSide) - fullPoissonAction) / ...
    norm(testRightHandSide);

trueResidualDifference = abs( ...
    trueResidualFromInnerProducts - trueResidualFromFullTensor);

fprintf('  True-residual consistency error: %.3e\n', ...
    trueResidualDifference);

assert(trueResidualDifference < 1e-12, ...
    'The inner-product true residual does not match the full residual.');

fprintf('  Test 2 passed.\n\n');


%% Test 3. Tucker-GMRES with the identity operator

fprintf('Test 3: Tucker-GMRES with the identity operator...\n');

Fidentity = hosvd( ...
    tensor(randn(4, 4, 4)), 1e-12, 'verbosity', 0);

U0identity = 0 * Fidentity;

identityOperator = @(U) U;

identityTrueResidual = @(U) ...
    norm(full(Fidentity) - full(U)) / norm(Fidentity);

[Uidentity, identityInfo] = tucker_gmres( ...
    identityOperator, Fidentity, U0identity, ...
    3, 1e-10, compressionTolerance, ...
    identityTrueResidual);

identityFinalResidual = identityTrueResidual(Uidentity);

fprintf('  Iterations: %d\n', identityInfo.iterations);
fprintf('  Final true residual: %.3e\n', identityFinalResidual);

assert(identityInfo.converged, ...
    'Tucker-GMRES did not converge for the identity operator.');

assert(identityInfo.iterations == 1, ...
    'The identity problem should be solved in one GMRES iteration.');

assert(identityInfo.orthogonalisation_passes == 1, ...
    'Tucker-GMRES should use one modified Gram--Schmidt pass.');

assert(identityFinalResidual < 1e-10, ...
    'The identity problem residual is too large.');

fprintf('  Test 3 passed.\n\n');


%% Test 4. Tiny four-dimensional Poisson solve

fprintf('Test 4: Tiny 4D Poisson solve...\n');

N = 4;
numberOfModes = 4;

h = 1 / (N + 1);
onesVector = ones(N, 1);

A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

% Again, only the tiny full-tensor reference calculation needs a dense
% copy. Tucker operations continue to use the sparse A1.
A1full = full(A1);

gridIndex = (1:N).';
firstModeVector = sin(pi * gridIndex / (N + 1));
secondModeVector = sin(2 * pi * gridIndex / (N + 1));

% Construct a manufactured solution from two separable terms. The two
% terms use different one-dimensional eigenvectors, so the solve is not a
% one-iteration eigenvector example.
firstTerm = ...
    reshape(firstModeVector, [N, 1, 1, 1]) .* ...
    reshape(firstModeVector, [1, N, 1, 1]) .* ...
    reshape(firstModeVector, [1, 1, N, 1]) .* ...
    reshape(firstModeVector, [1, 1, 1, N]);

secondTerm = ...
    reshape(firstModeVector,  [N, 1, 1, 1]) .* ...
    reshape(secondModeVector, [1, N, 1, 1]) .* ...
    reshape(firstModeVector,  [1, 1, N, 1]) .* ...
    reshape(secondModeVector, [1, 1, 1, N]);

exactSolutionValues = firstTerm + 0.3 * secondTerm;
exactSolutionTensor = tensor(exactSolutionValues);

% Calculate the right-hand side directly as a small full tensor.
fullRightHandSide = ttm(exactSolutionTensor, A1full, 1);

for mode = 2:numberOfModes
    fullRightHandSide = ...
        fullRightHandSide + ttm(exactSolutionTensor, A1full, mode);
end

F = hosvd(fullRightHandSide, 1e-12, 'verbosity', 0);
U0 = 0 * F;

poissonOperator = @(U) ...
    poisson_action_tucker(U, A1, compressionTolerance);

trueResidualFunction = @(U) ...
    true_residual_tucker_poisson(U, F, A1);

maximumIteration = 8;
targetTolerance = 1e-8;

[Utucker, tuckerInfo] = tucker_gmres( ...
    poissonOperator, F, U0, maximumIteration, ...
    targetTolerance, compressionTolerance, ...
    trueResidualFunction);

tuckerFinalResidual = trueResidualFunction(Utucker);

% Rebuild the explicit matrix for the tiny full-GMRES reference solve.
identityMatrix = speye(N);
matrixDimension = N^numberOfModes;
poissonMatrix = sparse(matrixDimension, matrixDimension);

for activeMode = 1:numberOfModes

    oneKroneckerTerm = 1;

    for mode = numberOfModes:-1:1

        if mode == activeMode
            nextMatrix = A1;
        else
            nextMatrix = identityMatrix;
        end

        oneKroneckerTerm = kron(oneKroneckerTerm, nextMatrix);

    end

    poissonMatrix = poissonMatrix + oneKroneckerTerm;

end


rightHandSideValues = double(full(F));
rightHandSideVector = rightHandSideValues(:);

[fullSolutionVector, fullFlag] = gmres( ...
    poissonMatrix, rightHandSideVector, [], ...
    1e-10, maximumIteration);

fullRelativeResidual = ...
    norm(rightHandSideVector - ...
    poissonMatrix * fullSolutionVector) / ...
    norm(rightHandSideVector);

tuckerSolutionValues = double(full(Utucker));

solutionDifference = ...
    norm(tuckerSolutionValues(:) - fullSolutionVector) / ...
    norm(fullSolutionVector);

fprintf('  Tucker-GMRES iterations: %d\n', tuckerInfo.iterations);
fprintf('  Tucker-GMRES true residual: %.3e\n', ...
    tuckerFinalResidual);
fprintf('  Full GMRES true residual: %.3e\n', ...
    fullRelativeResidual);
fprintf('  Relative solution difference: %.3e\n', ...
    solutionDifference);

assert(tuckerInfo.converged, ...
    'Tucker-GMRES did not reach the target on the tiny 4D problem.');

assert(fullFlag == 0, ...
    'Full GMRES did not converge on the tiny reference problem.');

assert(tuckerFinalResidual < targetTolerance, ...
    'The final Tucker-GMRES true residual is too large.');

assert(tuckerInfo.solution_assembly_count == 1, ...
    'Unrestarted Tucker-GMRES should assemble its solution once.');

assert(size(tuckerInfo.solution_ranks, 1) == 1 && ...
        tuckerInfo.solution_rank_iteration == tuckerInfo.iterations, ...
    'The solution rank should be recorded only at delayed assembly.');

assert(isequal(tuckerInfo.true_residual_iteration, ...
        [0; tuckerInfo.iterations]), ...
    'True residuals should be evaluated only before and after the cycle.');

assert(numel(tuckerInfo.computed_relative_residual) == ...
        tuckerInfo.iterations, ...
    'The reduced residual should still be recorded at every Arnoldi step.');

assert(fullRelativeResidual < 1e-10, ...
    'The full GMRES reference residual is too large.');

assert(solutionDifference < 1e-7, ...
    'Tucker-GMRES and full GMRES produced different small solutions.');

fprintf('  Test 4 passed.\n\n');


%% Test 5. Restarted full and Tucker-GMRES

fprintf('Test 5: Restarted full and Tucker-GMRES...\n');

restartLength = 1;
maximumCycles = 4;
restartStoppingTolerance = 1e-14;

[fullRestartedSolution, fullRestartedInfo] = ...
    gmres_restarted_custom( ...
    @(x) poissonMatrix * x, rightHandSideVector, ...
    zeros(size(rightHandSideVector)), restartLength, maximumCycles, ...
    restartStoppingTolerance, false);

[tuckerRestartedSolution, tuckerRestartedInfo] = ...
    tucker_gmres_restarted( ...
    poissonOperator, F, U0, restartLength, maximumCycles, ...
    restartStoppingTolerance, compressionTolerance, ...
    trueResidualFunction, false);

fullRestartedResidual = norm( ...
    rightHandSideVector - poissonMatrix * fullRestartedSolution) / ...
    norm(rightHandSideVector);

tuckerRestartedResidual = ...
    trueResidualFunction(tuckerRestartedSolution);

tuckerRestartedValues = double(full(tuckerRestartedSolution));

restartedSolutionDifference = norm( ...
    tuckerRestartedValues(:) - fullRestartedSolution) / ...
    norm(fullRestartedSolution);

fprintf('  Full restarted residual: %.3e\n', ...
    fullRestartedResidual);
fprintf('  Tucker restarted residual: %.3e\n', ...
    tuckerRestartedResidual);
fprintf('  Restarted solution difference: %.3e\n', ...
    restartedSolutionDifference);

assert(fullRestartedInfo.number_of_cycles == maximumCycles, ...
    'The full restarted solver did not complete the expected cycles.');

assert(tuckerRestartedInfo.number_of_cycles == maximumCycles, ...
    'The Tucker restarted solver did not complete the expected cycles.');

assert(fullRestartedInfo.orthogonalisation_passes == 1, ...
    'Restarted full GMRES should use one Gram--Schmidt pass.');

assert(tuckerRestartedInfo.orthogonalisation_passes == 1, ...
    'Restarted Tucker-GMRES should use one Gram--Schmidt pass.');

assert(fullRestartedInfo.iterations == ...
        restartLength * maximumCycles, ...
    'The full restarted cumulative iteration count is incorrect.');

assert(tuckerRestartedInfo.iterations == ...
        restartLength * maximumCycles, ...
    'The Tucker restarted cumulative iteration count is incorrect.');

assert(length(fullRestartedInfo.true_relative_residual) == ...
        fullRestartedInfo.iterations + 1, ...
    'The full restarted true-residual history has the wrong length.');

assert(length(tuckerRestartedInfo.true_relative_residual) == ...
        tuckerRestartedInfo.number_of_cycles + 1, ...
    ['Restarted Tucker-GMRES should evaluate the true residual only at ', ...
     'cycle boundaries.']);

assert(tuckerRestartedInfo.solution_assembly_count == ...
        tuckerRestartedInfo.number_of_cycles, ...
    'Restarted Tucker-GMRES should assemble once per completed cycle.');

assert(isequal(tuckerRestartedInfo.solution_rank_iteration, ...
        cumsum(tuckerRestartedInfo.cycle_iterations)), ...
    'Restarted solution ranks should be recorded at cycle endpoints.');

assert(fullRestartedResidual < 1, ...
    'Restarted full GMRES did not reduce the residual.');

assert(tuckerRestartedResidual < 1, ...
    'Restarted Tucker-GMRES did not reduce the residual.');

assert(restartedSolutionDifference < 1e-8, ...
    'The restarted full and Tucker solutions do not agree.');

fprintf('  Test 5 passed.\n\n');


%% Final message

fprintf('All Tucker-GMRES foundation tests passed.\n');

end
