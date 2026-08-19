function test_tucker_gmres_3d_stress_setup
%TEST_TUCKER_GMRES_3D_STRESS_SETUP Small checks for the 3D experiment.
%
% These checks use N=4. They validate the three-dimensional full and Tucker
% Poisson actions, the rank-one Gaussian right-hand side, and the new
% active-basis storage stop without allocating a large problem.


%% 1. Prepare MATLAB and the Tensor Toolbox

add_toolboxes();
rng(7, 'twister');

fprintf('\nStarting 3D Tucker-GMRES stress-experiment checks.\n\n');


%% 2. Construct a tiny three-dimensional Poisson problem

N = 4;
numberOfModes = 3;
h = 1 / (N + 1);
onesVector = ones(N, 1);

A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

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


%% 3. Check the matrix-free full-vector action

testVector = randn(matrixDimension, 1);

matrixFreeOutput = ...
    poisson_action_full_vector_3d(testVector, A1, N);
explicitOutput = poissonMatrix * testVector;

fullActionError = norm(matrixFreeOutput - explicitOutput) / ...
    norm(explicitOutput);

fprintf('Full-vector action error: %.3e\n', fullActionError);

assert(fullActionError < 1e-13, ...
    'The 3D matrix-free action does not match the Kronecker matrix.');


%% 4. Check the Tucker action and Gaussian vectorization

gridPoints = (1:N).' / (N + 1);
gaussianVector = exp(-5 * (gridPoints - 1/2).^2);

F = ttensor(tensor(1, [1, 1, 1]), ...
    {gaussianVector, gaussianVector, gaussianVector});
U0 = 0 * F;

rightHandSide = kron(gaussianVector, ...
    kron(gaussianVector, gaussianVector));
fullF = double(full(F));

assert(norm(rightHandSide - fullF(:)) / norm(rightHandSide) < 1e-14, ...
    'The 3D Gaussian Tucker tensor has the wrong vectorization.');

compressionTolerance = 1e-12;
tuckerOperator = @(U) ...
    poisson_action_tucker(U, A1, compressionTolerance);

tuckerAction = double(full(tuckerOperator(F)));
fullAction = poisson_action_full_vector_3d( ...
    rightHandSide, A1, N);

tuckerActionError = norm(tuckerAction(:) - fullAction) / ...
    norm(fullAction);

fprintf('Tucker action error: %.3e\n', tuckerActionError);

assert(tuckerActionError < 1e-11, ...
    'The 3D Tucker and full-vector actions do not agree.');


%% 5. Check the active-basis memory stop

trueResidualFunction = @(U) ...
    true_residual_tucker_poisson(U, F, A1);

% The initial rank-one tensor stores one core entry and 3*N factor entries.
% Allow that tensor, but force the next basis tensor to cross the limit.
maximumBasisStorageEntries = 1 + 3 * N;

[~, unrestartedInfo] = tucker_gmres( ...
    tuckerOperator, F, U0, 3, 1e-16, ...
    compressionTolerance, trueResidualFunction, false, ...
    maximumBasisStorageEntries);

assert(unrestartedInfo.stopped_for_basis_memory, ...
    'Unrestarted Tucker-GMRES ignored its basis storage limit.');
assert(unrestartedInfo.iterations == 1, ...
    'The basis storage test should stop after one completed iteration.');
assert(unrestartedInfo.peak_basis_storage_entries > ...
    maximumBasisStorageEntries, ...
    'The recorded peak basis storage should cross the safety limit.');

[~, restartedInfo] = tucker_gmres_restarted( ...
    tuckerOperator, F, U0, 2, 2, 1e-16, ...
    compressionTolerance, trueResidualFunction, false, ...
    maximumBasisStorageEntries);

assert(restartedInfo.stopped_for_basis_memory, ...
    'Restarted Tucker-GMRES ignored its cycle storage limit.');
assert(restartedInfo.iterations == 1, ...
    'The restarted storage test should stop in its first cycle.');

fprintf('Active-basis storage stop passed.\n\n');
fprintf('All 3D stress-experiment checks passed.\n');

end
