function test_poisson_fast_diagonalization_full_gmres
%TEST_POISSON_FAST_DIAGONALIZATION_FULL_GMRES Focused full-format checks.


%% 0. Prepare MATLAB

add_toolboxes();
rng(20260730, 'twister');

fprintf('\nStarting fast-diagonalization full-GMRES tests.\n\n');


%% Test 1. Full preconditioner action matches direct spectral division

fprintf('Test 1: full preconditioner action...\n');

N = 6;
h = 1 / (N + 1);
onesVector = ones(N, 1);
A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

[preconditioner, buildInfo] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner( ...
        A1, [N, N, N]);

inverseMultiplier = ...
    double(full(preconditioner.inverse_eigenvalue_tensor));
inputVector = randn(N^3, 1);

actualOutput = ...
    apply_poisson_fast_diagonalization_full_preconditioner( ...
        inputVector, inverseMultiplier);

inputTensor = reshape(inputVector, [N, N, N]);
spectralInput = apply_transform_in_all_modes(inputTensor, @dst);
lambda = preconditioner.one_dimensional_eigenvalues;
lambdaSum = lambda + reshape(lambda, 1, N) + ...
    reshape(lambda, 1, 1, N);
spectralReference = spectralInput ./ lambdaSum;
referenceTensor = ...
    apply_transform_in_all_modes(spectralReference, @idst);
referenceOutput = referenceTensor(:);

actionError = norm(actualOutput - referenceOutput) / ...
    norm(referenceOutput);

assert(actionError < 1e-12, ...
    'The full fast-diagonalization action is inconsistent.');
assert(buildInfo.inverse_eigenvalue_relative_frobenius_error < 1e-12, ...
    'The full-rank spectral multiplier should be exact.');

fprintf('  Relative action error: %.3e\n', actionError);
fprintf('  Test 1 passed.\n\n');


%% Test 2. Exact fast diagonalization gives a one-step GMRES solve

fprintf('Test 2: one-step exact-preconditioned full GMRES...\n');

gridPoints = (1:N).' / (N + 1);
oneDimensionalRightHandSide = ...
    exp(-5 * (gridPoints - 1/2).^2);
b = kron(oneDimensionalRightHandSide, ...
    kron(oneDimensionalRightHandSide, ...
         oneDimensionalRightHandSide));
x0 = zeros(N^3, 1);

operatorFunction = @(x) ...
    poisson_action_full_vector_3d(x, A1, N);
preconditionerFunction = @(x) ...
    apply_poisson_fast_diagonalization_full_preconditioner( ...
        x, inverseMultiplier);

[~, info] = gmres_left_preconditioned_full_custom( ...
    operatorFunction, preconditionerFunction, b, x0, ...
    3, 1e-11, false);

residualGap = max(abs( ...
    info.computed_preconditioned_relative_residual - ...
    info.true_preconditioned_relative_residual));

assert(info.converged, ...
    'The exact-preconditioned solve did not converge.');
assert(info.iterations == 1, ...
    'The exact-preconditioned solve should take one Arnoldi step.');
assert(info.original_true_relative_residual(end) < 1e-11, ...
    'The exact-preconditioned original residual is too large.');
assert(residualGap < 1e-11, ...
    'The reduced and independent preconditioned residuals disagree.');

fprintf('  Final original residual: %.3e\n', ...
    info.original_true_relative_residual(end));
fprintf('  Preconditioned residual gap: %.3e\n', residualGap);
fprintf('  Test 2 passed.\n\n');


%% Test 3. Delayed cycle matches ordinary full GMRES at the same step

fprintf('Test 3: delayed-assembly full GMRES cycle...\n');

identityPreconditioner = @(x) x;
fixedCycleLength = 3;

[ordinarySolution, ordinaryInfo] = ...
    gmres_left_preconditioned_full_custom( ...
        operatorFunction, identityPreconditioner, b, x0, ...
        fixedCycleLength, 1e-30, false);

[delayedSolution, delayedInfo] = ...
    gmres_left_preconditioned_full_fixed_cycle( ...
        operatorFunction, identityPreconditioner, b, x0, ...
        fixedCycleLength, 1e-30, false);

solutionDifference = norm( ...
    ordinarySolution - delayedSolution) / norm(ordinarySolution);
HessenbergDifference = norm( ...
    ordinaryInfo.H - delayedInfo.H, 'fro');
residualDifference = abs( ...
    ordinaryInfo.original_true_relative_residual(end) - ...
    delayedInfo.original_true_relative_residual);

assert(ordinaryInfo.iterations == fixedCycleLength, ...
    'The ordinary comparison did not complete the declared steps.');
assert(delayedInfo.iterations == fixedCycleLength, ...
    'The delayed comparison did not complete the declared steps.');
assert(~delayedInfo.breakdown, ...
    'The delayed comparison encountered an unexpected breakdown.');
assert(solutionDifference < 1e-13, ...
    'Delayed solution assembly changed the full-GMRES iterate.');
assert(HessenbergDifference < 1e-13, ...
    'The delayed cycle changed the Hessenberg matrix.');
assert(residualDifference < 1e-13, ...
    'The delayed cycle changed the original residual.');

fprintf('  Relative solution difference: %.3e\n', ...
    solutionDifference);
fprintf('  Original-residual difference: %.3e\n', ...
    residualDifference);
fprintf('  Test 3 passed.\n\n');

fprintf('All fast-diagonalization full-GMRES tests passed.\n');

end


function Y = apply_transform_in_all_modes(X, transformFunction)

Y = X;

for mode = 1:3

    permutation = [mode, setdiff(1:3, mode, 'stable')];
    inversePermutation = zeros(1, 3);
    inversePermutation(permutation) = 1:3;

    permutedY = permute(Y, permutation);
    transformedMatrix = transformFunction( ...
        reshape(permutedY, size(Y, mode), []));
    permutedY = reshape(transformedMatrix, size(permutedY));
    Y = permute(permutedY, inversePermutation);

end

end
