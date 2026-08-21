function test_poisson_fast_diagonalization_tucker_preconditioner
%TEST_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Focused checks.
%
% These checks validate the exact Tucker Hadamard representation, the
% DST-I fast-diagonalization action, and the identity M*A=I when the exact
% inverse-eigenvalue tensor is retained on a tiny grid.


%% 0. Prepare MATLAB and Tensor Toolbox

add_toolboxes();
rng(20260730, 'twister');

fprintf('\nStarting fast-diagonalization Tucker preconditioner tests.\n\n');


%% Test 1. Exact Tucker Hadamard representation

fprintf('Test 1: exact Tucker Hadamard product...\n');

X = hosvd(tensor(randn(4, 4, 4)), 1e-12, 'verbosity', 0);
Y = hosvd(tensor(randn(4, 4, 4)), 1e-12, 'verbosity', 0);
Z = tucker_hadamard_exact(X, Y);

hadamardReference = double(full(X)) .* double(full(Y));
hadamardDifference = double(full(Z)) - hadamardReference;
hadamardError = norm(hadamardDifference(:)) / ...
    norm(hadamardReference(:));

assert(hadamardError < 1e-12, ...
    'The exact Tucker Hadamard representation is inconsistent.');

fprintf('  Relative error: %.3e\n', hadamardError);
fprintf('  Test 1 passed.\n\n');


%% Test 1b. Exact rank-one Tucker Hadamard representation

fprintf('Test 1b: exact rank-one Tucker Hadamard product...\n');

rankOneFactorsX = {randn(4, 1), randn(4, 1), randn(4, 1)};
rankOneFactorsY = {randn(4, 1), randn(4, 1), randn(4, 1)};
rankOneX = ttensor(tensor(2, [1, 1, 1]), rankOneFactorsX);
rankOneY = ttensor(tensor(-3, [1, 1, 1]), rankOneFactorsY);
rankOneZ = tucker_hadamard_exact(rankOneX, rankOneY);
rankOneReference = double(full(rankOneX)) .* double(full(rankOneY));
rankOneDifference = double(full(rankOneZ)) - rankOneReference;
rankOneError = norm(rankOneDifference(:)) / norm(rankOneReference(:));

assert(ndims(rankOneZ) == 3, ...
    'The exact rank-one Hadamard result lost its tensor order.');
assert(rankOneError < 1e-12, ...
    'The exact rank-one Tucker Hadamard representation is inconsistent.');

fprintf('  Relative error: %.3e\n', rankOneError);
fprintf('  Test 1b passed.\n\n');


%% Test 2. Fast-diagonalization action against a full DST reference

fprintf('Test 2: Tucker action versus full DST reference...\n');

N = 6;
h = 1 / (N + 1);
onesVector = ones(N, 1);
A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], ...
    -1:1, N, N) / h^2;

[preconditioner, buildInfo] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner( ...
        A1, [N, N, N]);

Yfull = randn(N, N, N);
Ytucker = hosvd(tensor(Yfull), 1e-13, 'verbosity', 0);

[Ztucker, actionInfo] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner( ...
        Ytucker, preconditioner, 1e-13, [N, N, N]);
[Zexact, exactActionInfo] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact( ...
        Ytucker, preconditioner);

spectralY = apply_transform_in_all_modes(Yfull, @dst);

lambda = preconditioner.one_dimensional_eigenvalues;
lambdaSum = lambda + reshape(lambda, 1, N) + ...
    reshape(lambda, 1, 1, N);
spectralZ = spectralY ./ lambdaSum;
fullReference = apply_transform_in_all_modes(spectralZ, @idst);

actionDifference = double(full(Ztucker)) - fullReference;
actionError = norm(actionDifference(:)) / norm(fullReference(:));
exactActionDifference = double(full(Zexact)) - fullReference;
exactActionError = norm(exactActionDifference(:)) / ...
    norm(fullReference(:));

assert(actionError < 1e-10, ...
    'The Tucker fast-diagonalization action is inconsistent.');
assert(exactActionError < 1e-10, ...
    'The unrounded fixed-linear preconditioner action is inconsistent.');
assert(~exactActionInfo.rounding_performed, ...
    'The fixed-linear preconditioner must not recompress its input.');
assert(buildInfo.inverse_eigenvalue_relative_frobenius_error < 1e-12, ...
    'The full-rank inverse-eigenvalue tensor should be exact.');
assert(~actionInfo.rank_cap_active, ...
    'The full mode-size cap should remain inactive.');
assert(isfield(actionInfo, 'kernel_timing') && ...
        isfield(actionInfo, 'call_time_sec'), ...
    'The preconditioner did not return detailed kernel timings.');
assert(actionInfo.kernel_timing.preconditioner_forward_dst_time_sec >= 0 && ...
       actionInfo.kernel_timing.preconditioner_exact_hadamard_time_sec >= 0 && ...
       actionInfo.kernel_timing.preconditioner_inverse_dst_time_sec >= 0 && ...
       actionInfo.kernel_timing.sthosvd_svd_time_sec >= 0, ...
    'A fast-diagonalization kernel timing is negative.');
assert(actionInfo.call_time_sec > 0, ...
    'The preconditioner call time must be positive.');

fprintf('  Relative action error: %.3e\n', actionError);
fprintf('  Exact fixed-linear action error: %.3e\n', exactActionError);
fprintf('  Test 2 passed.\n\n');


%% Test 3. Exact full-rank preconditioning gives M*A=I

fprintf('Test 3: exact preconditioned Poisson action...\n');

Xfull = randn(N, N, N);
Xtucker = hosvd(tensor(Xfull), 1e-13, 'verbosity', 0);

AX = exact_poisson_action(Xtucker, A1);
[MAX, ~] = apply_poisson_fast_diagonalization_tucker_preconditioner( ...
    AX, preconditioner, 1e-13, [N, N, N]);

identityDifference = double(full(MAX)) - Xfull;
identityDefect = norm(identityDifference(:)) / norm(Xfull(:));

assert(identityDefect < 1e-10, ...
    'The full-rank fast-diagonalization preconditioner is not exact.');

fprintf('  Relative identity defect: %.3e\n', identityDefect);
fprintf('  Test 3 passed.\n\n');


%% Test 4. Independent preconditioned residual norm

fprintf('Test 4: independent preconditioned residual norm...\n');

Ffull = randn(N, N, N);
F = hosvd(tensor(Ffull), 1e-13, 'verbosity', 0);

diagnosticNorm = ...
    preconditioned_residual_norm_tucker_poisson_fast_diagonalization( ...
        Xtucker, F, A1, preconditioner, 1e-13);

AXfull = double(full(exact_poisson_action(Xtucker, A1)));
residualFull = Ffull - AXfull;
spectralResidual = apply_transform_in_all_modes(residualFull, @dst);
spectralPreconditionedResidual = spectralResidual ./ lambdaSum;
preconditionedResidualFull = apply_transform_in_all_modes( ...
    spectralPreconditionedResidual, @idst);
referenceNorm = norm(preconditionedResidualFull(:));
diagnosticRelativeError = abs(diagnosticNorm - referenceNorm) / ...
    referenceNorm;

assert(diagnosticRelativeError < 1e-10, ...
    'The independent preconditioned residual norm is inconsistent.');

fprintf('  Relative norm error: %.3e\n', diagnosticRelativeError);
fprintf('  Test 4 passed.\n\n');

fprintf('All fast-diagonalization Tucker preconditioner tests passed.\n');

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


function Y = exact_poisson_action(X, A1)

term1 = ttm(X, A1, 1);
term2 = ttm(X, A1, 2);
term3 = ttm(X, A1, 3);

partialSum = tucker_axpby_exact(term1, 1, term2, 1);
Y = tucker_axpby_exact(partialSum, 1, term3, 1);

end
