function test_poisson_fast_diagonalization_tucker_preconditioner
%TEST_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Focused checks.

%% 0. Prepare MATLAB and Tensor Toolbox

add_toolboxes();
rng(20260730, 'twister');

fprintf('\nStarting fast-diagonalization Tucker preconditioner tests.\n\n');

%% Test 1. Exact Tucker Hadamard representation

fprintf('Test 1: exact Tucker Hadamard product...\n');

X = hosvd(tensor(randn(4, 4, 4)), 1e-12, 'verbosity', 0);
Y = hosvd(tensor(randn(4, 4, 4)), 1e-12, 'verbosity', 0);
Z = tucker_hadamard_exact(X, Y);

hadamard_reference = double(full(X)) .* double(full(Y));
hadamard_difference = double(full(Z)) - hadamard_reference;
hadamard_error = norm(hadamard_difference(:)) / ...
    norm(hadamard_reference(:));

assert(hadamard_error < 1e-12, ...
    'The exact Tucker Hadamard representation is inconsistent.');

fprintf('  Relative error: %.3e\n', hadamard_error);
fprintf('  Test 1 passed.\n\n');

%% Test 1b. Exact rank-one Tucker Hadamard representation

fprintf('Test 1b: exact rank-one Tucker Hadamard product...\n');

rank_one_factors_x = {randn(4, 1), randn(4, 1), randn(4, 1)};
rank_one_factors_y = {randn(4, 1), randn(4, 1), randn(4, 1)};
rank_one_x = ttensor(tensor(2, [1, 1, 1]), rank_one_factors_x);
rank_one_y = ttensor(tensor(-3, [1, 1, 1]), rank_one_factors_y);
rank_one_z = tucker_hadamard_exact(rank_one_x, rank_one_y);
rank_one_reference = double(full(rank_one_x)) .* double(full(rank_one_y));
rank_one_difference = double(full(rank_one_z)) - rank_one_reference;
rank_one_error = norm(rank_one_difference(:)) / norm(rank_one_reference(:));

assert(ndims(rank_one_z) == 3, ...
    'The exact rank-one Hadamard result lost its tensor order.');
assert(rank_one_error < 1e-12, ...
    'The exact rank-one Tucker Hadamard representation is inconsistent.');

fprintf('  Relative error: %.3e\n', rank_one_error);
fprintf('  Test 1b passed.\n\n');

%% Test 2. Fast-diagonalization action against a full DST reference

fprintf('Test 2: Tucker action versus full DST reference...\n');

N = 6;
h = 1 / (N + 1);
e = ones(N, 1);
A1 = spdiags( ...
    [-e, 2 * e, -e], ...
    -1:1, N, N) / h^2;

[preconditioner, build_info] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner( ...
        A1, [N, N, N]);

Yfull = randn(N, N, N);
Ytucker = hosvd(tensor(Yfull), 1e-13, 'verbosity', 0);

[Ztucker, action_info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner( ...
        Ytucker, preconditioner, 1e-13, [N, N, N]);
[Zexact, exact_action_info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact( ...
        Ytucker, preconditioner);

spectral_y = apply_transform_in_all_modes(Yfull, @dst);

lambda = preconditioner.one_dimensional_eigenvalues;
lambda_sum = lambda + reshape(lambda, 1, N) + ...
    reshape(lambda, 1, 1, N);
spectral_z = spectral_y ./ lambda_sum;
full_reference = apply_transform_in_all_modes(spectral_z, @idst);

action_difference = double(full(Ztucker)) - full_reference;
action_error = norm(action_difference(:)) / norm(full_reference(:));
exact_action_difference = double(full(Zexact)) - full_reference;
exact_action_error = norm(exact_action_difference(:)) / ...
    norm(full_reference(:));

assert(action_error < 1e-10, ...
    'The Tucker fast-diagonalization action is inconsistent.');
assert(exact_action_error < 1e-10, ...
    'The unrounded fixed-linear preconditioner action is inconsistent.');
assert(~exact_action_info.rounding_performed, ...
    'The fixed-linear preconditioner must not recompress its input.');
assert(build_info.inverse_eigenvalue_relative_frobenius_error < 1e-12, ...
    'The full-rank inverse-eigenvalue tensor should be exact.');
assert(~action_info.rank_cap_active, ...
    'The full mode-size cap should remain inactive.');
assert(isfield(action_info, 'kernel_timing') && ...
        isfield(action_info, 'call_time_sec'), ...
    'The preconditioner did not return detailed kernel timings.');
assert(action_info.kernel_timing.preconditioner_forward_dst_time_sec >= 0 && ...
       action_info.kernel_timing.preconditioner_exact_hadamard_time_sec >= 0 && ...
       action_info.kernel_timing.preconditioner_inverse_dst_time_sec >= 0 && ...
       action_info.kernel_timing.sthosvd_svd_time_sec >= 0, ...
    'A fast-diagonalization kernel timing is negative.');
assert(action_info.call_time_sec > 0, ...
    'The preconditioner call time must be positive.');

fprintf('  Relative action error: %.3e\n', action_error);
fprintf('  Exact fixed-linear action error: %.3e\n', exact_action_error);
fprintf('  Test 2 passed.\n\n');

%% Test 3. Exact full-rank preconditioning gives M*A=I

fprintf('Test 3: exact preconditioned Poisson action...\n');

Xfull = randn(N, N, N);
Xtucker = hosvd(tensor(Xfull), 1e-13, 'verbosity', 0);

AX = exact_poisson_action(Xtucker, A1);
[MAX, ~] = apply_poisson_fast_diagonalization_tucker_preconditioner( ...
    AX, preconditioner, 1e-13, [N, N, N]);

identity_difference = double(full(MAX)) - Xfull;
identity_defect = norm(identity_difference(:)) / norm(Xfull(:));

assert(identity_defect < 1e-10, ...
    'The full-rank fast-diagonalization preconditioner is not exact.');

fprintf('  Relative identity defect: %.3e\n', identity_defect);
fprintf('  Test 3 passed.\n\n');

%% Test 4. Independent preconditioned residual norm

fprintf('Test 4: independent preconditioned residual norm...\n');

Ffull = randn(N, N, N);
F = hosvd(tensor(Ffull), 1e-13, 'verbosity', 0);

diagnostic_norm = ...
    preconditioned_residual_norm_tucker_poisson_fast_diagonalization( ...
        Xtucker, F, A1, preconditioner, 1e-13);

AXfull = double(full(exact_poisson_action(Xtucker, A1)));
residual_full = Ffull - AXfull;
spectral_residual = apply_transform_in_all_modes(residual_full, @dst);
spectral_preconditioned_residual = spectral_residual ./ lambda_sum;
preconditioned_residual_full = apply_transform_in_all_modes( ...
    spectral_preconditioned_residual, @idst);
reference_norm = norm(preconditioned_residual_full(:));
diagnostic_relative_error = abs(diagnostic_norm - reference_norm) / ...
    reference_norm;

assert(diagnostic_relative_error < 1e-10, ...
    'The independent preconditioned residual norm is inconsistent.');

fprintf('  Relative norm error: %.3e\n', diagnostic_relative_error);
fprintf('  Test 4 passed.\n\n');

fprintf('All fast-diagonalization Tucker preconditioner tests passed.\n');

end

function Y = apply_transform_in_all_modes(X, transform)

Y = X;

for mode = 1:3

    permutation = [mode, setdiff(1:3, mode, 'stable')];
    inverse_permutation = zeros(1, 3);
    inverse_permutation(permutation) = 1:3;

    Y_perm = permute(Y, permutation);
    transformed_matrix = transform( ...
        reshape(Y_perm, size(Y, mode), []));
    Y_perm = reshape(transformed_matrix, size(Y_perm));
    Y = permute(Y_perm, inverse_permutation);

end

end

function Y = exact_poisson_action(X, A1)

term1 = ttm(X, A1, 1);
term2 = ttm(X, A1, 2);
term3 = ttm(X, A1, 3);

partial_sum = tucker_axpby_exact(term1, 1, term2, 1);
Y = tucker_axpby_exact(partial_sum, 1, term3, 1);

end
