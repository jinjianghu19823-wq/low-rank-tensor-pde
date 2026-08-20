function test_tucker_sgmres_foundation
%TEST_TUCKER_SGMRES_FOUNDATION Validate the structured residual sketch.

add_toolboxes();
rng(17, 'twister');

n = [3, 4, 2];
sketch_size = 11;
seed = 29;

Xfull = tensor(randn(n));
Yfull = tensor(randn(n));
X = hosvd(Xfull, 1e-13, 'verbosity', 0);
Y = hosvd(Yfull, 1e-13, 'verbosity', 0);

sketch = create_tucker_row_khatri_rao_sketch( ...
    n, sketch_size, seed);
same_sketch = create_tucker_row_khatri_rao_sketch( ...
    n, sketch_size, seed);

for mode = 1:numel(n)
    assert(isequal(sketch.factors{mode}, same_sketch.factors{mode}), ...
        'The same seed did not reproduce the sketch factors.');
end

explicit_matrix = form_explicit_sketch_matrix(sketch);
Xvalues = double(full(X));
explicit_values = explicit_matrix * Xvalues(:);
implicit_values = apply_tucker_row_khatri_rao_sketch(X, sketch);
relative_application_error = ...
    norm(implicit_values - explicit_values) / norm(explicit_values);

assert(relative_application_error < 1e-12, ...
    'The implicit Tucker sketch does not match S*vec(X).');

alpha = 1.7;
beta = -0.4;
Z = tucker_axpby_exact(X, alpha, Y, beta);
sketch_z = apply_tucker_row_khatri_rao_sketch(Z, sketch);
linear_combination = alpha * ...
    apply_tucker_row_khatri_rao_sketch(X, sketch) + beta * ...
    apply_tucker_row_khatri_rao_sketch(Y, sketch);
relative_linearity_error = ...
    norm(sketch_z - linear_combination) / norm(linear_combination);

assert(relative_linearity_error < 1e-12, ...
    'The Tucker sketch failed its linearity check.');

fprintf('\nTucker sGMRES sketch foundation test\n');
fprintf('Explicit application relative error  %.3e\n', ...
    relative_application_error);
fprintf('Linearity relative error             %.3e\n', ...
    relative_linearity_error);

N = 6;
h = 1 / (N + 1);
e = ones(N, 1);
A1 = spdiags( ...
    [-e, 2 * e, -e], -1:1, N, N) / h^2;
grid_points = (1:N).' / (N + 1);
g = exp(-10 * (grid_points - 1/2).^2);
F = ttensor(tensor(1, [1, 1, 1]), ...
    {g, g, g});
U0 = 0 * F;
norm_f = norm(F);
A_terms = @(U) poisson_action_tucker_terms(U, A1);
original_residual_function = @(U) ...
    true_residual_tucker_poisson(U, F, A1);
identity_residual_norm_function = @(U) ...
    norm_f * true_residual_tucker_poisson(U, F, A1);
solver_sketch = create_tucker_row_khatri_rao_sketch( ...
    [N, N, N], 20, 101);

[~, solver_info] = tucker_sgmres_left_preconditioned_plain( ...
    A_terms, @identity_tucker_fixed_linear_preconditioner, ...
    F, U0, ...
    4, solver_sketch, 1e-14, 2, 1e-10, 1e-10, ...
    original_residual_function, identity_residual_norm_function, ...
    false, Inf, [N, N, N], [2; 4]);

assert(solver_info.iterations == 4, ...
    'The plain Tucker sGMRES smoke test ended before four steps.');
assert(solver_info.stop_reason == "iteration_limit", ...
    'The plain Tucker sGMRES smoke test stopped unexpectedly.');
assert(isfinite(solver_info.true_relative_residual(end)), ...
    'The independently evaluated final residual is not finite.');
assert(solver_info.true_relative_residual(end) < ...
    solver_info.true_relative_residual(1), ...
    'The independently evaluated residual did not decrease.');
assert(~solver_info.rank_cap_active, ...
    'The rank cap should not be active in the smoke test.');
assert(isequal(solver_info.true_residual_iteration, [0; 2; 4]), ...
    'The requested true-residual checkpoints were not recorded.');
assert(numel(solver_info.true_relative_residual) == 3 && ...
    all(isfinite(solver_info.true_relative_residual)), ...
    'The checkpoint true-residual history is incomplete.');

fprintf('Four step solver true residual       %.3e\n', ...
    solver_info.true_relative_residual(end));
fprintf('Four step sketch residual            %.3e\n', ...
    solver_info.computed_sketch_relative_residual(end));
fprintf('All structured residual sketch checks passed.\n\n');

end

function explicit_matrix = form_explicit_sketch_matrix(sketch)
%FORM_EXPLICIT_SKETCH_MATRIX Construct S only for the tiny test.

sketch_size = sketch.sketch_size;
n = sketch.tensor_dimensions;
d = sketch.number_of_modes;
explicit_matrix = zeros(sketch_size, prod(n));

for row_idx = 1:sketch_size
    one_row = sketch.factors{1}(row_idx, :);
    for mode = 2:d
        one_row = kron(sketch.factors{mode}(row_idx, :), one_row);
    end
    explicit_matrix(row_idx, :) = one_row;
end

end
