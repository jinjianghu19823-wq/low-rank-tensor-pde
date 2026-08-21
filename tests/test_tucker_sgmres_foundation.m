function test_tucker_sgmres_foundation
%TEST_TUCKER_SGMRES_FOUNDATION Validate the structured residual sketch.
%
% This correctness test forms the otherwise implicit full sketch on a tiny
% tensor. It checks the vectorisation order, Tucker contraction, linearity,
% and reproducibility. It is not a performance experiment.

add_toolboxes();
rng(17, 'twister');

tensorDimensions = [3, 4, 2];
sketchSize = 11;
randomSeed = 29;

Xfull = tensor(randn(tensorDimensions));
Yfull = tensor(randn(tensorDimensions));
X = hosvd(Xfull, 1e-13, 'verbosity', 0);
Y = hosvd(Yfull, 1e-13, 'verbosity', 0);

sketch = create_tucker_row_khatri_rao_sketch( ...
    tensorDimensions, sketchSize, randomSeed);
sameSketch = create_tucker_row_khatri_rao_sketch( ...
    tensorDimensions, sketchSize, randomSeed);

for mode = 1:numel(tensorDimensions)
    assert(isequal(sketch.factors{mode}, sameSketch.factors{mode}), ...
        'The same seed did not reproduce the sketch factors.');
end

explicitMatrix = form_explicit_sketch_matrix(sketch);
Xvalues = double(full(X));
explicitValues = explicitMatrix * Xvalues(:);
implicitValues = apply_tucker_row_khatri_rao_sketch(X, sketch);
relativeApplicationError = ...
    norm(implicitValues - explicitValues) / norm(explicitValues);

assert(relativeApplicationError < 1e-12, ...
    'The implicit Tucker sketch does not match S*vec(X).');

alpha = 1.7;
beta = -0.4;
Z = tucker_axpby_exact(X, alpha, Y, beta);
sketchZ = apply_tucker_row_khatri_rao_sketch(Z, sketch);
linearCombination = alpha * ...
    apply_tucker_row_khatri_rao_sketch(X, sketch) + beta * ...
    apply_tucker_row_khatri_rao_sketch(Y, sketch);
relativeLinearityError = ...
    norm(sketchZ - linearCombination) / norm(linearCombination);

assert(relativeLinearityError < 1e-12, ...
    'The Tucker sketch failed its linearity check.');

fprintf('\nTucker sGMRES sketch foundation test\n');
fprintf('Explicit application relative error  %.3e\n', ...
    relativeApplicationError);
fprintf('Linearity relative error             %.3e\n', ...
    relativeLinearityError);

% A small end to end smoke test checks that the plain solver completes its
% declared work and reduces an independently evaluated Poisson residual.
N = 6;
h = 1 / (N + 1);
onesVector = ones(N, 1);
A1 = spdiags( ...
    [-onesVector, 2 * onesVector, -onesVector], -1:1, N, N) / h^2;
gridPoints = (1:N).' / (N + 1);
gaussianVector = exp(-10 * (gridPoints - 1/2).^2);
F = ttensor(tensor(1, [1, 1, 1]), ...
    {gaussianVector, gaussianVector, gaussianVector});
U0 = 0 * F;
normF = norm(F);
operatorTermsFunction = @(U) poisson_action_tucker_terms(U, A1);
originalResidualFunction = @(U) ...
    true_residual_tucker_poisson(U, F, A1);
identityResidualNormFunction = @(U) ...
    normF * true_residual_tucker_poisson(U, F, A1);
solverSketch = create_tucker_row_khatri_rao_sketch( ...
    [N, N, N], 20, 101);

[~, solverInfo] = tucker_sgmres_left_preconditioned_plain( ...
    operatorTermsFunction, @identity_tucker_fixed_linear_preconditioner, ...
    F, U0, ...
    4, solverSketch, 1e-14, 2, 1e-10, 1e-10, ...
    originalResidualFunction, identityResidualNormFunction, ...
    false, Inf, [N, N, N], [2; 4]);

assert(solverInfo.iterations == 4, ...
    'The plain Tucker sGMRES smoke test ended before four steps.');
assert(solverInfo.stop_reason == "iteration_limit", ...
    'The plain Tucker sGMRES smoke test stopped unexpectedly.');
assert(isfinite(solverInfo.true_relative_residual(end)), ...
    'The independently evaluated final residual is not finite.');
assert(solverInfo.true_relative_residual(end) < ...
    solverInfo.true_relative_residual(1), ...
    'The independently evaluated residual did not decrease.');
assert(~solverInfo.rank_cap_active, ...
    'The rank cap should not be active in the smoke test.');
assert(isequal(solverInfo.true_residual_iteration, [0; 2; 4]), ...
    'The requested true-residual checkpoints were not recorded.');
assert(numel(solverInfo.true_relative_residual) == 3 && ...
    all(isfinite(solverInfo.true_relative_residual)), ...
    'The checkpoint true-residual history is incomplete.');

fprintf('Four step solver true residual       %.3e\n', ...
    solverInfo.true_relative_residual(end));
fprintf('Four step sketch residual            %.3e\n', ...
    solverInfo.computed_sketch_relative_residual(end));
fprintf('All structured residual sketch checks passed.\n\n');

end


function explicitMatrix = form_explicit_sketch_matrix(sketch)
%FORM_EXPLICIT_SKETCH_MATRIX Construct S only for the tiny test.

sketchSize = sketch.sketch_size;
tensorDimensions = sketch.tensor_dimensions;
numberOfModes = sketch.number_of_modes;
explicitMatrix = zeros(sketchSize, prod(tensorDimensions));

for rowIndex = 1:sketchSize
    oneRow = sketch.factors{1}(rowIndex, :);
    for mode = 2:numberOfModes
        oneRow = kron(sketch.factors{mode}(rowIndex, :), oneRow);
    end
    explicitMatrix(rowIndex, :) = oneRow;
end

end
