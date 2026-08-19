function test_poisson_fast_diagonalization_tucker_preconditioner_4d
%TEST_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_4D Focused tests.

fprintf('\nStarting four dimensional fast diagonalisation tests.\n');
N = 5;
h = 1 / (N + 1);
e = ones(N, 1);
A1 = spdiags([-e, 2 * e, -e], -1:1, N, N) / h^2;
[P, buildInfo] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner_4d(A1, N);
assert(buildInfo.spectral_delta < 1e-10, ...
    'The full rank inverse multiplier is not exact.');

rng(17, 'twister');
factors = cell(4, 1);
for mode = 1:4
    [factors{mode}, ~] = qr(randn(N, 2), 0);
end
Y = ttensor(tensor(randn(2, 2, 2, 2)), factors);

[Zexact, exactInfo] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact_4d(Y, P);
[Zrounded, roundedInfo] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_4d( ...
    Y, P, 1e-13, [N, N, N, N]);

fullY = double(full(Y));
spectralY = apply_transform_all_modes(fullY, @dst);
lambda = P.one_dimensional_eigenvalues;
lambdaSum = reshape(lambda, N, 1, 1, 1) + ...
    reshape(lambda, 1, N, 1, 1) + ...
    reshape(lambda, 1, 1, N, 1) + reshape(lambda, 1, 1, 1, N);
fullReference = apply_transform_all_modes(spectralY ./ lambdaSum, @idst);

exactDifference = double(full(Zexact)) - fullReference;
roundedDifference = double(full(Zrounded)) - fullReference;
exactError = norm(exactDifference(:)) / norm(fullReference(:));
roundedError = norm(roundedDifference(:)) / norm(fullReference(:));
assert(exactError < 1e-11 && roundedError < 1e-10, ...
    'The four dimensional Tucker preconditioner action is inconsistent.');
assert(~exactInfo.rounding_performed, ...
    'The fixed linear preconditioner must not round.');
assert(~roundedInfo.rank_cap_active, ...
    'The full rank correctness case unexpectedly activated its cap.');

AY = poisson_action_tucker(Y, A1, 1e-13, [N, N, N, N]);
[PAY, ~] = apply_poisson_fast_diagonalization_tucker_preconditioner_4d( ...
    AY, P, 1e-12, [N, N, N, N]);
identityDifference = tucker_axpby_exact(PAY, 1, Y, -1);
identityError = norm(identityDifference) / norm(Y);
assert(identityError < 1e-9, ...
    'The full rank four dimensional preconditioner is not an inverse.');

fprintf('Exact action error %.3e\n', exactError);
fprintf('Rounded action error %.3e\n', roundedError);
fprintf('Preconditioned identity error %.3e\n', identityError);
fprintf('All four dimensional fast diagonalisation tests passed.\n\n');

end


function output = apply_transform_all_modes(input, transform)

output = input;
for mode = 1:4
    permutation = [mode, 1:(mode - 1), (mode + 1):4];
    permuted = permute(output, permutation);
    unfolded = reshape(permuted, size(output, mode), []);
    transformed = transform(unfolded);
    output = ipermute(reshape(transformed, size(permuted)), permutation);
end

end
