function test_matrix_tensor_algorithms
%TEST_MATRIX_TENSOR_ALGORITHMS Smoke-test Algorithms 3.1--3.7.

add_toolboxes();
rng(20260821, 'twister');

A = randn(24, 16);
targetRank = 5;
oversampling = 3;

[U, S, V] = rsvd(A, targetRank, oversampling);
assert(isequal(size(U), [24, targetRank]));
assert(isequal(size(S), [targetRank, targetRank]));
assert(isequal(size(V), [16, targetRank]));
assert(all(isfinite(U * S * V'), 'all'));

[L, M] = generalized_nystrom(A, targetRank, oversampling, false);
assert(isequal(size(L), [24, targetRank]));
assert(isequal(size(M), [targetRank, 16]));
assert(all(isfinite(L * M), 'all'));

X = tensor(randn(6, 5, 4));
ranks = [2, 2, 2];
order = [1, 2, 3];

[~, ~, Xhosvd] = truncated_hosvd(X, ranks);
[~, ~, Xsthosvd] = truncated_sthosvd(X, ranks, order);
[~, ~, Xrhosvd] = r_hosvd(X, ranks, 1);
[~, ~, Xrsthosvd] = r_sthosvd(X, ranks, 1, order);
[~, ~, Xmln] = multilinear_nystrom(X, ranks, [1, 1, 1]);

approximations = {Xhosvd, Xsthosvd, Xrhosvd, Xrsthosvd, Xmln};
for index = 1:numel(approximations)
    approximation = double(approximations{index});
    assert(isequal(size(approximation), size(X)));
    assert(all(isfinite(approximation), 'all'));
end

fprintf('Matrix and tensor algorithm smoke tests passed.\n');

end
