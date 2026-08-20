function Z = tucker_axpby_exact(X, alpha, Y, beta)
%TUCKER_AXPBY_EXACT Add two Tucker tensors without truncation.

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(X), size(Y))
    error('X and Y must have the same tensor dimensions.');
end

if ~isscalar(alpha) || ~isscalar(beta)
    error('alpha and beta must be scalar numbers.');
end

d = ndims(X);

rank_x = size(X.core);
rank_y = size(Y.core);
combined_core_size = rank_x + rank_y;

core_values = zeros(combined_core_size);

index_x = cell(d, 1);
index_y = cell(d, 1);

for mode = 1:d

    index_x{mode} = 1:rank_x(mode);
    index_y{mode} = rank_x(mode) + (1:rank_y(mode));

end

core_values(index_x{:}) = alpha * double(X.core);
core_values(index_y{:}) = beta * double(Y.core);

combined_core = tensor(core_values);

combined_factors = cell(d, 1);

for mode = 1:d
    combined_factors{mode} = [X.u{mode}, Y.u{mode}];
end

% Orthonormalise the joined factors

factors = cell(d, 1);
core = combined_core;

for mode = 1:d

    [Q, R] = qr(combined_factors{mode}, 0);

    factors{mode} = Q;
    core = ttm(core, R, mode);

end

Z = ttensor(core, factors);

end
