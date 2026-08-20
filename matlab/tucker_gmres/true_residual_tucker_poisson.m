function relative_residual = true_residual_tucker_poisson(U, F, A1)
%TRUE_RESIDUAL_TUCKER_POISSON Calculate the true relative residual.

if ~isa(U, 'ttensor') || ~isa(F, 'ttensor')
    error('U and F must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(U), size(F))
    error('U and F must have the same tensor dimensions.');
end

norm_f = norm(F);

if norm_f == 0
    error('The right-hand side F must be nonzero.');
end

d = ndims(U);

residual_tensor = F;

for mode = 1:d

    one_mode_contribution = ttm(U, A1, mode);

    residual_tensor = tucker_axpby_exact(residual_tensor, 1, one_mode_contribution, -1);

end

relative_residual = norm(residual_tensor) / norm_f;

end
