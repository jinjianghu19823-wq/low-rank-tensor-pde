function [Z, info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact_4d( ...
    Y, preconditioner)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_EXACT_4D Apply P.
%
% This fixed linear action performs no recompression and is used before the
% Tucker sGMRES residual sketch.

callTimer = tic;
if ~isa(Y, 'ttensor') || ndims(Y) ~= 4
    error('Y must be an order four Tucker tensor.');
end
if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end

forwardFactors = cell(4, 1);
for mode = 1:4
    forwardFactors{mode} = dst(Y.u{mode});
end
spectralY = ttensor(Y.core, forwardFactors);
exactProduct = tucker_hadamard_exact_nd( ...
    preconditioner.inverse_eigenvalue_tensor, spectralY);
inverseFactors = cell(4, 1);
for mode = 1:4
    inverseFactors{mode} = idst(exactProduct.u{mode});
end
Z = ttensor(exactProduct.core, inverseFactors);

info.input_ranks = double(size(Y.core));
info.output_ranks = double(size(Z.core));
info.rounding_performed = false;
info.rank_cap_active = false;
info.relative_error_estimate = 0;
info.call_time_sec = toc(callTimer);

end
