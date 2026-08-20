function [Z, info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact( ...
    Y, preconditioner)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_EXACT Apply P.

call_timer = tic;

if ~isa(Y, 'ttensor')
    error('Y must be a Tensor Toolbox ttensor object.');
end

if ndims(Y) ~= 3
    error('The fast diagonalization preconditioner requires order 3.');
end

if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end

forward_factors = cell(3, 1);
component_timer = tic;

for mode = 1:3
    forward_factors{mode} = dst(Y.u{mode});
end

forward_dst_time = toc(component_timer);
spectral_y = ttensor(Y.core, forward_factors);

component_timer = tic;
exact_spectral_product = tucker_hadamard_exact( ...
    preconditioner.inverse_eigenvalue_tensor, spectral_y);
exact_hadamard_time = toc(component_timer);

inverse_factors = cell(3, 1);
component_timer = tic;

for mode = 1:3
    inverse_factors{mode} = idst(exact_spectral_product.u{mode});
end

inverse_dst_time = toc(component_timer);
Z = ttensor(exact_spectral_product.core, inverse_factors);

info.input_ranks = size(Y.core);
info.output_ranks = size(Z.core);
info.rounding_performed = false;
info.rank_cap_active = false;
info.relative_error_estimate = 0;
info.forward_dst_time_sec = forward_dst_time;
info.exact_hadamard_time_sec = exact_hadamard_time;
info.inverse_dst_time_sec = inverse_dst_time;
info.call_time_sec = toc(call_timer);

end
