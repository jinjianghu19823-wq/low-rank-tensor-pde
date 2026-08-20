function [Z, info] = apply_poisson_fast_diagonalization_tucker_preconditioner(Y, preconditioner, eta, max_rank)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Apply M_r.

call_timer = tic;

if ~isa(Y, 'ttensor')
    error('Y must be a Tensor Toolbox ttensor object.');
end

if ndims(Y) ~= 3
    error('The fast-diagonalization preconditioner requires order 3.');
end

if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end

if nargin < 4
    max_rank = [];
end

% Transform the Tucker factors into the sine basis

forward_factors = cell(3, 1);
component_timer = tic;

for mode = 1:3
    forward_factors{mode} = dst(Y.u{mode});
end

forward_dst_time = toc(component_timer);

spectral_y = ttensor(Y.core, forward_factors);

component_timer = tic;
exact_spectral_product = tucker_hadamard_exact(preconditioner.inverse_eigenvalue_tensor, spectral_y);
exact_hadamard_time = toc(component_timer);
temporary_ranks = size(exact_spectral_product.core);

[rounded_spectral_product, round_info] = tucker_round(exact_spectral_product, eta, max_rank);

inverse_factors = cell(3, 1);
component_timer = tic;

for mode = 1:3
    inverse_factors{mode} = idst(rounded_spectral_product.u{mode});
end

inverse_dst_time = toc(component_timer);

Z = ttensor(rounded_spectral_product.core, inverse_factors);

info.requested_tolerance = eta;
info.inverse_eigenvalue_ranks = ...
    preconditioner.inverse_eigenvalue_ranks;
info.input_ranks = size(Y.core);
info.temporary_hadamard_ranks = temporary_ranks;
info.output_ranks = size(Z.core);
info.rank_cap_active = round_info.rank_cap_active;
info.relative_error_estimate = round_info.relative_error_estimate;

kernel_timing = round_info.kernel_timing;
kernel_timing.preconditioner_forward_dst_time_sec = ...
    kernel_timing.preconditioner_forward_dst_time_sec + forward_dst_time;
kernel_timing.preconditioner_exact_hadamard_time_sec = ...
    kernel_timing.preconditioner_exact_hadamard_time_sec + exact_hadamard_time;
kernel_timing.preconditioner_inverse_dst_time_sec = ...
    kernel_timing.preconditioner_inverse_dst_time_sec + inverse_dst_time;

info.kernel_timing = kernel_timing;
info.call_time_sec = toc(call_timer);

end
