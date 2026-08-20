function [Y, info] = poisson_action_tucker(X, A1, eta, max_rank)
%POISSON_ACTION_TUCKER Apply the d-dimensional Poisson operator.

call_timer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

if eta <= 0 || eta >= 1
    error('eta must be between 0 and 1.');
end

if nargin < 4
    max_rank = [];
end

max_ranks = normalise_tucker_rank_cap(max_rank, size(X));

d = ndims(X);
tensor_size = size(X);

for mode = 1:d

    if tensor_size(mode) ~= size(A1, 1)
        error('Every mode of X must have the same size as A1.');
    end

end

component_timer = tic;
Y = ttm(X, A1, 1);
mode_product_time = toc(component_timer);

kernel_timing = empty_tucker_kernel_timing();

addition_cap_active = false(d - 1, 1);
addition_error_estimates = zeros(d - 1, 1);

for mode = 2:d

    component_timer = tic;
    next_term = ttm(X, A1, mode);
    mode_product_time = mode_product_time + toc(component_timer);

    [Y, add_info] = tucker_axpby_round(Y, 1, next_term, 1, eta, max_ranks);

    kernel_timing = add_tucker_kernel_timing( ...
        kernel_timing, add_info.kernel_timing);

    addition_cap_active(mode - 1) = add_info.rank_cap_active;
    addition_error_estimates(mode - 1) = ...
        add_info.relative_error_estimate;

end

info.requested_tolerance = eta;
info.maximum_ranks = max_ranks;
info.addition_rank_cap_active = addition_cap_active;
info.addition_relative_error_estimate = addition_error_estimates;
info.rank_cap_active = any(addition_cap_active);
info.maximum_local_recompression_error = ...
    max([0; addition_error_estimates]);
kernel_timing.poisson_mode_product_time_sec = ...
    kernel_timing.poisson_mode_product_time_sec + mode_product_time;
info.kernel_timing = kernel_timing;
info.call_time_sec = toc(call_timer);

end
