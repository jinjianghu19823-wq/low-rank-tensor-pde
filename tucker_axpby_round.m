function [Z, info] = tucker_axpby_round(X, alpha, Y, beta, eta, max_rank)
%TUCKER_AXPBY_ROUND Add two Tucker tensors and recompress the result.

call_timer = tic;

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(X), size(Y))
    error('X and Y must have the same tensor dimensions.');
end

if ~isscalar(alpha) || ~isscalar(beta)
    error('alpha and beta must be scalar numbers.');
end

if eta <= 0 || eta >= 1
    error('eta must be between 0 and 1.');
end

if nargin < 6
    max_rank = [];
end

max_ranks = normalise_tucker_rank_cap(max_rank, size(X));

d = ndims(X);

rank_x = size(X.core);
rank_y = size(Y.core);

combined_core_size = rank_x + rank_y;

component_timer = tic;

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
exact_sum_core_time = toc(component_timer);

component_timer = tic;
combined_factors = cell(d, 1);

for mode = 1:d

    combined_factors{mode} = [X.u{mode}, Y.u{mode}];

end

factor_concatenation_time = toc(component_timer);

% Orthonormalise the joined factor matrices

factors = cell(d, 1);
core = combined_core;
factor_qr_time = 0;
core_transform_time = 0;

for mode = 1:d

    component_timer = tic;
    [Q, R] = qr(combined_factors{mode}, 0);
    factor_qr_time = factor_qr_time + toc(component_timer);

    factors{mode} = Q;

    component_timer = tic;
    core = ttm(core, R, mode);
    core_transform_time = core_transform_time + toc(component_timer);

end

size_before_cancellation = abs(alpha) * norm(X) + abs(beta) * norm(Y);

size_after_cancellation = norm(core);

if size_after_cancellation <= 100 * eps * size_before_cancellation

    Z = 0 * X;

    info.requested_tolerance = eta;
    info.maximum_ranks = max_ranks;
    info.tolerance_ranks = size(X.core);
    info.retained_ranks = size(X.core);
    info.relative_error_estimate = 0;
    info.rank_cap_active_by_mode = false(1, d);
    info.rank_cap_active = false;
    info.zero_result = true;
    kernel_timing = empty_tucker_kernel_timing();
    kernel_timing.exact_sum_core_time_sec = exact_sum_core_time;
    kernel_timing.factor_concatenation_time_sec = ...
        factor_concatenation_time;
    kernel_timing.factor_qr_time_sec = factor_qr_time;
    kernel_timing.core_transform_time_sec = core_transform_time;
    info.kernel_timing = kernel_timing;
    info.call_time_sec = toc(call_timer);
    return

end

% Compress the small orthogonal core with STHOSVD

[compressed_core_tensor, core_info] = sthosvd_round_tensor(core, eta, max_ranks, 1:d);

final_factors = cell(d, 1);
factor_reconstruction_time = 0;

for mode = 1:d

    component_timer = tic;
    final_factors{mode} = factors{mode} * compressed_core_tensor.u{mode};
    factor_reconstruction_time = ...
        factor_reconstruction_time + toc(component_timer);

end

Z = ttensor(compressed_core_tensor.core, final_factors);

info = core_info;
info.zero_result = false;

kernel_timing = core_info.kernel_timing;
kernel_timing.exact_sum_core_time_sec = ...
    kernel_timing.exact_sum_core_time_sec + exact_sum_core_time;
kernel_timing.factor_concatenation_time_sec = ...
    kernel_timing.factor_concatenation_time_sec + factor_concatenation_time;
kernel_timing.factor_qr_time_sec = ...
    kernel_timing.factor_qr_time_sec + factor_qr_time;
kernel_timing.core_transform_time_sec = ...
    kernel_timing.core_transform_time_sec + core_transform_time;
kernel_timing.factor_reconstruction_time_sec = ...
    kernel_timing.factor_reconstruction_time_sec + factor_reconstruction_time;

info.kernel_timing = kernel_timing;
info.call_time_sec = toc(call_timer);

end
