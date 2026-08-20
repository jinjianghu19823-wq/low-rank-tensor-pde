function [Z, info] = tucker_round( ...
    X, eta, max_rank)
%TUCKER_ROUND Recompress one Tucker tensor without forming its full array.

call_timer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if ~isscalar(eta) || eta <= 0 || eta >= 1
    error('eta must be one number between 0 and 1.');
end

if nargin < 3
    max_rank = [];
end

d = ndims(X);
max_ranks = normalise_tucker_rank_cap(max_rank, size(X));

factors = cell(d, 1);
core = X.core;
factor_qr_time = 0;
core_transform_time = 0;

for mode = 1:d

    component_timer = tic;
    [Q, R] = qr(X.u{mode}, 0);
    factor_qr_time = factor_qr_time + toc(component_timer);

    factors{mode} = Q;

    component_timer = tic;
    core = ttm(core, R, mode);
    core_transform_time = core_transform_time + toc(component_timer);

end

if norm(core) == 0
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
    kernel_timing.factor_qr_time_sec = factor_qr_time;
    kernel_timing.core_transform_time_sec = core_transform_time;
    info.kernel_timing = kernel_timing;
    info.call_time_sec = toc(call_timer);
    return
end

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
kernel_timing.factor_qr_time_sec = ...
    kernel_timing.factor_qr_time_sec + factor_qr_time;
kernel_timing.core_transform_time_sec = ...
    kernel_timing.core_transform_time_sec + core_transform_time;
kernel_timing.factor_reconstruction_time_sec = ...
    kernel_timing.factor_reconstruction_time_sec + factor_reconstruction_time;

info.kernel_timing = kernel_timing;
info.call_time_sec = toc(call_timer);

end
