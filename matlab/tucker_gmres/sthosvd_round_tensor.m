function [T, info] = sthosvd_round_tensor(X, tol, max_rank, mode_order)
%STHOSVD_ROUND_TENSOR Recompress a small tensor with tolerance and rank cap.

call_timer = tic;

if ~isa(X, 'tensor')
    error('X must be a Tensor Toolbox tensor object.');
end

if ~isscalar(tol) || tol <= 0 || tol >= 1
    error('tol must be one number between 0 and 1.');
end

d = ndims(X);
n = size(X);

if nargin < 3
    max_rank = [];
end

if nargin < 4 || isempty(mode_order)
    mode_order = 1:d;
end

mode_order = double(mode_order(:).');

if ~isequal(sort(mode_order), 1:d)
    error('mode_order must contain every tensor mode exactly once.');
end

max_ranks = normalise_tucker_rank_cap(max_rank, n);

norm_x = norm(X);
mode_error_threshold = tol * norm_x / sqrt(d);
mode_error_budget = mode_error_threshold^2;

factor_matrices = cell(d, 1);
tolerance_ranks = zeros(1, d);
retained_ranks = zeros(1, d);
discarded_norms = zeros(1, d);
discarded_energy = zeros(1, d);
rank_cap_active_by_mode = false(1, d);
unfolding_time_by_mode = zeros(1, d);
svd_time_by_mode = zeros(1, d);
rank_selection_time_by_mode = zeros(1, d);
projection_time_by_mode = zeros(1, d);

working_core = X;

for order_index = 1:d

    mode = mode_order(order_index);

    component_timer = tic;
    unfolding = double(tenmat(working_core, mode));
    unfolding_time_by_mode(mode) = toc(component_timer);

    % The left singular vectors span the current mode subspace.
    component_timer = tic;
    [left_vectors, singular_value_matrix, ~] = svd(unfolding, 'econ');
    svd_time_by_mode(mode) = toc(component_timer);

    component_timer = tic;
    svals = diag(singular_value_matrix);

    tail_norm_after_rank = zeros(size(svals));

    for rank_index = numel(svals)-1:-1:1
        tail_norm_after_rank(rank_index) = hypot(svals(rank_index + 1), tail_norm_after_rank(rank_index + 1));
    end

    tolerance_rank = find(tail_norm_after_rank <= mode_error_threshold, 1, 'first');

    if isempty(tolerance_rank)
        tolerance_rank = numel(svals);
    end

    retained_rank = min(tolerance_rank, max_ranks(mode));

    tolerance_ranks(mode) = tolerance_rank;
    retained_ranks(mode) = retained_rank;
    rank_cap_active_by_mode(mode) = retained_rank < tolerance_rank;
    discarded_norms(mode) = tail_norm_after_rank(retained_rank);
    discarded_energy(mode) = discarded_norms(mode)^2;

    factor_matrices{mode} = left_vectors(:, 1:retained_rank);
    rank_selection_time_by_mode(mode) = toc(component_timer);

    component_timer = tic;
    working_core = ttm(working_core, factor_matrices{mode}.', mode);
    projection_time_by_mode(mode) = toc(component_timer);

end

T = ttensor(working_core, factor_matrices);

if norm_x == 0
    relative_error_estimate = 0;
else
    relative_error_estimate = norm(discarded_norms) / norm_x;
end

info.requested_tolerance = tol;
info.mode_error_threshold = mode_error_threshold;
info.mode_error_budget = mode_error_budget;
info.maximum_ranks = max_ranks;
info.tolerance_ranks = tolerance_ranks;
info.retained_ranks = retained_ranks;
info.discarded_norm = discarded_norms;
info.discarded_energy = discarded_energy;
info.relative_error_estimate = relative_error_estimate;
info.rank_cap_active_by_mode = rank_cap_active_by_mode;
info.rank_cap_active = any(rank_cap_active_by_mode);
info.mode_order = mode_order;

kernel_timing = empty_tucker_kernel_timing();
kernel_timing.sthosvd_unfolding_time_sec = sum(unfolding_time_by_mode);
kernel_timing.sthosvd_svd_time_sec = sum(svd_time_by_mode);
kernel_timing.sthosvd_rank_selection_time_sec = ...
    sum(rank_selection_time_by_mode);
kernel_timing.sthosvd_projection_time_sec = sum(projection_time_by_mode);

info.kernel_timing = kernel_timing;
info.call_time_sec = toc(call_timer);
info.unfolding_time_by_mode_sec = unfolding_time_by_mode;
info.svd_time_by_mode_sec = svd_time_by_mode;
info.rank_selection_time_by_mode_sec = rank_selection_time_by_mode;
info.projection_time_by_mode_sec = projection_time_by_mode;

end
