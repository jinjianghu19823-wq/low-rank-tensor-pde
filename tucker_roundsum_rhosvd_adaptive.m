function [Z, info] = tucker_roundsum_rhosvd_adaptive( ...
    terms, coeff, oversampling, tolerance, max_rank_value, ...
    seed, check_error)
%TUCKER_ROUNDSUM_RHOSVD_ADAPTIVE Paper-style adaptive RHOSVD RoundSum.

if nargin < 7
    check_error = true;
end

adaptive_call_timer = tic;

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end
terms = terms(:);
coeff = double(coeff(:));

if numel(coeff) ~= numel(terms) || ...
        any(~isfinite(coeff))
    error('coeff must contain one finite value per Tucker term.');
end
if any(~cellfun(@(term) isa(term, 'ttensor'), terms))
    error('Every RoundSum term must be a Tensor Toolbox ttensor.');
end
if ~isscalar(oversampling) || oversampling < 0 || ...
        oversampling ~= floor(oversampling)
    error('oversampling must be a nonnegative integer.');
end
if ~isscalar(tolerance) || tolerance <= 0 || tolerance >= 1
    error('tolerance must be one number between 0 and 1.');
end
if ~isscalar(max_rank_value) || max_rank_value < 1 || ...
        max_rank_value ~= floor(max_rank_value)
    error('max_rank_value must be a positive integer.');
end

n = double(size(terms{1}));
d = numel(n);

for term_idx = 2:numel(terms)
    if ~isequal(double(size(terms{term_idx})), n)
        error('Every RoundSum term must have the same tensor dimensions.');
    end
end

rank_estimation_timer = tic;
effective_ranks = estimate_effective_ranks( ...
    terms, coeff, tolerance, d);
rank_estimation_time = toc(rank_estimation_timer);
common_range_size = min(max(effective_ranks), max_rank_value) + oversampling;
range_sizes = min(common_range_size, n);
max_ranks = min(max_rank_value, range_sizes);

[Z, fixed_info] = tucker_roundsum_rhosvd( ...
    terms, coeff, range_sizes, tolerance, max_ranks, ...
    seed, check_error);

info = fixed_info;
info.base_roundsum_algorithm_time_sec = fixed_info.algorithm_time_sec;
info.rank_estimation_time_sec = rank_estimation_time;
info.algorithm_time_sec = ...
    fixed_info.algorithm_time_sec + rank_estimation_time;
info.adaptive_call_time_sec = toc(adaptive_call_timer);
info.adaptive_rank_selection = true;
info.effective_rank_estimates = effective_ranks;
info.oversampling = oversampling;
info.maximum_rank_cap = max_rank_value;
info.common_range_size_before_mode_caps = common_range_size;

end

function effective_ranks = estimate_effective_ranks( ...
    terms, coeff, tolerance, d)
%ESTIMATE_EFFECTIVE_RANKS Reproduce the paper code's Gramian estimator.

effective_ranks = ones(1, d);

for mode_index = 1:d
    weighted_factors = [];

    for term_idx = 1:numel(terms)
        scaled_core_norm = abs(coeff(term_idx)) * ...
            norm(terms{term_idx}.core);
        weighted_factors = [weighted_factors, ...
            scaled_core_norm * terms{term_idx}.u{mode_index}]; %#ok<AGROW>
    end

    G = weighted_factors.' * weighted_factors;
    eigenvalues = sort(max(0, eig(G)), 'descend');

    if isempty(eigenvalues) || sum(eigenvalues) == 0
        effective_ranks(mode_index) = 1;
        continue
    end

    discarded_energy = cumsum(eigenvalues, 'reverse');
    threshold = tolerance^2 * sum(eigenvalues) / d;
    retained_rank = find(discarded_energy > threshold, 1, 'last');

    if isempty(retained_rank)
        retained_rank = 1;
    end
    effective_ranks(mode_index) = retained_rank;
end

end
