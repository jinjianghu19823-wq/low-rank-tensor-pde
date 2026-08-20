function [Z, info] = tucker_roundsum_rhosvd( ...
    terms, coeff, range_sizes, eta, ...
    max_rank, seed, check_error)
%TUCKER_ROUNDSUM_RHOSVD Approximate a weighted Tucker sum by RHOSVD.

if nargin < 7
    check_error = true;
end

if ~islogical(check_error) || ...
        ~isscalar(check_error)
    error('check_error must be one logical value.');
end

call_timer = tic;

[terms, coeff, n] = ...
    check_roundsum_inputs(terms, coeff);

d = numel(n);
range_sizes = expand_mode_vector( ...
    range_sizes, d, 'range_sizes');

if any(range_sizes < 1) || ...
        any(range_sizes ~= floor(range_sizes)) || ...
        any(range_sizes > n)
    error(['Every range sketch size must be a positive integer no ', ...
           'larger than its tensor mode.']);
end

if ~isscalar(eta) || eta <= 0 || ...
        eta >= 1
    error('eta must be one number between 0 and 1.');
end

if isempty(max_rank)
    max_ranks = range_sizes;
else
    max_ranks = normalise_tucker_rank_cap( ...
        max_rank, n);
end

if any(max_ranks > range_sizes)
    error('Every RoundSum output rank R_n must satisfy R_n <= k_n.');
end

if ~isscalar(seed) || ~isfinite(seed) || ...
        seed < 0 || seed ~= floor(seed)
    error('seed must be a nonnegative integer.');
end

[norm_x, input_norm_info] = ...
    tucker_weighted_sum_norm(terms, coeff);
input_scale = 0;

for term_idx = 1:numel(terms)
    input_scale = input_scale + ...
        abs(coeff(term_idx)) * norm(terms{term_idx});
end

zero_threshold = 100 * eps * max(1, input_scale);

if norm_x <= zero_threshold
    Z = 0 * terms{1};
    info = zero_roundsum_info( ...
        norm_x, input_norm_info, range_sizes, max_ranks, ...
        eta, seed, d, ...
        check_error, call_timer);
    return
end

% Draw fresh standard Gaussian mode range maps

map_timer = tic;
maps = create_tucker_roundsum_maps( ...
    n, range_sizes, seed);
map_generation_time = toc(map_timer);

% Form the mode range sketches term by term

range_bases = cell(d, 1);
range_sketch_numerical_ranks = zeros(1, d);
range_sketch_time_by_mode = zeros(1, d);
range_qr_time_by_mode = zeros(1, d);

for target_mode = 1:d
    component_timer = tic;
    one_range_sketch = zeros( ...
        n(target_mode), range_sizes(target_mode));

    for term_idx = 1:numel(terms)
        one_term = terms{term_idx};

        for sketch_column = 1:range_sizes(target_mode)
            contracted_core = one_term.core;

            for source_mode = 1:d
                if source_mode ~= target_mode
                    projected_vector = one_term.u{source_mode}.' * ...
                        maps.factors{target_mode}{source_mode}( ...
                        :, sketch_column);
                    contracted_core = ttm( ...
                        contracted_core, projected_vector.', source_mode);
                end
            end

            mode_vector = reshape(double(contracted_core), [], 1);
            one_range_sketch(:, sketch_column) = ...
                one_range_sketch(:, sketch_column) + ...
                coeff(term_idx) * ...
                (one_term.u{target_mode} * mode_vector);
        end
    end

    range_sketch_time_by_mode(target_mode) = toc(component_timer);
    range_sketch_numerical_ranks(target_mode) = ...
        numerical_matrix_rank(one_range_sketch);

    component_timer = tic;
    [range_bases{target_mode}, ~] = qr(one_range_sketch, 0);
    range_qr_time_by_mode(target_mode) = toc(component_timer);
end

% Form and compress the small projected core

component_timer = tic;
projected_core_values = zeros(range_sizes);

for term_idx = 1:numel(terms)
    projected_term_core = terms{term_idx}.core;

    for mode = 1:d
        projected_term_core = ttm( ...
            projected_term_core, ...
            range_bases{mode}.' * terms{term_idx}.u{mode}, mode);
    end

    projected_core_values = projected_core_values + ...
        coeff(term_idx) * double(projected_term_core);
end

projected_core = tensor(projected_core_values);
projected_core_time = toc(component_timer);

component_timer = tic;
[compressed_core_tensor, compression_info] = sthosvd_round_tensor( ...
    projected_core, eta, max_ranks, 1:d);
projected_core_compression_time = toc(component_timer);

component_timer = tic;
output_factors = cell(d, 1);

for mode = 1:d
    output_factors{mode} = ...
        range_bases{mode} * compressed_core_tensor.u{mode};
end

Z = ttensor(compressed_core_tensor.core, output_factors);
factor_reconstruction_time = toc(component_timer);

algorithm_time = toc(call_timer);
if check_error
    error_diagnostic_timer = tic;
    projected_tensor = ttensor(projected_core, range_bases);
    [range_error, ~] = tucker_weighted_sum_norm( ...
        [terms; {projected_tensor}], [coeff; -1]);
    [total_error, ~] = tucker_weighted_sum_norm( ...
        [terms; {Z}], [coeff; -1]);

    if norm_x == 0
        relative_range_error = 0;
        relative_total_error = 0;
    else
        relative_range_error = range_error / norm_x;
        relative_total_error = total_error / norm_x;
    end

    compressed_projected_core = full(compressed_core_tensor);
    core_compression_error = norm(projected_core - compressed_projected_core);
    error_diagnostic_time = toc(error_diagnostic_timer);
else
    range_error = NaN;
    relative_range_error = NaN;
    core_compression_error = NaN;
    total_error = NaN;
    relative_total_error = NaN;
    error_diagnostic_time = 0;
end

info.input_formal_norm = norm_x;
info.input_norm_gram_matrix = input_norm_info.gram_matrix;
info.number_of_terms = numel(terms);
info.range_sketch_sizes = range_sizes;
info.maximum_ranks = max_ranks;
info.requested_tolerance = eta;
info.random_seed = seed;
info.maps_are_new_for_call = true;
info.error_diagnostics_measured = check_error;
info.map_distribution = maps.distribution;
info.map_factor_entry_mean = maps.factor_entry_mean;
info.map_factor_entry_variance = maps.factor_entry_variance;
info.range_sketch_numerical_ranks = range_sketch_numerical_ranks;
info.range_error_norm = range_error;
info.relative_range_error = relative_range_error;
info.core_compression_error_norm = core_compression_error;
info.total_error_norm = total_error;
info.relative_total_error = relative_total_error;
info.output_ranks = size(Z.core);
info.rank_cap_active = compression_info.rank_cap_active;
info.sthosvd = compression_info;
info.map_generation_time_sec = map_generation_time;
info.range_sketch_time_by_mode_sec = range_sketch_time_by_mode;
info.range_qr_time_by_mode_sec = range_qr_time_by_mode;
info.projected_core_time_sec = projected_core_time;
info.projected_core_compression_time_sec = ...
    projected_core_compression_time;
info.factor_reconstruction_time_sec = factor_reconstruction_time;
info.algorithm_time_sec = algorithm_time;
info.error_diagnostic_time_sec = error_diagnostic_time;
info.call_time_sec = toc(call_timer);

end

function [terms, coeff, n] = ...
    check_roundsum_inputs(terms, coeff)
%CHECK_ROUNDSUM_INPUTS Validate the weighted Tucker collection.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end

terms = terms(:);
coeff = double(coeff(:));

if numel(coeff) ~= numel(terms) || ...
        any(~isfinite(coeff))
    error('coeff must contain one finite value per Tucker term.');
end

if ~isa(terms{1}, 'ttensor')
    error('Every RoundSum term must be a Tensor Toolbox ttensor.');
end

n = double(size(terms{1}));

for term_idx = 1:numel(terms)
    if ~isa(terms{term_idx}, 'ttensor') || ...
            ~isequal(double(size(terms{term_idx})), n)
        error('Every RoundSum term must be a same-sized ttensor.');
    end
end

end

function values = expand_mode_vector(values, d, arg_name)
%EXPAND_MODE_VECTOR Expand a scalar to one value per tensor mode.

values = double(values(:).');

if isscalar(values)
    values = repmat(values, 1, d);
elseif numel(values) ~= d
    error('%s must be scalar or contain one value per mode.', arg_name);
end

end

function numerical_rank = numerical_matrix_rank(matrix)
%NUMERICAL_MATRIX_RANK Compute a scale-aware numerical rank.

svals = svd(matrix, 'econ');

if isempty(svals) || svals(1) == 0
    numerical_rank = 0;
else
    threshold = max(size(matrix)) * eps(svals(1));
    numerical_rank = sum(svals > threshold);
end

end

function info = zero_roundsum_info( ...
    norm_x, input_norm_info, range_sizes, max_ranks, ...
    eta, seed, d, ...
    check_error, call_timer)
%ZERO_ROUNDSUM_INFO Report a formal sum that is already zero.

info.input_formal_norm = norm_x;
info.input_norm_gram_matrix = input_norm_info.gram_matrix;
info.number_of_terms = input_norm_info.number_of_terms;
info.range_sketch_sizes = range_sizes;
info.maximum_ranks = max_ranks;
info.requested_tolerance = eta;
info.random_seed = seed;
info.maps_are_new_for_call = false;
info.error_diagnostics_measured = check_error;
info.map_distribution = "standard_normal";
info.map_factor_entry_mean = 0;
info.map_factor_entry_variance = 1;
info.range_sketch_numerical_ranks = zeros(1, d);
info.range_error_norm = 0;
info.relative_range_error = 0;
info.core_compression_error_norm = 0;
info.total_error_norm = 0;
info.relative_total_error = 0;
info.output_ranks = zeros(1, d);
info.rank_cap_active = false;
info.map_generation_time_sec = 0;
info.range_sketch_time_by_mode_sec = zeros(1, d);
info.range_qr_time_by_mode_sec = zeros(1, d);
info.projected_core_time_sec = 0;
info.projected_core_compression_time_sec = 0;
info.factor_reconstruction_time_sec = 0;
info.algorithm_time_sec = toc(call_timer);
info.error_diagnostic_time_sec = 0;
info.call_time_sec = info.algorithm_time_sec;

end
