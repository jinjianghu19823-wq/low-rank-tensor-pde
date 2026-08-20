function maps = create_tucker_roundsum_maps( ...
    n, range_sizes, seed)
%CREATE_TUCKER_ROUNDSUM_MAPS Draw the RHOSVD mode range maps.

n = double(n(:).');
range_sizes = double(range_sizes(:).');
d = numel(n);

if d < 2 || any(n < 1) || ...
        any(n ~= floor(n))
    error('n must contain at least two positive integers.');
end

if isscalar(range_sizes)
    range_sizes = repmat(range_sizes, 1, d);
elseif numel(range_sizes) ~= d
    error('range_sizes must be scalar or contain one value per mode.');
end

if any(range_sizes < 1) || ...
        any(range_sizes ~= floor(range_sizes)) || ...
        any(range_sizes > n)
    error(['Every range sketch size must be a positive integer no ', ...
           'larger than its tensor mode.']);
end

if ~isscalar(seed) || ~isfinite(seed) || ...
        seed < 0 || seed ~= floor(seed)
    error('seed must be a nonnegative integer.');
end

stream = RandStream('mt19937ar', 'Seed', seed);
factors = cell(d, 1);

for target_mode = 1:d
    factors{target_mode} = cell(d, 1);

    for source_mode = 1:d
        if source_mode ~= target_mode
            factors{target_mode}{source_mode} = randn( ...
                stream, n(source_mode), ...
                range_sizes(target_mode));
        end
    end
end

maps.type = "column_khatri_rao_roundsum";
maps.distribution = "standard_normal";
maps.factor_entry_mean = 0;
maps.factor_entry_variance = 1;
maps.tensor_dimensions = n;
maps.number_of_modes = d;
maps.range_sketch_sizes = range_sizes;
maps.random_seed = seed;
maps.factors = factors;

end
