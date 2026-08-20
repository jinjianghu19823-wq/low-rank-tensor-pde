function sketch = create_tucker_row_khatri_rao_sketch( ...
    n, sketch_size, seed)
%CREATE_TUCKER_ROW_KHATRI_RAO_SKETCH Draw the residual sketch in Section 5.6.

n = double(n(:).');
d = numel(n);

if d < 2 || any(n < 1) || ...
        any(n ~= floor(n))
    error('n must contain at least two positive integers.');
end

if ~isscalar(sketch_size) || sketch_size < 1 || ...
        sketch_size ~= floor(sketch_size)
    error('sketch_size must be a positive integer.');
end

if ~isscalar(seed) || ~isfinite(seed) || ...
        seed < 0 || seed ~= floor(seed)
    error('seed must be a nonnegative integer.');
end

stream = RandStream('mt19937ar', 'Seed', seed);
entry_standard_deviation = sketch_size^(-1 / (2 * d));
factors = cell(d, 1);

for mode = 1:d
    factors{mode} = entry_standard_deviation * ...
        randn(stream, sketch_size, n(mode));
end

sketch.type = "row_khatri_rao";
sketch.tensor_dimensions = n;
sketch.number_of_modes = d;
sketch.sketch_size = sketch_size;
sketch.random_seed = seed;
sketch.factor_entry_variance = sketch_size^(-1 / d);
sketch.factors = factors;
sketch.storage_entries = sketch_size * sum(n);

end
