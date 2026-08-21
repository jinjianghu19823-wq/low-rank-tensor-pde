function maps = create_tucker_roundsum_maps( ...
    tensorDimensions, rangeSketchSizes, randomSeed)
%CREATE_TUCKER_ROUNDSUM_MAPS Draw the RHOSVD mode range maps.
%
% For every target mode n and every other mode m, this routine draws
%
%     A_m^(n) in R^(I_m by k_n)
%
% with independent standard Gaussian entries. Their implicit column
% Khatri Rao product is the range map for mode n. A new structure is made
% for every call, so callers can use a different seed for every RoundSum.

tensorDimensions = double(tensorDimensions(:).');
rangeSketchSizes = double(rangeSketchSizes(:).');
numberOfModes = numel(tensorDimensions);

if numberOfModes < 2 || any(tensorDimensions < 1) || ...
        any(tensorDimensions ~= floor(tensorDimensions))
    error('tensorDimensions must contain at least two positive integers.');
end

if isscalar(rangeSketchSizes)
    rangeSketchSizes = repmat(rangeSketchSizes, 1, numberOfModes);
elseif numel(rangeSketchSizes) ~= numberOfModes
    error('rangeSketchSizes must be scalar or contain one value per mode.');
end

if any(rangeSketchSizes < 1) || ...
        any(rangeSketchSizes ~= floor(rangeSketchSizes)) || ...
        any(rangeSketchSizes > tensorDimensions)
    error(['Every range sketch size must be a positive integer no ', ...
           'larger than its tensor mode.']);
end

if ~isscalar(randomSeed) || ~isfinite(randomSeed) || ...
        randomSeed < 0 || randomSeed ~= floor(randomSeed)
    error('randomSeed must be a nonnegative integer.');
end

stream = RandStream('mt19937ar', 'Seed', randomSeed);
factors = cell(numberOfModes, 1);

for targetMode = 1:numberOfModes
    factors{targetMode} = cell(numberOfModes, 1);

    for sourceMode = 1:numberOfModes
        if sourceMode ~= targetMode
            factors{targetMode}{sourceMode} = randn( ...
                stream, tensorDimensions(sourceMode), ...
                rangeSketchSizes(targetMode));
        end
    end
end

maps.type = "column_khatri_rao_roundsum";
maps.distribution = "standard_normal";
maps.factor_entry_mean = 0;
maps.factor_entry_variance = 1;
maps.tensor_dimensions = tensorDimensions;
maps.number_of_modes = numberOfModes;
maps.range_sketch_sizes = rangeSketchSizes;
maps.random_seed = randomSeed;
maps.factors = factors;

end
