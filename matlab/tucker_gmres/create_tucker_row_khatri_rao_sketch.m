function sketch = create_tucker_row_khatri_rao_sketch( ...
    tensorDimensions, sketchSize, randomSeed)
%CREATE_TUCKER_ROW_KHATRI_RAO_SKETCH Draw the residual sketch in Section 5.6.
%
% Row a of the implicit full matrix is
%
%   S_d(a,:) kron ... kron S_1(a,:).
%
% The full matrix is never formed. Each factor entry has variance
% sketchSize^(-1/d), so each entry of the implicit full row has variance
% 1/sketchSize.

tensorDimensions = double(tensorDimensions(:).');
numberOfModes = numel(tensorDimensions);

if numberOfModes < 2 || any(tensorDimensions < 1) || ...
        any(tensorDimensions ~= floor(tensorDimensions))
    error('tensorDimensions must contain at least two positive integers.');
end

if ~isscalar(sketchSize) || sketchSize < 1 || ...
        sketchSize ~= floor(sketchSize)
    error('sketchSize must be a positive integer.');
end

if ~isscalar(randomSeed) || ~isfinite(randomSeed) || ...
        randomSeed < 0 || randomSeed ~= floor(randomSeed)
    error('randomSeed must be a nonnegative integer.');
end

stream = RandStream('mt19937ar', 'Seed', randomSeed);
entryStandardDeviation = sketchSize^(-1 / (2 * numberOfModes));
factors = cell(numberOfModes, 1);

for mode = 1:numberOfModes
    factors{mode} = entryStandardDeviation * ...
        randn(stream, sketchSize, tensorDimensions(mode));
end

sketch.type = "row_khatri_rao";
sketch.tensor_dimensions = tensorDimensions;
sketch.number_of_modes = numberOfModes;
sketch.sketch_size = sketchSize;
sketch.random_seed = randomSeed;
sketch.factor_entry_variance = sketchSize^(-1 / numberOfModes);
sketch.factors = factors;
sketch.storage_entries = sketchSize * sum(tensorDimensions);

end
