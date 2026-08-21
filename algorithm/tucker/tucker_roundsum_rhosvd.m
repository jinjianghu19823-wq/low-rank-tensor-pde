function [Z, info] = tucker_roundsum_rhosvd( ...
    terms, coefficients, rangeSketchSizes, compressionTolerance, ...
    maximumMultilinearRank, randomSeed, measureErrorDiagnostics)
%TUCKER_ROUNDSUM_RHOSVD Approximate a weighted Tucker sum by RHOSVD.
%
% The mode range sketches are accumulated term by term. The exact block
% Tucker core of the complete sum is never constructed. Independent
% standard Gaussian factor matrices represent each mode range map. The
% caller supplies a seed, and a fresh map collection is drawn on every
% invocation.
%
% The requested output rank must satisfy R_n <= k_n in every mode. If no
% rank cap is supplied, the range sketch sizes are used as the cap.
% Exact realised errors are measured by default. A timing experiment may
% disable them without changing the randomized approximation.

if nargin < 7
    measureErrorDiagnostics = true;
end

if ~islogical(measureErrorDiagnostics) || ...
        ~isscalar(measureErrorDiagnostics)
    error('measureErrorDiagnostics must be one logical value.');
end

callTimer = tic;


%% 1. Check the formal sum and RHOSVD settings

[terms, coefficients, tensorDimensions] = ...
    check_roundsum_inputs(terms, coefficients);

numberOfModes = numel(tensorDimensions);
rangeSketchSizes = expand_mode_vector( ...
    rangeSketchSizes, numberOfModes, 'rangeSketchSizes');

if any(rangeSketchSizes < 1) || ...
        any(rangeSketchSizes ~= floor(rangeSketchSizes)) || ...
        any(rangeSketchSizes > tensorDimensions)
    error(['Every range sketch size must be a positive integer no ', ...
           'larger than its tensor mode.']);
end

if ~isscalar(compressionTolerance) || compressionTolerance <= 0 || ...
        compressionTolerance >= 1
    error('compressionTolerance must be one number between 0 and 1.');
end

if isempty(maximumMultilinearRank)
    maximumRanks = rangeSketchSizes;
else
    maximumRanks = normalise_tucker_rank_cap( ...
        maximumMultilinearRank, tensorDimensions);
end

if any(maximumRanks > rangeSketchSizes)
    error('Every RoundSum output rank R_n must satisfy R_n <= k_n.');
end

if ~isscalar(randomSeed) || ~isfinite(randomSeed) || ...
        randomSeed < 0 || randomSeed ~= floor(randomSeed)
    error('randomSeed must be a nonnegative integer.');
end

[inputNorm, inputNormInfo] = ...
    tucker_weighted_sum_norm(terms, coefficients);
inputScale = 0;

for termIndex = 1:numel(terms)
    inputScale = inputScale + ...
        abs(coefficients(termIndex)) * norm(terms{termIndex});
end

zeroThreshold = 100 * eps * max(1, inputScale);

if inputNorm <= zeroThreshold
    Z = 0 * terms{1};
    info = zero_roundsum_info( ...
        inputNorm, inputNormInfo, rangeSketchSizes, maximumRanks, ...
        compressionTolerance, randomSeed, numberOfModes, ...
        measureErrorDiagnostics, callTimer);
    return
end


%% 2. Draw fresh standard Gaussian mode range maps

mapTimer = tic;
maps = create_tucker_roundsum_maps( ...
    tensorDimensions, rangeSketchSizes, randomSeed);
mapGenerationTime = toc(mapTimer);


%% 3. Form the mode range sketches term by term

rangeBases = cell(numberOfModes, 1);
rangeSketchNumericalRanks = zeros(1, numberOfModes);
rangeSketchTimeByMode = zeros(1, numberOfModes);
rangeQrTimeByMode = zeros(1, numberOfModes);

for targetMode = 1:numberOfModes
    componentTimer = tic;
    oneRangeSketch = zeros( ...
        tensorDimensions(targetMode), rangeSketchSizes(targetMode));

    for termIndex = 1:numel(terms)
        oneTerm = terms{termIndex};

        for sketchColumn = 1:rangeSketchSizes(targetMode)
            contractedCore = oneTerm.core;

            for sourceMode = 1:numberOfModes
                if sourceMode ~= targetMode
                    projectedVector = oneTerm.u{sourceMode}.' * ...
                        maps.factors{targetMode}{sourceMode}( ...
                        :, sketchColumn);
                    contractedCore = ttm( ...
                        contractedCore, projectedVector.', sourceMode);
                end
            end

            modeVector = reshape(double(contractedCore), [], 1);
            oneRangeSketch(:, sketchColumn) = ...
                oneRangeSketch(:, sketchColumn) + ...
                coefficients(termIndex) * ...
                (oneTerm.u{targetMode} * modeVector);
        end
    end

    rangeSketchTimeByMode(targetMode) = toc(componentTimer);
    rangeSketchNumericalRanks(targetMode) = ...
        numerical_matrix_rank(oneRangeSketch);

    componentTimer = tic;
    [rangeBases{targetMode}, ~] = qr(oneRangeSketch, 0);
    rangeQrTimeByMode(targetMode) = toc(componentTimer);
end


%% 4. Form and compress the small projected core

componentTimer = tic;
projectedCoreValues = zeros(rangeSketchSizes);

for termIndex = 1:numel(terms)
    projectedTermCore = terms{termIndex}.core;

    for mode = 1:numberOfModes
        projectedTermCore = ttm( ...
            projectedTermCore, ...
            rangeBases{mode}.' * terms{termIndex}.u{mode}, mode);
    end

    projectedCoreValues = projectedCoreValues + ...
        coefficients(termIndex) * double(projectedTermCore);
end

projectedCore = tensor(projectedCoreValues);
projectedCoreTime = toc(componentTimer);

componentTimer = tic;
[compressedCoreTensor, compressionInfo] = sthosvd_round_tensor( ...
    projectedCore, compressionTolerance, maximumRanks, 1:numberOfModes);
projectedCoreCompressionTime = toc(componentTimer);


%% 5. Return to the original mode spaces

componentTimer = tic;
outputFactors = cell(numberOfModes, 1);

for mode = 1:numberOfModes
    outputFactors{mode} = ...
        rangeBases{mode} * compressedCoreTensor.u{mode};
end

Z = ttensor(compressedCoreTensor.core, outputFactors);
factorReconstructionTime = toc(componentTimer);


%% 6. Record realised errors and map information

algorithmTime = toc(callTimer);
if measureErrorDiagnostics
    errorDiagnosticTimer = tic;
    projectedTensor = ttensor(projectedCore, rangeBases);
    [rangeError, ~] = tucker_weighted_sum_norm( ...
        [terms; {projectedTensor}], [coefficients; -1]);
    [totalError, ~] = tucker_weighted_sum_norm( ...
        [terms; {Z}], [coefficients; -1]);

    if inputNorm == 0
        relativeRangeError = 0;
        relativeTotalError = 0;
    else
        relativeRangeError = rangeError / inputNorm;
        relativeTotalError = totalError / inputNorm;
    end

    compressedProjectedCore = full(compressedCoreTensor);
    coreCompressionError = norm(projectedCore - compressedProjectedCore);
    errorDiagnosticTime = toc(errorDiagnosticTimer);
else
    rangeError = NaN;
    relativeRangeError = NaN;
    coreCompressionError = NaN;
    totalError = NaN;
    relativeTotalError = NaN;
    errorDiagnosticTime = 0;
end

info.input_formal_norm = inputNorm;
info.input_norm_gram_matrix = inputNormInfo.gram_matrix;
info.number_of_terms = numel(terms);
info.range_sketch_sizes = rangeSketchSizes;
info.maximum_ranks = maximumRanks;
info.requested_tolerance = compressionTolerance;
info.random_seed = randomSeed;
info.maps_are_new_for_call = true;
info.error_diagnostics_measured = measureErrorDiagnostics;
info.map_distribution = maps.distribution;
info.map_factor_entry_mean = maps.factor_entry_mean;
info.map_factor_entry_variance = maps.factor_entry_variance;
info.range_sketch_numerical_ranks = rangeSketchNumericalRanks;
info.range_error_norm = rangeError;
info.relative_range_error = relativeRangeError;
info.core_compression_error_norm = coreCompressionError;
info.total_error_norm = totalError;
info.relative_total_error = relativeTotalError;
info.output_ranks = size(Z.core);
info.rank_cap_active = compressionInfo.rank_cap_active;
info.sthosvd = compressionInfo;
info.map_generation_time_sec = mapGenerationTime;
info.range_sketch_time_by_mode_sec = rangeSketchTimeByMode;
info.range_qr_time_by_mode_sec = rangeQrTimeByMode;
info.projected_core_time_sec = projectedCoreTime;
info.projected_core_compression_time_sec = ...
    projectedCoreCompressionTime;
info.factor_reconstruction_time_sec = factorReconstructionTime;
info.algorithm_time_sec = algorithmTime;
info.error_diagnostic_time_sec = errorDiagnosticTime;
info.call_time_sec = toc(callTimer);

end


function [terms, coefficients, tensorDimensions] = ...
    check_roundsum_inputs(terms, coefficients)
%CHECK_ROUNDSUM_INPUTS Validate the weighted Tucker collection.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end

terms = terms(:);
coefficients = double(coefficients(:));

if numel(coefficients) ~= numel(terms) || ...
        any(~isfinite(coefficients))
    error('coefficients must contain one finite value per Tucker term.');
end

if ~isa(terms{1}, 'ttensor')
    error('Every RoundSum term must be a Tensor Toolbox ttensor.');
end

tensorDimensions = double(size(terms{1}));

for termIndex = 1:numel(terms)
    if ~isa(terms{termIndex}, 'ttensor') || ...
            ~isequal(double(size(terms{termIndex})), tensorDimensions)
        error('Every RoundSum term must be a same-sized ttensor.');
    end
end

end


function values = expand_mode_vector(values, numberOfModes, argumentName)
%EXPAND_MODE_VECTOR Expand a scalar to one value per tensor mode.

values = double(values(:).');

if isscalar(values)
    values = repmat(values, 1, numberOfModes);
elseif numel(values) ~= numberOfModes
    error('%s must be scalar or contain one value per mode.', argumentName);
end

end


function numericalRank = numerical_matrix_rank(matrix)
%NUMERICAL_MATRIX_RANK Compute a scale-aware numerical rank.

singularValues = svd(matrix, 'econ');

if isempty(singularValues) || singularValues(1) == 0
    numericalRank = 0;
else
    threshold = max(size(matrix)) * eps(singularValues(1));
    numericalRank = sum(singularValues > threshold);
end

end


function info = zero_roundsum_info( ...
    inputNorm, inputNormInfo, rangeSketchSizes, maximumRanks, ...
    compressionTolerance, randomSeed, numberOfModes, ...
    measureErrorDiagnostics, callTimer)
%ZERO_ROUNDSUM_INFO Report a formal sum that is already zero.

info.input_formal_norm = inputNorm;
info.input_norm_gram_matrix = inputNormInfo.gram_matrix;
info.number_of_terms = inputNormInfo.number_of_terms;
info.range_sketch_sizes = rangeSketchSizes;
info.maximum_ranks = maximumRanks;
info.requested_tolerance = compressionTolerance;
info.random_seed = randomSeed;
info.maps_are_new_for_call = false;
info.error_diagnostics_measured = measureErrorDiagnostics;
info.map_distribution = "standard_normal";
info.map_factor_entry_mean = 0;
info.map_factor_entry_variance = 1;
info.range_sketch_numerical_ranks = zeros(1, numberOfModes);
info.range_error_norm = 0;
info.relative_range_error = 0;
info.core_compression_error_norm = 0;
info.total_error_norm = 0;
info.relative_total_error = 0;
info.output_ranks = zeros(1, numberOfModes);
info.rank_cap_active = false;
info.map_generation_time_sec = 0;
info.range_sketch_time_by_mode_sec = zeros(1, numberOfModes);
info.range_qr_time_by_mode_sec = zeros(1, numberOfModes);
info.projected_core_time_sec = 0;
info.projected_core_compression_time_sec = 0;
info.factor_reconstruction_time_sec = 0;
info.algorithm_time_sec = toc(callTimer);
info.error_diagnostic_time_sec = 0;
info.call_time_sec = info.algorithm_time_sec;

end
