function [Z, info] = tucker_roundsum_rhosvd_adaptive( ...
    terms, coefficients, oversampling, tolerance, maximumRank, ...
    randomSeed, measureErrorDiagnostics)
%TUCKER_ROUNDSUM_RHOSVD_ADAPTIVE Paper-style adaptive RHOSVD RoundSum.
%
% The effective multilinear rank is estimated separately for every
% weighted Tucker sum. A common range size is selected from the largest
% estimated mode rank, limited by maximumRank, plus oversampling. The
% resulting range sketch is passed to the existing RHOSVD RoundSum routine.

if nargin < 7
    measureErrorDiagnostics = true;
end

adaptiveCallTimer = tic;

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end
terms = terms(:);
coefficients = double(coefficients(:));

if numel(coefficients) ~= numel(terms) || ...
        any(~isfinite(coefficients))
    error('coefficients must contain one finite value per Tucker term.');
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
if ~isscalar(maximumRank) || maximumRank < 1 || ...
        maximumRank ~= floor(maximumRank)
    error('maximumRank must be a positive integer.');
end

tensorDimensions = double(size(terms{1}));
numberOfModes = numel(tensorDimensions);

for termIndex = 2:numel(terms)
    if ~isequal(double(size(terms{termIndex})), tensorDimensions)
        error('Every RoundSum term must have the same tensor dimensions.');
    end
end

rankEstimationTimer = tic;
effectiveRanks = estimate_effective_ranks( ...
    terms, coefficients, tolerance, numberOfModes);
rankEstimationTime = toc(rankEstimationTimer);
commonRangeSize = min(max(effectiveRanks), maximumRank) + oversampling;
rangeSketchSizes = min(commonRangeSize, tensorDimensions);
maximumRanks = min(maximumRank, rangeSketchSizes);

[Z, fixedInfo] = tucker_roundsum_rhosvd( ...
    terms, coefficients, rangeSketchSizes, tolerance, maximumRanks, ...
    randomSeed, measureErrorDiagnostics);

info = fixedInfo;
info.base_roundsum_algorithm_time_sec = fixedInfo.algorithm_time_sec;
info.rank_estimation_time_sec = rankEstimationTime;
info.algorithm_time_sec = ...
    fixedInfo.algorithm_time_sec + rankEstimationTime;
info.adaptive_call_time_sec = toc(adaptiveCallTimer);
info.adaptive_rank_selection = true;
info.effective_rank_estimates = effectiveRanks;
info.oversampling = oversampling;
info.maximum_rank_cap = maximumRank;
info.common_range_size_before_mode_caps = commonRangeSize;

end


function effectiveRanks = estimate_effective_ranks( ...
    terms, coefficients, tolerance, numberOfModes)
%ESTIMATE_EFFECTIVE_RANKS Reproduce the paper code's Gramian estimator.

effectiveRanks = ones(1, numberOfModes);

for modeIndex = 1:numberOfModes
    weightedFactors = [];

    for termIndex = 1:numel(terms)
        scaledCoreNorm = abs(coefficients(termIndex)) * ...
            norm(terms{termIndex}.core);
        weightedFactors = [weightedFactors, ...
            scaledCoreNorm * terms{termIndex}.u{modeIndex}]; %#ok<AGROW>
    end

    gramMatrix = weightedFactors.' * weightedFactors;
    eigenvalues = sort(max(0, eig(gramMatrix)), 'descend');

    if isempty(eigenvalues) || sum(eigenvalues) == 0
        effectiveRanks(modeIndex) = 1;
        continue
    end

    discardedEnergy = cumsum(eigenvalues, 'reverse');
    threshold = tolerance^2 * sum(eigenvalues) / numberOfModes;
    retainedRank = find(discardedEnergy > threshold, 1, 'last');

    if isempty(retainedRank)
        retainedRank = 1;
    end
    effectiveRanks(modeIndex) = retainedRank;
end

end
