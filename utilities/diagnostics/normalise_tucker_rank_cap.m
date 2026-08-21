function maximumRanks = normalise_tucker_rank_cap( ...
    maximumMultilinearRank, tensorDimensions)
%NORMALISE_TUCKER_RANK_CAP Check and expand a Tucker rank cap.
%
% The optional maximum multilinear rank may be supplied as:
%
%   []          no additional cap;
%   R           the same maximum rank in every mode;
%   [R1,...,Rd] one maximum rank for each mode.
%
% A missing cap is represented by the full tensor dimensions. Therefore the
% returned vector is always finite and has one entry per tensor mode.
%
% Thesis notation (Section 5.5):
%   maximumMultilinearRank, maximumRanks  <->  \boldsymbol R=(R_1,...,R_d)
%   tensorDimensions                     <->  (I_1,...,I_d)
%   numberOfModes                        <->  d


%% 1. Check the tensor dimensions

tensorDimensions = double(tensorDimensions(:).');
numberOfModes = numel(tensorDimensions);

if numberOfModes < 1 || any(tensorDimensions < 1) || ...
        any(tensorDimensions ~= floor(tensorDimensions))
    error('tensorDimensions must contain positive integer dimensions.');
end


%% 2. Expand a missing or scalar cap

if isempty(maximumMultilinearRank)
    maximumRanks = tensorDimensions;
    return
end

maximumRanks = double(maximumMultilinearRank(:).');

if isscalar(maximumRanks)
    maximumRanks = repmat(maximumRanks, 1, numberOfModes);
elseif numel(maximumRanks) ~= numberOfModes
    error(['maximumMultilinearRank must be empty, scalar, or contain ', ...
           'one entry per tensor mode.']);
end


%% 3. Validate and clip the cap to the ambient mode sizes

if any(~isfinite(maximumRanks)) || any(maximumRanks < 1) || ...
        any(maximumRanks ~= floor(maximumRanks))
    error('Every maximum Tucker rank must be a positive finite integer.');
end

maximumRanks = min(maximumRanks, tensorDimensions);

end
