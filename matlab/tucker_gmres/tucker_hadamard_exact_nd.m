function Z = tucker_hadamard_exact_nd(X, Y)
%TUCKER_HADAMARD_EXACT_ND Exact Tucker Hadamard representation for order d.

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must be Tensor Toolbox ttensor objects.');
end
if ndims(X) ~= ndims(Y) || any(size(X) ~= size(Y))
    error('X and Y must have the same tensor order and dimensions.');
end

numberOfModes = ndims(X);
ranksX = double(size(X.core));
ranksY = double(size(Y.core));

shapeX = ones(1, 2 * numberOfModes);
shapeY = ones(1, 2 * numberOfModes);
shapeX(1:2:end) = ranksX;
shapeY(2:2:end) = ranksY;

expandedX = reshape(double(X.core), shapeX);
expandedY = reshape(double(Y.core), shapeY);
% Preserve trailing singleton core dimensions explicitly. Numeric reshape
% alone cannot carry the intended tensor order when the last ranks are one.
pairedCoreValues = expandedX .* expandedY;
pairedCore = tensor(pairedCoreValues(:), ranksX .* ranksY);

pairedFactors = cell(numberOfModes, 1);
for mode = 1:numberOfModes
    factorX = X.u{mode};
    factorY = Y.u{mode};
    pairedFactor = zeros(size(factorX, 1), ranksX(mode) * ranksY(mode));
    columnIndex = 0;
    for columnY = 1:ranksY(mode)
        for columnX = 1:ranksX(mode)
            columnIndex = columnIndex + 1;
            pairedFactor(:, columnIndex) = ...
                factorX(:, columnX) .* factorY(:, columnY);
        end
    end
    pairedFactors{mode} = pairedFactor;
end

Z = ttensor(pairedCore, pairedFactors);

end
