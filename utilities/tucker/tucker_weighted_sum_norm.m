function [sumNorm, info] = tucker_weighted_sum_norm(terms, coefficients)
%TUCKER_WEIGHTED_SUM_NORM Norm of a formal weighted Tucker sum.
%
% The calculation uses pairwise Tucker inner products. It does not form the
% block Tucker core of the complete sum.


%% 1. Check the formal sum

[terms, coefficients] = check_formal_sum(terms, coefficients);
numberOfTerms = numel(terms);


%% 2. Form the small Gram matrix

gramMatrix = zeros(numberOfTerms);

for rowIndex = 1:numberOfTerms
    gramMatrix(rowIndex, rowIndex) = ...
        real(innerprod(terms{rowIndex}, terms{rowIndex}));

    for columnIndex = rowIndex + 1:numberOfTerms
        value = real(innerprod(terms{rowIndex}, terms{columnIndex}));
        gramMatrix(rowIndex, columnIndex) = value;
        gramMatrix(columnIndex, rowIndex) = value;
    end
end

squaredNorm = real(coefficients.' * gramMatrix * coefficients);
roundoffScale = norm(coefficients, 1)^2 * ...
    max(1, max(abs(gramMatrix), [], 'all'));

if squaredNorm < 0 && abs(squaredNorm) <= 100 * eps * roundoffScale
    squaredNorm = 0;
elseif squaredNorm < 0
    error('The weighted Tucker sum produced a negative squared norm.');
end

sumNorm = sqrt(squaredNorm);

info.gram_matrix = gramMatrix;
info.squared_norm = squaredNorm;
info.number_of_terms = numberOfTerms;

end


function [terms, coefficients] = check_formal_sum(terms, coefficients)
%CHECK_FORMAL_SUM Validate a weighted collection of Tucker tensors.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end

terms = terms(:);
coefficients = double(coefficients(:));

if numel(coefficients) ~= numel(terms) || ...
        any(~isfinite(coefficients))
    error('coefficients must contain one finite value per Tucker term.');
end

referenceSize = size(terms{1});

for termIndex = 1:numel(terms)
    if ~isa(terms{termIndex}, 'ttensor')
        error('Every formal-sum term must be a Tensor Toolbox ttensor.');
    end
    if ~isequal(size(terms{termIndex}), referenceSize)
        error('Every formal-sum term must have the same dimensions.');
    end
end

end
