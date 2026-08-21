function X = tucker_exact_sum_from_terms(terms, coefficients)
%TUCKER_EXACT_SUM_FROM_TERMS Form a formal weighted Tucker sum exactly.
%
% No recompression is performed. The block Tucker ranks may therefore grow
% with the number of terms. This helper is intended for small formal sums
% immediately before one explicit rounding operation.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array.');
end

coefficients = double(coefficients(:));

if numel(coefficients) ~= numel(terms) || ...
        any(~isfinite(coefficients))
    error('coefficients must contain one finite value per Tucker term.');
end

for termIndex = 1:numel(terms)
    if ~isa(terms{termIndex}, 'ttensor')
        error('Every formal sum term must be a ttensor.');
    end
    if termIndex > 1 && ...
            ~isequal(size(terms{termIndex}), size(terms{1}))
        error('Every formal sum term must have the same dimensions.');
    end
end

X = coefficients(1) * terms{1};

for termIndex = 2:numel(terms)
    X = tucker_axpby_exact( ...
        X, 1, terms{termIndex}, coefficients(termIndex));
end

end
