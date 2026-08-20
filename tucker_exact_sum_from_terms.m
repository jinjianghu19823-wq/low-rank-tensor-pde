function X = tucker_exact_sum_from_terms(terms, coeff)
%TUCKER_EXACT_SUM_FROM_TERMS Form a formal weighted Tucker sum exactly.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array.');
end

coeff = double(coeff(:));

if numel(coeff) ~= numel(terms) || ...
        any(~isfinite(coeff))
    error('coeff must contain one finite value per Tucker term.');
end

for term_idx = 1:numel(terms)
    if ~isa(terms{term_idx}, 'ttensor')
        error('Every formal sum term must be a ttensor.');
    end
    if term_idx > 1 && ...
            ~isequal(size(terms{term_idx}), size(terms{1}))
        error('Every formal sum term must have the same dimensions.');
    end
end

X = coeff(1) * terms{1};

for term_idx = 2:numel(terms)
    X = tucker_axpby_exact( ...
        X, 1, terms{term_idx}, coeff(term_idx));
end

end
