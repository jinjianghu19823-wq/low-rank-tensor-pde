function [sum_norm, info] = tucker_weighted_sum_norm(terms, coeff)
%TUCKER_WEIGHTED_SUM_NORM Norm of a formal weighted Tucker sum.

[terms, coeff] = check_formal_sum(terms, coeff);
nterms = numel(terms);

% Form the small Gram matrix

G = zeros(nterms);

for row_idx = 1:nterms
    G(row_idx, row_idx) = ...
        real(innerprod(terms{row_idx}, terms{row_idx}));

    for col_idx = row_idx + 1:nterms
        value = real(innerprod(terms{row_idx}, terms{col_idx}));
        G(row_idx, col_idx) = value;
        G(col_idx, row_idx) = value;
    end
end

norm_sq = real(coeff.' * G * coeff);
roundoff_scale = norm(coeff, 1)^2 * ...
    max(1, max(abs(G), [], 'all'));

if norm_sq < 0 && abs(norm_sq) <= 100 * eps * roundoff_scale
    norm_sq = 0;
elseif norm_sq < 0
    error('The weighted Tucker sum produced a negative squared norm.');
end

sum_norm = sqrt(norm_sq);

info.gram_matrix = G;
info.squared_norm = norm_sq;
info.number_of_terms = nterms;

end

function [terms, coeff] = check_formal_sum(terms, coeff)
%CHECK_FORMAL_SUM Validate a weighted collection of Tucker tensors.

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array of Tucker tensors.');
end

terms = terms(:);
coeff = double(coeff(:));

if numel(coeff) ~= numel(terms) || ...
        any(~isfinite(coeff))
    error('coeff must contain one finite value per Tucker term.');
end

reference_size = size(terms{1});

for term_idx = 1:numel(terms)
    if ~isa(terms{term_idx}, 'ttensor')
        error('Every formal-sum term must be a Tensor Toolbox ttensor.');
    end
    if ~isequal(size(terms{term_idx}), reference_size)
        error('Every formal-sum term must have the same dimensions.');
    end
end

end
