function [values, info] = apply_tucker_row_khatri_rao_sketch_sum( ...
    terms, coefficients, sketch)
%APPLY_TUCKER_ROW_KHATRI_RAO_SKETCH_SUM Sketch a formal Tucker sum.
%
% Linearity permits the residual sketch to be accumulated term by term.
% The complete Tucker sum and its large block core are not formed.

callTimer = tic;

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array.');
end

coefficients = double(coefficients(:));

if numel(coefficients) ~= numel(terms) || ...
        any(~isfinite(coefficients))
    error('coefficients must contain one finite value per Tucker term.');
end

values = zeros(sketch.sketch_size, 1);
termTime = zeros(numel(terms), 1);

for termIndex = 1:numel(terms)
    componentTimer = tic;
    oneValue = apply_tucker_row_khatri_rao_sketch( ...
        terms{termIndex}, sketch);
    termTime(termIndex) = toc(componentTimer);
    values = values + coefficients(termIndex) * oneValue;
end

info.number_of_terms = numel(terms);
info.term_time_sec = termTime;
info.call_time_sec = toc(callTimer);

end
