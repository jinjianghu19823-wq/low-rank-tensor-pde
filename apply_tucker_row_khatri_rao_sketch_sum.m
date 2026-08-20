function [values, info] = apply_tucker_row_khatri_rao_sketch_sum( ...
    terms, coeff, sketch)
%APPLY_TUCKER_ROW_KHATRI_RAO_SKETCH_SUM Sketch a formal Tucker sum.

call_timer = tic;

if ~iscell(terms) || isempty(terms)
    error('terms must be a nonempty cell array.');
end

coeff = double(coeff(:));

if numel(coeff) ~= numel(terms) || ...
        any(~isfinite(coeff))
    error('coeff must contain one finite value per Tucker term.');
end

values = zeros(sketch.sketch_size, 1);
term_time = zeros(numel(terms), 1);

for term_idx = 1:numel(terms)
    component_timer = tic;
    one_value = apply_tucker_row_khatri_rao_sketch( ...
        terms{term_idx}, sketch);
    term_time(term_idx) = toc(component_timer);
    values = values + coeff(term_idx) * one_value;
end

info.number_of_terms = numel(terms);
info.term_time_sec = term_time;
info.call_time_sec = toc(call_timer);

end
