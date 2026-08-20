function [terms, coeff, info] = poisson_action_tucker_terms(X, A1)
%POISSON_ACTION_TUCKER_TERMS Return the exact formal Poisson sum.

call_timer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

n = double(size(X));
d = ndims(X);

if any(n ~= size(A1, 1))
    error('Every mode of X must have the same size as A1.');
end

terms = cell(d, 1);
mode_product_time = zeros(d, 1);

for mode = 1:d
    component_timer = tic;
    terms{mode} = ttm(X, A1, mode);
    mode_product_time(mode) = toc(component_timer);
end

coeff = ones(d, 1);

info.number_of_terms = d;
info.coeff = coeff;
info.rounding_performed = false;
info.mode_product_time_sec = mode_product_time;
info.call_time_sec = toc(call_timer);

end
