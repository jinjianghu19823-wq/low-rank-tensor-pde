function [terms, coefficients, info] = poisson_action_tucker_terms(X, A1)
%POISSON_ACTION_TUCKER_TERMS Return the exact formal Poisson sum.
%
% The result represents
%
%     A0(X) = sum_n X x_n A1
%
% as separate Tucker terms. No addition or recompression is performed.
% This representation permits a residual sketch to be accumulated before
% any deterministic or randomized rounding is applied.

callTimer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

tensorDimensions = double(size(X));
numberOfModes = ndims(X);

if any(tensorDimensions ~= size(A1, 1))
    error('Every mode of X must have the same size as A1.');
end

terms = cell(numberOfModes, 1);
modeProductTime = zeros(numberOfModes, 1);

for mode = 1:numberOfModes
    componentTimer = tic;
    terms{mode} = ttm(X, A1, mode);
    modeProductTime(mode) = toc(componentTimer);
end

coefficients = ones(numberOfModes, 1);

info.number_of_terms = numberOfModes;
info.coefficients = coefficients;
info.rounding_performed = false;
info.mode_product_time_sec = modeProductTime;
info.call_time_sec = toc(callTimer);

end
