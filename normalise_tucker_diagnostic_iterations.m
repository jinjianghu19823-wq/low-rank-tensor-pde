function iterations = normalise_tucker_diagnostic_iterations( ...
    iterations, maxit)
%NORMALISE_TUCKER_DIAGNOSTIC_ITERATIONS Validate optional checkpoints.

if nargin < 1 || isempty(iterations)
    iterations = zeros(0, 1);
    return
end

iterations = unique(double(iterations(:)));

if any(~isfinite(iterations)) || ...
        any(iterations ~= floor(iterations)) || ...
        any(iterations < 1) || any(iterations > maxit)
    error(['Diagnostic iterations must be integers between 1 and ', ...
           'maxit.']);
end

end
