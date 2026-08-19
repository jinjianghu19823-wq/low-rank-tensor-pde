function [Z, info] = identity_tucker_preconditioner( ...
    Y, requestedTolerance, maximumRanks)
%IDENTITY_TUCKER_PRECONDITIONER Return a Tucker tensor unchanged.
%
% This helper gives the left-preconditioned solvers a common identity
% preconditioner for correctness controls. No rounding is performed.

if ~isa(Y, 'ttensor')
    error('The identity preconditioner input must be a ttensor.');
end

Z = Y;
info.requested_tolerance = requestedTolerance;
info.maximum_ranks = maximumRanks;
info.rank_cap_active = false;
info.relative_error_estimate = 0;
info.kernel_timing = empty_tucker_kernel_timing();

end
