function [Z, info] = identity_tucker_fixed_linear_preconditioner(Y)
%IDENTITY_TUCKER_FIXED_LINEAR_PRECONDITIONER Exact identity map.

if ~isa(Y, 'ttensor')
    error('The identity preconditioner input must be a ttensor.');
end

Z = Y;
info.rounding_performed = false;
info.rank_cap_active = false;
info.relative_error_estimate = 0;
info.call_time_sec = 0;

end
