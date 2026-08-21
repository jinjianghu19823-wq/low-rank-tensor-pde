function [U, info] = ...
    tucker_gmres_left_preconditioned_capped_relaxed( ...
        originalOperatorFunction, preconditionerFunction, F0, U0, ...
        maximumIteration, internalPreconditionedStoppingTolerance, ...
        relaxationErrorBudget, maximumRelaxedTolerance, ...
        originalTrueResidualFunction, ...
        preconditionedResidualNormFunction, displayProgress, ...
        maximumBasisStorageEntries, maximumMultilinearRank, ...
        diagnosticIterations)
%TUCKER_GMRES_LEFT_PRECONDITIONED_CAPPED_RELAXED Run Algorithm 5.6.
%
% This function separates the three controls in the practical algorithm:
%
%   internalPreconditionedStoppingTolerance
%       The computed preconditioned residual threshold.
%
%   relaxationErrorBudget
%       The numerator in the inverse-relative-residual schedule.
%
%   maximumRelaxedTolerance
%       The largest recompression tolerance permitted at any step.
%
% The schedule is
%
%   eta_j = min(maximumRelaxedTolerance, ...
%       relaxationErrorBudget / ...
%       (computedResidualNorm_{j-1}/beta)).
%
% Thesis/experiment notation (Sections 5.5 and 6.2):
%   originalOperatorFunction       <->  \mathcal A_0
%   preconditionerFunction         <->  \mathcal P
%   F0, U0, U                      <->  \mathcal B_0, \mathcal X_0, \mathcal X_j
%   maximumIteration               <->  four-step basis limit in Section 6.2
%   internalPreconditionedStoppingTolerance
%                                   <->  internal threshold (10^(-14))
%   relaxationErrorBudget          <->  base budget (10^(-10))
%   maximumRelaxedTolerance        <->  \eta_max (10^(-2))
%   maximumMultilinearRank         <->  \boldsymbol R=(R_1,...,R_d)
% The base budget is passed twice below because the shared routine keeps a
% fixed-tolerance argument in its common interface; in relaxed mode, the
% final two arguments define the active \eta_j schedule.

if nargin < 11
    displayProgress = false;
end

if nargin < 12
    maximumBasisStorageEntries = Inf;
end

if nargin < 13
    maximumMultilinearRank = [];
end
if nargin < 14
    diagnosticIterations = [];
end

[U, info] = run_left_preconditioned_tucker_gmres_cycle( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, internalPreconditionedStoppingTolerance, ...
    relaxationErrorBudget, "relaxed", ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank, relaxationErrorBudget, ...
    maximumRelaxedTolerance, diagnosticIterations);

end
