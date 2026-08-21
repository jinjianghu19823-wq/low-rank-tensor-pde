function [U, info] = tucker_gmres_left_preconditioned_fixed( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, targetTolerance, compressionTolerance, ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank, diagnosticIterations)
%TUCKER_GMRES_LEFT_PRECONDITIONED_FIXED Fixed-tolerance control method.
%
% This method applies the fixed-tolerance algorithm from Section 5.5 to the
% left-preconditioned equation. It uses the same compressionTolerance at
% every Arnoldi step. Its experimental role is to isolate the effect of
% relaxation after both Tucker methods receive the same preconditioner.
%
% Thesis notation (Section 5.5):
%   originalOperatorFunction       <->  \mathcal A_0
%   preconditionerFunction         <->  \mathcal P
%   F0, U0, U                      <->  \mathcal B_0, \mathcal X_0, \mathcal X_j
%   maximumIteration               <->  \ell_max
%   targetTolerance                <->  internal residual threshold
%   compressionTolerance           <->  fixed \eta
%   maximumMultilinearRank         <->  \boldsymbol R=(R_1,...,R_d)
% The wrapper sets toleranceMode="fixed" and delegates all Arnoldi work to
% run_left_preconditioned_tucker_gmres_cycle.

if nargin < 10
    displayProgress = false;
end

if nargin < 11
    maximumBasisStorageEntries = Inf;
end

if nargin < 12
    maximumMultilinearRank = [];
end
if nargin < 13
    diagnosticIterations = [];
end

[U, info] = run_left_preconditioned_tucker_gmres_cycle( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, targetTolerance, compressionTolerance, "fixed", ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank, targetTolerance, 1 - eps, ...
    diagnosticIterations);

end
