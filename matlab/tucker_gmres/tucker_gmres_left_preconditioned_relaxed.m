function [U, info] = tucker_gmres_left_preconditioned_relaxed( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, targetTolerance, ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank)
%TUCKER_GMRES_LEFT_PRECONDITIONED_RELAXED Apply the Section 5.5 rule.
%
% One left-preconditioned Tucker-Arnoldi cycle is built. At step j, the
% operator product and Gram--Schmidt updates use
%
%     eta_j = targetTolerance ...
%             / (||computed preconditioned residual_{j-1}|| / beta).
%
% The starting residual and partial solution sums use targetTolerance. The
% original unpreconditioned true residual is evaluated independently and is
% the final convergence test.
%
% Thesis notation (Section 5.5):
%   originalOperatorFunction       <->  \mathcal A_0
%   preconditionerFunction         <->  \mathcal P
%   F0, U0, U                      <->  \mathcal B_0, \mathcal X_0, \mathcal X_j
%   maximumIteration               <->  \ell_max
%   targetTolerance                <->  \varepsilon
%   maximumMultilinearRank         <->  \boldsymbol R=(R_1,...,R_d)
%   computed residual / beta       <->  ||\widetilde r_(j-1)^P||_2/\beta
% The wrapper sets toleranceMode="relaxed" and uses targetTolerance as the
% numerator in \eta_j=\varepsilon/(||\widetilde r_(j-1)^P||_2/\beta).

if nargin < 9
    displayProgress = false;
end

if nargin < 10
    maximumBasisStorageEntries = Inf;
end

if nargin < 11
    maximumMultilinearRank = [];
end

[U, info] = run_left_preconditioned_tucker_gmres_cycle( ...
    originalOperatorFunction, preconditionerFunction, F0, U0, ...
    maximumIteration, targetTolerance, targetTolerance, "relaxed", ...
    originalTrueResidualFunction, preconditionedResidualNormFunction, ...
    displayProgress, maximumBasisStorageEntries, ...
    maximumMultilinearRank);

end
