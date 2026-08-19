function [Y, info] = poisson_action_tucker(X, A1, compressionTolerance, maximumMultilinearRank)
%POISSON_ACTION_TUCKER Apply the d-dimensional Poisson operator.
%
% This function calculates
%
%     L(X) = X x_1 A1 + X x_2 A1 + ... + X x_d A1,
%
% where x_n denotes multiplication in tensor mode n.
%
% Inputs:
%   X
%       Input tensor stored as a Tensor Toolbox ttensor.
%
%   A1
%       One-dimensional finite-difference matrix.
%
%   compressionTolerance
%       Relative tolerance used after adding the Tucker terms.
%
%   maximumMultilinearRank (optional)
%       Scalar or mode-wise hard rank cap used after every addition.
%
% Output:
%   Y
%       Compressed Tucker representation of L(X).
%
%   info
%       Diagnostics for the sequential additions.
%
% Thesis notation (Sections 5.5 and 6.2):
%   X, Y                        <->  \mathcal X, \mathcal A_0(\mathcal X)
%   A1                          <->  A_1
%   compressionTolerance       <->  \eta or \eta_j
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
%   nextTerm                   <->  \mathcal X \times_n A_1
%   numberOfModes              <->  d
% Here \mathcal A_0(\mathcal X)=\sum_{n=1}^d \mathcal X \times_n A_1.

callTimer = tic;


%% 1. Check the inputs

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

if compressionTolerance <= 0 || compressionTolerance >= 1
    error('compressionTolerance must be between 0 and 1.');
end

if nargin < 4
    maximumMultilinearRank = [];
end

maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, size(X));

numberOfModes = ndims(X);
tensorSize = size(X);

for mode = 1:numberOfModes

    if tensorSize(mode) ~= size(A1, 1)
        error('Every mode of X must have the same size as A1.');
    end

end


%% 2. Apply A1 in the first mode

% Tensor Toolbox's ttm command performs a tensor-times-matrix product.
% The third input, 1, says that A1 acts in mode 1.
componentTimer = tic;
Y = ttm(X, A1, 1);
modeProductTime = toc(componentTimer);

kernelTiming = empty_tucker_kernel_timing();

additionCapActive = false(numberOfModes - 1, 1);
additionErrorEstimates = zeros(numberOfModes - 1, 1);


%% 3. Add the contributions from the remaining modes

for mode = 2:numberOfModes

    % Calculate the next term X x_mode A1.
    componentTimer = tic;
    nextTerm = ttm(X, A1, mode);
    modeProductTime = modeProductTime + toc(componentTimer);

    % Add the new term to the running result and recompress it.
    [Y, additionInfo] = tucker_axpby_round(Y, 1, nextTerm, 1, compressionTolerance, maximumRanks);

    kernelTiming = add_tucker_kernel_timing( ...
        kernelTiming, additionInfo.kernel_timing);

    additionCapActive(mode - 1) = additionInfo.rank_cap_active;
    additionErrorEstimates(mode - 1) = ...
        additionInfo.relative_error_estimate;

end

info.requested_tolerance = compressionTolerance;
info.maximum_ranks = maximumRanks;
info.addition_rank_cap_active = additionCapActive;
info.addition_relative_error_estimate = additionErrorEstimates;
info.rank_cap_active = any(additionCapActive);
info.maximum_local_recompression_error = ...
    max([0; additionErrorEstimates]);
kernelTiming.poisson_mode_product_time_sec = ...
    kernelTiming.poisson_mode_product_time_sec + modeProductTime;
info.kernel_timing = kernelTiming;
info.call_time_sec = toc(callTimer);

end
