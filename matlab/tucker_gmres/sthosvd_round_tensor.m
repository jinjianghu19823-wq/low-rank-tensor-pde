function [T, info] = sthosvd_round_tensor(X, relativeTolerance, maximumMultilinearRank, modeOrder)
%STHOSVD_ROUND_TENSOR Recompress a small tensor with tolerance and rank cap.
%
% This routine implements the STHOSVD step used inside Tucker
% recompression. For each processed mode, it selects the smallest rank whose
% discarded squared singular values fit inside the mode error budget
%
%     relativeTolerance^2 * norm(X)^2 / d.
%
% The retained rank is then limited by maximumMultilinearRank. If that cap
% is smaller than the tolerance-selected rank, the cap is marked active and
% the requested tolerance is no longer guaranteed.
%
% Inputs:
%   X
%       Small Tensor Toolbox tensor, usually the orthogonalised Tucker core.
%
%   relativeTolerance
%       Requested global relative STHOSVD tolerance.
%
%   maximumMultilinearRank
%       Optional scalar or mode-wise maximum rank. Use [] for no additional
%       cap.
%
%   modeOrder
%       Optional permutation describing the sequential processing order.
%
% Outputs:
%   T
%       Recompressed ttensor in the coordinate space of X.
%
%   info
%       Selected ranks, tolerance ranks, discarded energy, and whether the
%       rank cap was active.
%
% Thesis notation (Section 5.5 and Algorithm Tucker AXPBY):
%   X, workingCore             <->  \mathcal C^(0), \mathcal C^(j-1)
%   relativeTolerance          <->  \eta
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
%   modeOrder                  <->  \rho=(\rho_1,...,\rho_d)
%   mode, orderIndex           <->  n=\rho_j, j
%   unfolding                  <->  C_(n)^(j-1)
%   leftVectors                <->  L^(n)
%   toleranceRank              <->  s_n^(\eta)
%   retainedRank               <->  s_n=min{s_n^(\eta),R_n}
%   factorMatrices{mode}       <->  P^(n)
%   T                          <->  recompressed core tensor

callTimer = tic;


%% 1. Check the inputs

if ~isa(X, 'tensor')
    error('X must be a Tensor Toolbox tensor object.');
end

if ~isscalar(relativeTolerance) || relativeTolerance <= 0 || relativeTolerance >= 1
    error('relativeTolerance must be one number between 0 and 1.');
end

numberOfModes = ndims(X);
tensorDimensions = size(X);

if nargin < 3
    maximumMultilinearRank = [];
end

if nargin < 4 || isempty(modeOrder)
    modeOrder = 1:numberOfModes;
end

modeOrder = double(modeOrder(:).');

if ~isequal(sort(modeOrder), 1:numberOfModes)
    error('modeOrder must contain every tensor mode exactly once.');
end

maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, tensorDimensions);


%% 2. Define the global and mode-wise error budgets

inputNorm = norm(X);
modeErrorThreshold = relativeTolerance * inputNorm / sqrt(numberOfModes);
modeErrorBudget = modeErrorThreshold^2;

factorMatrices = cell(numberOfModes, 1);
toleranceRanks = zeros(1, numberOfModes);
retainedRanks = zeros(1, numberOfModes);
discardedNorms = zeros(1, numberOfModes);
discardedEnergy = zeros(1, numberOfModes);
rankCapActiveByMode = false(1, numberOfModes);
unfoldingTimeByMode = zeros(1, numberOfModes);
svdTimeByMode = zeros(1, numberOfModes);
rankSelectionTimeByMode = zeros(1, numberOfModes);
projectionTimeByMode = zeros(1, numberOfModes);

workingCore = X;


%% 3. Process the unfoldings sequentially

for orderIndex = 1:numberOfModes

    mode = modeOrder(orderIndex);

    componentTimer = tic;
    unfolding = double(tenmat(workingCore, mode));
    unfoldingTimeByMode(mode) = toc(componentTimer);

    % The left singular vectors span the current mode subspace.
    componentTimer = tic;
    [leftVectors, singularValueMatrix, ~] = svd(unfolding, 'econ');
    svdTimeByMode(mode) = toc(componentTimer);

    componentTimer = tic;
    singularValues = diag(singularValueMatrix);

    % tailNormAfterRank(r) is the Frobenius norm discarded after retaining
    % the first r singular vectors. Build these norms backwards with hypot.
    % This avoids subtracting two nearly equal accumulated energies when
    % the requested squared tolerance is close to machine precision.
    tailNormAfterRank = zeros(size(singularValues));

    for rankIndex = numel(singularValues)-1:-1:1
        tailNormAfterRank(rankIndex) = hypot(singularValues(rankIndex + 1), tailNormAfterRank(rankIndex + 1));
    end

    toleranceRank = find(tailNormAfterRank <= modeErrorThreshold, 1, 'first');

    if isempty(toleranceRank)
        toleranceRank = numel(singularValues);
    end

    retainedRank = min(toleranceRank, maximumRanks(mode));

    toleranceRanks(mode) = toleranceRank;
    retainedRanks(mode) = retainedRank;
    rankCapActiveByMode(mode) = retainedRank < toleranceRank;
    discardedNorms(mode) = tailNormAfterRank(retainedRank);
    discardedEnergy(mode) = discardedNorms(mode)^2;

    factorMatrices{mode} = leftVectors(:, 1:retainedRank);
    rankSelectionTimeByMode(mode) = toc(componentTimer);

    % Compress immediately so that the next unfolding is formed from the
    % already reduced core.
    componentTimer = tic;
    workingCore = ttm(workingCore, factorMatrices{mode}.', mode);
    projectionTimeByMode(mode) = toc(componentTimer);

end


%% 4. Return the Tucker tensor and error diagnostics

T = ttensor(workingCore, factorMatrices);

if inputNorm == 0
    relativeErrorEstimate = 0;
else
    relativeErrorEstimate = norm(discardedNorms) / inputNorm;
end

info.requested_tolerance = relativeTolerance;
info.mode_error_threshold = modeErrorThreshold;
info.mode_error_budget = modeErrorBudget;
info.maximum_ranks = maximumRanks;
info.tolerance_ranks = toleranceRanks;
info.retained_ranks = retainedRanks;
info.discarded_norm = discardedNorms;
info.discarded_energy = discardedEnergy;
info.relative_error_estimate = relativeErrorEstimate;
info.rank_cap_active_by_mode = rankCapActiveByMode;
info.rank_cap_active = any(rankCapActiveByMode);
info.mode_order = modeOrder;

kernelTiming = empty_tucker_kernel_timing();
kernelTiming.sthosvd_unfolding_time_sec = sum(unfoldingTimeByMode);
kernelTiming.sthosvd_svd_time_sec = sum(svdTimeByMode);
kernelTiming.sthosvd_rank_selection_time_sec = ...
    sum(rankSelectionTimeByMode);
kernelTiming.sthosvd_projection_time_sec = sum(projectionTimeByMode);

info.kernel_timing = kernelTiming;
info.call_time_sec = toc(callTimer);
info.unfolding_time_by_mode_sec = unfoldingTimeByMode;
info.svd_time_by_mode_sec = svdTimeByMode;
info.rank_selection_time_by_mode_sec = rankSelectionTimeByMode;
info.projection_time_by_mode_sec = projectionTimeByMode;

end
