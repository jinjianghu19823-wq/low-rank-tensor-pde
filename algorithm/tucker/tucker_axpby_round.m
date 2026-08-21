function [Z, info] = tucker_axpby_round(X, alpha, Y, beta, compressionTolerance, maximumMultilinearRank)
%TUCKER_AXPBY_ROUND Add two Tucker tensors and recompress the result.
%
% This function calculates
%
%     Z = round(alpha * X + beta * Y).
%
% The name AXPBY comes from the familiar vector operation
%
%     alpha * x + beta * y.
%
% Inputs:
%   X, Y
%       Tucker tensors stored as Tensor Toolbox ttensor objects.
%
%   alpha, beta
%       Scalar coefficients multiplying X and Y.
%
%   compressionTolerance
%       Requested relative STHOSVD recompression tolerance.
%
%   maximumMultilinearRank (optional)
%       Scalar or mode-wise hard rank cap. Use [] for no additional cap.
%
% Output:
%   Z
%       The recompressed Tucker tensor.
%
%   info
%       Recompression diagnostics, including whether the rank cap was
%       active.
%
% The full tensors are never formed. The exact Tucker sum is first written
% with joined factor matrices and a block core. QR factorisations then make
% the joined factor matrices orthonormal. Finally, STHOSVD is applied only
% to the much smaller transformed core. Tensor Toolbox exposes this method
% through its HOSVD function with the sequential option enabled.
%
% Thesis notation (Section 5.5 and Algorithm Tucker AXPBY):
%   X, Y, Z                    <->  \mathcal X, \mathcal Y, \mathcal Z
%   alpha                      <->  \alpha
%   beta                       <->  \gamma (the AXPBY coefficient)
%   compressionTolerance       <->  \eta
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
%   combinedCore               <->  \widehat{\mathcal G}
%   combinedFactors{mode}      <->  \widehat U^(n)
%   orthogonalFactors{mode}    <->  Q^(n)
%   orthogonalCore             <->  \mathcal G_Q=\mathcal C^(0)
%   compressedCoreTensor.u{n}  <->  P^(n)
%   finalFactors{n}            <->  U_Z^(n)=Q^(n)P^(n)
% Here beta is an AXPBY coefficient; it is not the GMRES quantity
% \beta=||\widetilde r_0||.

callTimer = tic;


%% 1. Check the inputs

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(X), size(Y))
    error('X and Y must have the same tensor dimensions.');
end

if ~isscalar(alpha) || ~isscalar(beta)
    error('alpha and beta must be scalar numbers.');
end

if compressionTolerance <= 0 || compressionTolerance >= 1
    error('compressionTolerance must be between 0 and 1.');
end

if nargin < 6
    maximumMultilinearRank = [];
end

maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, size(X));


%% 2. Read the Tucker ranks

numberOfModes = ndims(X);

rankX = size(X.core);
rankY = size(Y.core);

% Adding Tucker tensors joins their mode subspaces. Before recompression,
% the new rank in each mode is the sum of the two old ranks.
combinedCoreSize = rankX + rankY;


%% 3. Build the exact block core of alpha*X + beta*Y

componentTimer = tic;

% Start with a core filled with zeros.
combinedCoreValues = zeros(combinedCoreSize);

% These cell arrays describe the two diagonal blocks of the new core.
indexX = cell(numberOfModes, 1);
indexY = cell(numberOfModes, 1);

for mode = 1:numberOfModes

    % The first block stores the core of X.
    indexX{mode} = 1:rankX(mode);

    % The second block stores the core of Y.
    indexY{mode} = rankX(mode) + (1:rankY(mode));

end

% Place alpha times the core of X in the first block.
combinedCoreValues(indexX{:}) = alpha * double(X.core);

% Place beta times the core of Y in the second block.
combinedCoreValues(indexY{:}) = beta * double(Y.core);

combinedCore = tensor(combinedCoreValues);
exactSumCoreTime = toc(componentTimer);


%% 4. Join the factor matrices

componentTimer = tic;
combinedFactors = cell(numberOfModes, 1);

for mode = 1:numberOfModes

    % Horizontal concatenation joins the two mode subspaces:
    %
    %     [U_X^(mode), U_Y^(mode)].
    combinedFactors{mode} = [X.u{mode}, Y.u{mode}];

end

factorConcatenationTime = toc(componentTimer);


%% 5. Orthonormalise the joined factor matrices

orthogonalFactors = cell(numberOfModes, 1);
orthogonalCore = combinedCore;
factorQrTime = 0;
coreTransformTime = 0;

for mode = 1:numberOfModes

    % Thin QR gives
    %
    %     combinedFactors{mode} = Q * R.
    componentTimer = tic;
    [Q, R] = qr(combinedFactors{mode}, 0);
    factorQrTime = factorQrTime + toc(componentTimer);

    orthogonalFactors{mode} = Q;

    % The R factor must be absorbed into the core so that the represented
    % full tensor does not change.
    componentTimer = tic;
    orthogonalCore = ttm(orthogonalCore, R, mode);
    coreTransformTime = coreTransformTime + toc(componentTimer);

end


%% 6. Handle an exact or nearly exact cancellation

sizeBeforeCancellation = abs(alpha) * norm(X) + abs(beta) * norm(Y);

sizeAfterCancellation = norm(orthogonalCore);

if sizeAfterCancellation <= 100 * eps * sizeBeforeCancellation

    % For example, X - X is the zero tensor. Multiplying X by zero keeps a
    % valid ttensor representation without constructing a full zero array.
    Z = 0 * X;

    info.requested_tolerance = compressionTolerance;
    info.maximum_ranks = maximumRanks;
    info.tolerance_ranks = size(X.core);
    info.retained_ranks = size(X.core);
    info.relative_error_estimate = 0;
    info.rank_cap_active_by_mode = false(1, numberOfModes);
    info.rank_cap_active = false;
    info.zero_result = true;
    kernelTiming = empty_tucker_kernel_timing();
    kernelTiming.exact_sum_core_time_sec = exactSumCoreTime;
    kernelTiming.factor_concatenation_time_sec = ...
        factorConcatenationTime;
    kernelTiming.factor_qr_time_sec = factorQrTime;
    kernelTiming.core_transform_time_sec = coreTransformTime;
    info.kernel_timing = kernelTiming;
    info.call_time_sec = toc(callTimer);
    return

end


%% 7. Compress the small orthogonal core with STHOSVD

[compressedCoreTensor, coreInfo] = sthosvd_round_tensor(orthogonalCore, compressionTolerance, maximumRanks, 1:numberOfModes);


%% 8. Combine the two levels of factor matrices

finalFactors = cell(numberOfModes, 1);
factorReconstructionTime = 0;

for mode = 1:numberOfModes

    % STHOSVD supplies a small factor matrix for the transformed core. It is
    % multiplied by the earlier QR factor to obtain a factor matrix in the
    % original tensor space.
    componentTimer = tic;
    finalFactors{mode} = orthogonalFactors{mode} * compressedCoreTensor.u{mode};
    factorReconstructionTime = ...
        factorReconstructionTime + toc(componentTimer);

end


%% 9. Return the final Tucker tensor

Z = ttensor(compressedCoreTensor.core, finalFactors);

info = coreInfo;
info.zero_result = false;

kernelTiming = coreInfo.kernel_timing;
kernelTiming.exact_sum_core_time_sec = ...
    kernelTiming.exact_sum_core_time_sec + exactSumCoreTime;
kernelTiming.factor_concatenation_time_sec = ...
    kernelTiming.factor_concatenation_time_sec + factorConcatenationTime;
kernelTiming.factor_qr_time_sec = ...
    kernelTiming.factor_qr_time_sec + factorQrTime;
kernelTiming.core_transform_time_sec = ...
    kernelTiming.core_transform_time_sec + coreTransformTime;
kernelTiming.factor_reconstruction_time_sec = ...
    kernelTiming.factor_reconstruction_time_sec + factorReconstructionTime;

info.kernel_timing = kernelTiming;
info.call_time_sec = toc(callTimer);

end
