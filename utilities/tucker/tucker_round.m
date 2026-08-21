function [Z, info] = tucker_round( ...
    X, compressionTolerance, maximumMultilinearRank)
%TUCKER_ROUND Recompress one Tucker tensor without forming its full array.
%
% The factor matrices of X are first orthogonalised by thin QR
% factorisations. Their R factors are absorbed into the core. STHOSVD is
% then applied only to this small orthogonalised core.
%
% The optional maximum multilinear rank is a hard cap. If it is smaller
% than the rank needed by compressionTolerance, info.rank_cap_active is
% true and the requested tolerance is not guaranteed.
%
% Thesis notation (Section 5.5):
%   X, Z                       <->  \mathcal X,
%                                   \mathcal T_{\eta,\boldsymbol R}(\mathcal X)
%   compressionTolerance       <->  \eta
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
%   X.core                     <->  \mathcal G
%   X.u{mode}                  <->  U^(n)
%   Q, R                       <->  Q^(n), R^(n), with U^(n)=Q^(n)R^(n)
%   orthogonalCore             <->  \mathcal G_Q=\mathcal C^(0)
%   compressedCoreTensor.u{n}  <->  P^(n)
%   finalFactors{n}            <->  U_Z^(n)=Q^(n)P^(n)

callTimer = tic;


%% 1. Check the inputs

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

if ~isscalar(compressionTolerance) || compressionTolerance <= 0 || compressionTolerance >= 1
    error('compressionTolerance must be one number between 0 and 1.');
end

if nargin < 3
    maximumMultilinearRank = [];
end

numberOfModes = ndims(X);
maximumRanks = normalise_tucker_rank_cap(maximumMultilinearRank, size(X));


%% 2. Orthogonalise the factor matrices

orthogonalFactors = cell(numberOfModes, 1);
orthogonalCore = X.core;
factorQrTime = 0;
coreTransformTime = 0;

for mode = 1:numberOfModes

    componentTimer = tic;
    [Q, R] = qr(X.u{mode}, 0);
    factorQrTime = factorQrTime + toc(componentTimer);

    orthogonalFactors{mode} = Q;

    componentTimer = tic;
    orthogonalCore = ttm(orthogonalCore, R, mode);
    coreTransformTime = coreTransformTime + toc(componentTimer);

end


%% 3. Handle an exact zero tensor

if norm(orthogonalCore) == 0
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
    kernelTiming.factor_qr_time_sec = factorQrTime;
    kernelTiming.core_transform_time_sec = coreTransformTime;
    info.kernel_timing = kernelTiming;
    info.call_time_sec = toc(callTimer);
    return
end


%% 4. Compress the orthogonalised core

[compressedCoreTensor, coreInfo] = sthosvd_round_tensor(orthogonalCore, compressionTolerance, maximumRanks, 1:numberOfModes);

finalFactors = cell(numberOfModes, 1);
factorReconstructionTime = 0;

for mode = 1:numberOfModes
    componentTimer = tic;
    finalFactors{mode} = orthogonalFactors{mode} * compressedCoreTensor.u{mode};
    factorReconstructionTime = ...
        factorReconstructionTime + toc(componentTimer);
end

Z = ttensor(compressedCoreTensor.core, finalFactors);

info = coreInfo;
info.zero_result = false;

kernelTiming = coreInfo.kernel_timing;
kernelTiming.factor_qr_time_sec = ...
    kernelTiming.factor_qr_time_sec + factorQrTime;
kernelTiming.core_transform_time_sec = ...
    kernelTiming.core_transform_time_sec + coreTransformTime;
kernelTiming.factor_reconstruction_time_sec = ...
    kernelTiming.factor_reconstruction_time_sec + factorReconstructionTime;

info.kernel_timing = kernelTiming;
info.call_time_sec = toc(callTimer);

end
