function [Z, info] = apply_poisson_fast_diagonalization_tucker_preconditioner(Y, preconditioner, compressionTolerance, maximumMultilinearRank)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Apply M_r.
%
% The action has three stages:
%
%   1. apply DST-I to every Tucker factor matrix;
%   2. multiply entrywise by the Tucker inverse-eigenvalue tensor;
%   3. recompress and apply the inverse DST-I to every factor matrix.
%
% The Hadamard product is represented exactly before the single requested
% recompression. Therefore, unlike the sequential exponential-sum action,
% this routine does not perform one rounded addition per exponential term.
%
% Thesis/experiment notation (Section 6.2):
%   Y, Z                        <->  \mathcal Y, \mathcal P_r(\mathcal Y)
%   preconditioner.inverse_eigenvalue_tensor
%                               <->  D_r
%   spectralY                  <->  DST(\mathcal Y)
%   exactSpectralProduct       <->  D_r .* DST(\mathcal Y)
%   roundedSpectralProduct     <->  recompressed spectral product
%   compressionTolerance       <->  \eta or \eta_j
%   maximumMultilinearRank     <->  \boldsymbol R=(R_1,...,R_d)
% Hence Z=IDST(D_r .* DST(Y)), up to the requested recompression.

callTimer = tic;


%% 1. Check the inputs

if ~isa(Y, 'ttensor')
    error('Y must be a Tensor Toolbox ttensor object.');
end

if ndims(Y) ~= 3
    error('The fast-diagonalization preconditioner requires order 3.');
end

if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end

if nargin < 4
    maximumMultilinearRank = [];
end


%% 2. Transform the Tucker factors into the sine basis

forwardFactors = cell(3, 1);
componentTimer = tic;

for mode = 1:3
    forwardFactors{mode} = dst(Y.u{mode});
end

forwardDstTime = toc(componentTimer);

spectralY = ttensor(Y.core, forwardFactors);


%% 3. Multiply by the compressed inverse-eigenvalue tensor

componentTimer = tic;
exactSpectralProduct = tucker_hadamard_exact(preconditioner.inverse_eigenvalue_tensor, spectralY);
exactHadamardTime = toc(componentTimer);
temporaryRanks = size(exactSpectralProduct.core);

[roundedSpectralProduct, roundInfo] = tucker_round(exactSpectralProduct, compressionTolerance, maximumMultilinearRank);


%% 4. Transform back to the physical grid

inverseFactors = cell(3, 1);
componentTimer = tic;

for mode = 1:3
    inverseFactors{mode} = idst(roundedSpectralProduct.u{mode});
end


inverseDstTime = toc(componentTimer);

Z = ttensor(roundedSpectralProduct.core, inverseFactors);


%% 5. Return diagnostics

info.requested_tolerance = compressionTolerance;
info.inverse_eigenvalue_ranks = ...
    preconditioner.inverse_eigenvalue_ranks;
info.input_ranks = size(Y.core);
info.temporary_hadamard_ranks = temporaryRanks;
info.output_ranks = size(Z.core);
info.rank_cap_active = roundInfo.rank_cap_active;
info.relative_error_estimate = roundInfo.relative_error_estimate;

kernelTiming = roundInfo.kernel_timing;
kernelTiming.preconditioner_forward_dst_time_sec = ...
    kernelTiming.preconditioner_forward_dst_time_sec + forwardDstTime;
kernelTiming.preconditioner_exact_hadamard_time_sec = ...
    kernelTiming.preconditioner_exact_hadamard_time_sec + exactHadamardTime;
kernelTiming.preconditioner_inverse_dst_time_sec = ...
    kernelTiming.preconditioner_inverse_dst_time_sec + inverseDstTime;

info.kernel_timing = kernelTiming;
info.call_time_sec = toc(callTimer);

end
