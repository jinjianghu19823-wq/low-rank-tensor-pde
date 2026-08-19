function [Z, info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_4d( ...
    Y, preconditioner, compressionTolerance, maximumMultilinearRank)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_4D Apply P_r.

callTimer = tic;
if ~isa(Y, 'ttensor') || ndims(Y) ~= 4
    error('Y must be an order four Tucker tensor.');
end
if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end
if nargin < 4
    maximumMultilinearRank = [];
end

forwardFactors = cell(4, 1);
componentTimer = tic;
for mode = 1:4
    forwardFactors{mode} = dst(Y.u{mode});
end
forwardDstTime = toc(componentTimer);
spectralY = ttensor(Y.core, forwardFactors);

componentTimer = tic;
exactProduct = tucker_hadamard_exact_nd( ...
    preconditioner.inverse_eigenvalue_tensor, spectralY);
exactHadamardTime = toc(componentTimer);
temporaryRanks = double(size(exactProduct.core));
[roundedProduct, roundInfo] = tucker_round( ...
    exactProduct, compressionTolerance, maximumMultilinearRank);

inverseFactors = cell(4, 1);
componentTimer = tic;
for mode = 1:4
    inverseFactors{mode} = idst(roundedProduct.u{mode});
end
inverseDstTime = toc(componentTimer);
Z = ttensor(roundedProduct.core, inverseFactors);

info.requested_tolerance = compressionTolerance;
info.inverse_eigenvalue_ranks = ...
    preconditioner.inverse_eigenvalue_ranks;
info.input_ranks = double(size(Y.core));
info.temporary_hadamard_ranks = temporaryRanks;
info.output_ranks = double(size(Z.core));
info.rank_cap_active = roundInfo.rank_cap_active;
info.relative_error_estimate = roundInfo.relative_error_estimate;
info.kernel_timing = roundInfo.kernel_timing;
info.kernel_timing.preconditioner_forward_dst_time_sec = ...
    info.kernel_timing.preconditioner_forward_dst_time_sec + forwardDstTime;
info.kernel_timing.preconditioner_exact_hadamard_time_sec = ...
    info.kernel_timing.preconditioner_exact_hadamard_time_sec + ...
    exactHadamardTime;
info.kernel_timing.preconditioner_inverse_dst_time_sec = ...
    info.kernel_timing.preconditioner_inverse_dst_time_sec + inverseDstTime;
info.call_time_sec = toc(callTimer);

end
