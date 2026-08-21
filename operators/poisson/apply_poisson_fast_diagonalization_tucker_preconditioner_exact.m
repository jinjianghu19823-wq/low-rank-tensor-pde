function [Z, info] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner_exact( ...
    Y, preconditioner)
%APPLY_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_EXACT Apply P.
%
% This fixed linear version performs the sine transforms and the exact
% Tucker Hadamard product without recompression. It is intended for forming
% the columns of the sketched residual problem. Any later rounding is the
% responsibility of the calling Krylov method.

callTimer = tic;

if ~isa(Y, 'ttensor')
    error('Y must be a Tensor Toolbox ttensor object.');
end

if ndims(Y) ~= 3
    error('The fast diagonalization preconditioner requires order 3.');
end

if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must match the preconditioner size.');
end

forwardFactors = cell(3, 1);
componentTimer = tic;

for mode = 1:3
    forwardFactors{mode} = dst(Y.u{mode});
end

forwardDstTime = toc(componentTimer);
spectralY = ttensor(Y.core, forwardFactors);

componentTimer = tic;
exactSpectralProduct = tucker_hadamard_exact( ...
    preconditioner.inverse_eigenvalue_tensor, spectralY);
exactHadamardTime = toc(componentTimer);

inverseFactors = cell(3, 1);
componentTimer = tic;

for mode = 1:3
    inverseFactors{mode} = idst(exactSpectralProduct.u{mode});
end

inverseDstTime = toc(componentTimer);
Z = ttensor(exactSpectralProduct.core, inverseFactors);

info.input_ranks = size(Y.core);
info.output_ranks = size(Z.core);
info.rounding_performed = false;
info.rank_cap_active = false;
info.relative_error_estimate = 0;
info.forward_dst_time_sec = forwardDstTime;
info.exact_hadamard_time_sec = exactHadamardTime;
info.inverse_dst_time_sec = inverseDstTime;
info.call_time_sec = toc(callTimer);

end
