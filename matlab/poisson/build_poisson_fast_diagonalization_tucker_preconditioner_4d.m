function [preconditioner, info] = ...
    build_poisson_fast_diagonalization_tucker_preconditioner_4d( ...
    A1, inverseEigenvalueRank)
%BUILD_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER_4D Build P_r.
%
% The four dimensional inverse multiplier is
%
%   D(i,j,k,l) = 1/(lambda_i+lambda_j+lambda_k+lambda_l).
%
% This proof of concept builder forms D explicitly before its Tucker HOSVD.

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be square.');
end
N = size(A1, 1);
maximumRanks = normalise_tucker_rank_cap( ...
    inverseEigenvalueRank, [N, N, N, N]);

diagonalCoefficient = full(A1(1, 1));
if N > 1
    offDiagonalCoefficient = full(A1(1, 2));
else
    offDiagonalCoefficient = 0;
end
referenceA1 = spdiags( ...
    [offDiagonalCoefficient * ones(N, 1), ...
     diagonalCoefficient * ones(N, 1), ...
     offDiagonalCoefficient * ones(N, 1)], -1:1, N, N);
if norm(A1 - referenceA1, 'fro') > ...
        1e-12 * max(1, norm(A1, 'fro'))
    error('A1 must be the constant coefficient Dirichlet Poisson matrix.');
end

setupTimer = tic;
indices = (1:N).';
lambda = diagonalCoefficient + ...
    2 * offDiagonalCoefficient * cos(indices * pi / (N + 1));
if any(lambda <= 0)
    error('A1 must be positive definite.');
end

lambdaSum = reshape(lambda, N, 1, 1, 1) + ...
    reshape(lambda, 1, N, 1, 1) + ...
    reshape(lambda, 1, 1, N, 1) + ...
    reshape(lambda, 1, 1, 1, N);
exactInverse = 1 ./ lambdaSum;

compressionTimer = tic;
inverseTensor = hosvd(tensor(exactInverse), 1e-14, ...
    'ranks', maximumRanks, 'sequential', false, 'verbosity', 0);
compressionTime = toc(compressionTimer);

diagnosticTimer = tic;
approximateInverse = double(full(inverseTensor));
preconditionedEigenvalues = lambdaSum .* approximateInverse;
relativeInverseError = norm(exactInverse(:) - approximateInverse(:)) / ...
    norm(exactInverse(:));
minimumEigenvalue = min(preconditionedEigenvalues(:));
maximumEigenvalue = max(preconditionedEigenvalues(:));
spectralDelta = max(abs(1 - preconditionedEigenvalues(:)));
if minimumEigenvalue > 0
    conditionNumber = maximumEigenvalue / minimumEigenvalue;
else
    conditionNumber = NaN;
end
diagnosticTime = toc(diagnosticTimer);

retainedRanks = double(size(inverseTensor.core));
storageEntries = numel(inverseTensor.core);
for mode = 1:4
    storageEntries = storageEntries + numel(inverseTensor.u{mode});
end

preconditioner.N = N;
preconditioner.number_of_modes = 4;
preconditioner.transform = 'DST-I';
preconditioner.one_dimensional_eigenvalues = lambda;
preconditioner.inverse_eigenvalue_tensor = inverseTensor;
preconditioner.inverse_eigenvalue_ranks = retainedRanks;

info.N = N;
info.requested_ranks = maximumRanks;
info.retained_ranks = retainedRanks;
info.inverse_eigenvalue_relative_frobenius_error = relativeInverseError;
info.minimum_preconditioned_eigenvalue = minimumEigenvalue;
info.maximum_preconditioned_eigenvalue = maximumEigenvalue;
info.spectral_delta = spectralDelta;
info.is_positive_definite = minimumEigenvalue > 0;
info.condition_number = conditionNumber;
info.storage_entries = storageEntries;
info.storage_mib = 8 * storageEntries / 2^20;
info.full_spectral_storage_gib = 8 * N^4 / 2^30;
info.compression_time_sec = compressionTime;
info.diagnostic_time_sec = diagnosticTime;
info.setup_time_sec = toc(setupTimer);

end
