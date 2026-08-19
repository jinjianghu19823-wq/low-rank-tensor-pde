function [preconditioner, info] = build_poisson_fast_diagonalization_tucker_preconditioner(A1, inverseEigenvalueRank)
%BUILD_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Build M_r.
%
% The three-dimensional Dirichlet finite-difference Poisson matrix is
% diagonal in the DST-I basis. This function constructs the exact spectral
% inverse tensor
%
%   D(i,j,k) = 1 / (lambda_i + lambda_j + lambda_k)
%
% and approximates it by a fixed multilinear-rank Tucker tensor. The
% resulting preconditioner is intended for the fast-diagonalization action
%
%   P(Y) = IDST( D_r .* DST(Y) ).
%
% This is a proof-of-concept builder for the three-dimensional dissertation
% experiment. It forms the N-by-N-by-N inverse-eigenvalue tensor during
% setup, but stores only its Tucker approximation afterward.
%
% Thesis/experiment notation (Section 6.2):
%   A1                          <->  A_1
%   N                           <->  number of interior points per mode
%   oneDimensionalEigenvalues  <->  lambda_i
%   lambdaSum(i,j,k)           <->  lambda_i+lambda_j+lambda_k
%   exactInverseEigenvalues    <->  D(i,j,k)=1/(lambda_i+lambda_j+lambda_k)
%   inverseEigenvalueRank      <->  prescribed Tucker rank r
%   inverseEigenvalueTensor    <->  D_r
%   preconditioner             <->  reusable representation of \mathcal P_r
%   spectralDelta              <->  delta=max|1-(lambda_i+lambda_j+lambda_k)D_r|
%   conditionNumber            <->  predicted kappa_2(M_r A_0)


%% 1. Check the inputs and the one-dimensional matrix

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

N = size(A1, 1);
maximumRanks = normalise_tucker_rank_cap(inverseEigenvalueRank, [N, N, N]);

diagonalCoefficient = full(A1(1, 1));

if N > 1
    offDiagonalCoefficient = full(A1(1, 2));
else
    offDiagonalCoefficient = 0;
end

referenceA1 = spdiags([offDiagonalCoefficient * ones(N, 1), diagonalCoefficient * ones(N, 1), offDiagonalCoefficient * ones(N, 1)], -1:1, N, N);

matrixMismatch = norm(A1 - referenceA1, 'fro') / max(norm(A1, 'fro'), eps);

if matrixMismatch > 1e-12
    error(['A1 must be the symmetric constant-coefficient Dirichlet ', ...
           'tridiagonal matrix diagonalised by DST-I.']);
end

if offDiagonalCoefficient >= 0
    error('The Dirichlet Poisson off-diagonal coefficient must be negative.');
end


%% 2. Construct the exact inverse eigenvalues

setupTimer = tic;

indices = (1:N).';
oneDimensionalEigenvalues = diagonalCoefficient + 2 * offDiagonalCoefficient * cos(indices * pi / (N + 1));

if any(oneDimensionalEigenvalues <= 0)
    error('A1 must be positive definite.');
end

lambdaSum = oneDimensionalEigenvalues + reshape(oneDimensionalEigenvalues, 1, N) + reshape(oneDimensionalEigenvalues, 1, 1, N);
exactInverseEigenvalues = 1 ./ lambdaSum;


%% 3. Compress the spectral multiplier at the prescribed rank

compressionTimer = tic;

inverseEigenvalueTensor = hosvd(tensor(exactInverseEigenvalues), 1e-14, 'ranks', maximumRanks, 'sequential', false, 'verbosity', 0);

compressionTime = toc(compressionTimer);


%% 4. Evaluate setup diagnostics

diagnosticTimer = tic;

approximatedInverseEigenvalues = double(full(inverseEigenvalueTensor));
preconditionedEigenvalues = lambdaSum .* approximatedInverseEigenvalues;

inverseError = norm(exactInverseEigenvalues(:) - approximatedInverseEigenvalues(:)) / norm(exactInverseEigenvalues(:));

minimumPreconditionedEigenvalue = min(preconditionedEigenvalues(:));
maximumPreconditionedEigenvalue = max(preconditionedEigenvalues(:));
spectralDelta = max(abs(1 - preconditionedEigenvalues(:)));

isPositiveDefinite = minimumPreconditionedEigenvalue > 0;

if isPositiveDefinite
    conditionNumber = maximumPreconditionedEigenvalue / minimumPreconditionedEigenvalue;
else
    conditionNumber = NaN;
end

retainedRanks = size(inverseEigenvalueTensor.core);
storageEntries = numel(inverseEigenvalueTensor.core);

for mode = 1:3
    storageEntries = storageEntries + ...
        numel(inverseEigenvalueTensor.u{mode});
end

diagnosticTime = toc(diagnosticTimer);
setupTime = toc(setupTimer);


%% 5. Return the reusable preconditioner and diagnostics

preconditioner.N = N;
preconditioner.number_of_modes = 3;
preconditioner.transform = 'DST-I';
preconditioner.one_dimensional_eigenvalues = ...
    oneDimensionalEigenvalues;
preconditioner.inverse_eigenvalue_tensor = ...
    inverseEigenvalueTensor;
preconditioner.inverse_eigenvalue_ranks = retainedRanks;

info.N = N;
info.requested_ranks = maximumRanks;
info.retained_ranks = retainedRanks;
info.inverse_eigenvalue_relative_frobenius_error = inverseError;
info.minimum_preconditioned_eigenvalue = ...
    minimumPreconditionedEigenvalue;
info.maximum_preconditioned_eigenvalue = ...
    maximumPreconditionedEigenvalue;
info.spectral_delta = spectralDelta;
info.is_positive_definite = isPositiveDefinite;
info.condition_number = conditionNumber;
info.storage_entries = storageEntries;
info.storage_mib = 8 * storageEntries / 2^20;
info.full_spectral_storage_mib = ...
    8 * numel(exactInverseEigenvalues) / 2^20;
info.compression_time_sec = compressionTime;
info.diagnostic_time_sec = diagnosticTime;
info.setup_time_sec = setupTime;

end
