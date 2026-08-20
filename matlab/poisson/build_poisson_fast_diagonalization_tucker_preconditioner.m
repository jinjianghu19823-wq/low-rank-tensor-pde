function [preconditioner, info] = build_poisson_fast_diagonalization_tucker_preconditioner(A1, inverse_eigenvalue_rank)
%BUILD_POISSON_FAST_DIAGONALIZATION_TUCKER_PRECONDITIONER Build M_r.

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

N = size(A1, 1);
max_ranks = normalise_tucker_rank_cap(inverse_eigenvalue_rank, [N, N, N]);

diagonal_coefficient = full(A1(1, 1));

if N > 1
    offdiag = full(A1(1, 2));
else
    offdiag = 0;
end

reference_a1 = spdiags([offdiag * ones(N, 1), diagonal_coefficient * ones(N, 1), offdiag * ones(N, 1)], -1:1, N, N);

matrix_mismatch = norm(A1 - reference_a1, 'fro') / max(norm(A1, 'fro'), eps);

if matrix_mismatch > 1e-12
    error(['A1 must be the symmetric constant-coefficient Dirichlet ', ...
           'tridiagonal matrix diagonalised by DST-I.']);
end

if offdiag >= 0
    error('The Dirichlet Poisson off-diagonal coefficient must be negative.');
end

setup_timer = tic;

indices = (1:N).';
one_dimensional_eigenvalues = diagonal_coefficient + 2 * offdiag * cos(indices * pi / (N + 1));

if any(one_dimensional_eigenvalues <= 0)
    error('A1 must be positive definite.');
end

lambda_sum = one_dimensional_eigenvalues + reshape(one_dimensional_eigenvalues, 1, N) + reshape(one_dimensional_eigenvalues, 1, 1, N);
exact_inverse_eigenvalues = 1 ./ lambda_sum;

compression_timer = tic;

inverse_eigenvalue_tensor = hosvd(tensor(exact_inverse_eigenvalues), 1e-14, 'ranks', max_ranks, 'sequential', false, 'verbosity', 0);

compression_time = toc(compression_timer);

diagnostic_timer = tic;

approximated_inverse_eigenvalues = double(full(inverse_eigenvalue_tensor));
preconditioned_eigenvalues = lambda_sum .* approximated_inverse_eigenvalues;

inverse_error = norm(exact_inverse_eigenvalues(:) - approximated_inverse_eigenvalues(:)) / norm(exact_inverse_eigenvalues(:));

minimum_preconditioned_eigenvalue = min(preconditioned_eigenvalues(:));
maximum_preconditioned_eigenvalue = max(preconditioned_eigenvalues(:));
spectral_delta = max(abs(1 - preconditioned_eigenvalues(:)));

is_positive_definite = minimum_preconditioned_eigenvalue > 0;

if is_positive_definite
    condition_number = maximum_preconditioned_eigenvalue / minimum_preconditioned_eigenvalue;
else
    condition_number = NaN;
end

retained_ranks = size(inverse_eigenvalue_tensor.core);
storage_entries = numel(inverse_eigenvalue_tensor.core);

for mode = 1:3
    storage_entries = storage_entries + ...
        numel(inverse_eigenvalue_tensor.u{mode});
end

diagnostic_time = toc(diagnostic_timer);
setup_time = toc(setup_timer);

preconditioner.N = N;
preconditioner.number_of_modes = 3;
preconditioner.transform = 'DST-I';
preconditioner.one_dimensional_eigenvalues = ...
    one_dimensional_eigenvalues;
preconditioner.inverse_eigenvalue_tensor = ...
    inverse_eigenvalue_tensor;
preconditioner.inverse_eigenvalue_ranks = retained_ranks;

info.N = N;
info.requested_ranks = max_ranks;
info.retained_ranks = retained_ranks;
info.inverse_eigenvalue_relative_frobenius_error = inverse_error;
info.minimum_preconditioned_eigenvalue = ...
    minimum_preconditioned_eigenvalue;
info.maximum_preconditioned_eigenvalue = ...
    maximum_preconditioned_eigenvalue;
info.spectral_delta = spectral_delta;
info.is_positive_definite = is_positive_definite;
info.condition_number = condition_number;
info.storage_entries = storage_entries;
info.storage_mib = 8 * storage_entries / 2^20;
info.full_spectral_storage_mib = ...
    8 * numel(exact_inverse_eigenvalues) / 2^20;
info.compression_time_sec = compression_time;
info.diagnostic_time_sec = diagnostic_time;
info.setup_time_sec = setup_time;

end
