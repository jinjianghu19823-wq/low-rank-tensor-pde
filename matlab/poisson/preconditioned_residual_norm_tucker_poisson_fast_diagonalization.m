function residual_norm = ...
    preconditioned_residual_norm_tucker_poisson_fast_diagonalization( ...
        U, F, A1, preconditioner, diagnostic_tolerance)
%PRECONDITIONED_RESIDUAL_NORM_TUCKER_POISSON_FAST_DIAGONALIZATION Independently recompute ||M_r(F-L(U))||_F.

if diagnostic_tolerance <= 0 || diagnostic_tolerance >= 1
    error('diagnostic_tolerance must be between 0 and 1.');
end

full_ranks = size(F);

[AU, ~] = poisson_action_tucker( ...
    U, A1, diagnostic_tolerance, full_ranks);

original_residual = tucker_axpby_exact(F, 1, AU, -1);

[r_precond, ~] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner( ...
        original_residual, preconditioner, diagnostic_tolerance, ...
        full_ranks);

residual_norm = norm(r_precond);

end
