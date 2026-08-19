function residualNorm = preconditioned_residual_norm_tucker_poisson( ...
    U, F, A1, preconditioner, diagnosticTolerance)
%PRECONDITIONED_RESIDUAL_NORM_TUCKER_POISSON Recompute ||M_q(F-AU)||_F.
%
% This diagnostic is independent of the Hessenberg least-squares residual.
% It uses a fixed, tight recompression tolerance and no additional Tucker
% rank cap. The result is an absolute preconditioned residual norm; a solver
% cycle may divide it by its own starting norm beta.

if diagnosticTolerance <= 0 || diagnosticTolerance >= 1
    error('diagnosticTolerance must be between 0 and 1.');
end

fullRanks = size(F);

[AU, ~] = poisson_action_tucker( ...
    U, A1, diagnosticTolerance, fullRanks);

originalResidual = tucker_axpby_exact(F, 1, AU, -1);

[preconditionedResidual, ~] = ...
    apply_poisson_exponential_sum_preconditioner( ...
        originalResidual, preconditioner, diagnosticTolerance, ...
        fullRanks);

residualNorm = norm(preconditionedResidual);

end
