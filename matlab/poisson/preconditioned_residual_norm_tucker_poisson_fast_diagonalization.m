function residualNorm = ...
    preconditioned_residual_norm_tucker_poisson_fast_diagonalization( ...
        U, F, A1, preconditioner, diagnosticTolerance)
%PRECONDITIONED_RESIDUAL_NORM_TUCKER_POISSON_FAST_DIAGONALIZATION
% Independently recompute ||M_r(F-L(U))||_F.
%
% The diagnostic uses one fixed, tight tolerance. It is independent of the
% Hessenberg least-squares residual and of the step-dependent tolerances
% used inside relaxed Tucker-GMRES.
%
% Thesis/experiment notation (Sections 5.5 and 6.2):
%   U, F                        <->  \mathcal X_j, \mathcal B_0
%   A1                          <->  A_1
%   originalResidual           <->  \mathcal B_0-\mathcal A_0(\mathcal X_j)
%   preconditionedResidual     <->  \mathcal P_r(originalResidual)
%   residualNorm               <->  ||\mathcal P_r(\mathcal B_0-
%                                      \mathcal A_0(\mathcal X_j))||_F
%   diagnosticTolerance        <->  fixed diagnostic-only tolerance

if diagnosticTolerance <= 0 || diagnosticTolerance >= 1
    error('diagnosticTolerance must be between 0 and 1.');
end

fullRanks = size(F);

[AU, ~] = poisson_action_tucker( ...
    U, A1, diagnosticTolerance, fullRanks);

originalResidual = tucker_axpby_exact(F, 1, AU, -1);

[preconditionedResidual, ~] = ...
    apply_poisson_fast_diagonalization_tucker_preconditioner( ...
        originalResidual, preconditioner, diagnosticTolerance, ...
        fullRanks);

residualNorm = norm(preconditionedResidual);

end
