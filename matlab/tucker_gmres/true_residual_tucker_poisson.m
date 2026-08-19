function relativeResidual = true_residual_tucker_poisson(U, F, A1)
%TRUE_RESIDUAL_TUCKER_POISSON Calculate the true relative residual.
%
% This function calculates
%
%                 ||F - L(U)||_F
%     residual = -----------------.
%                      ||F||_F
%
% The residual is evaluated without truncating F - L(U). This is why it is
% called the true residual rather than the internal Tucker-GMRES residual.
%
% The full tensors are not formed. The residual terms are added in Tucker
% form with tucker_axpby_exact. This is more stable near convergence than
% expanding the squared norm into many inner products, where large terms
% can cancel and cause a false residual floor near sqrt(eps).
%
% Inputs:
%   U
%       Approximate solution stored as a ttensor.
%
%   F
%       Right-hand side stored as a ttensor.
%
%   A1
%       One-dimensional finite-difference matrix.
%
% Output:
%   relativeResidual
%       True relative Frobenius residual.
%
% Thesis notation (Sections 5.5 and 6.2):
%   U, F                        <->  \mathcal X_j, \mathcal B_0
%   A1                          <->  A_1
%   oneModeContribution        <->  \mathcal X_j \times_n A_1
%   residualTensor             <->  \mathcal B_0-\mathcal A_0(\mathcal X_j)
%   relativeResidual           <->  ||\mathcal B_0-\mathcal A_0(\mathcal X_j)||_F
%                                      / ||\mathcal B_0||_F


%% 1. Check the inputs

if ~isa(U, 'ttensor') || ~isa(F, 'ttensor')
    error('U and F must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(U), size(F))
    error('U and F must have the same tensor dimensions.');
end

normF = norm(F);

if normF == 0
    error('The right-hand side F must be nonzero.');
end


%% 2. Start with the right-hand side

numberOfModes = ndims(U);

residualTensor = F;


%% 3. Subtract every mode contribution of the Poisson action

for mode = 1:numberOfModes

    oneModeContribution = ttm(U, A1, mode);

    residualTensor = tucker_axpby_exact(residualTensor, 1, oneModeContribution, -1);

end


%% 4. Return the true relative residual

relativeResidual = norm(residualTensor) / normF;

end
