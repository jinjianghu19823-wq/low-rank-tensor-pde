function Z = tucker_axpby_exact(X, alpha, Y, beta)
%TUCKER_AXPBY_EXACT Add two Tucker tensors without truncation.
%
% This function calculates
%
%     Z = alpha * X + beta * Y.
%
% It is similar to tucker_axpby_round, but it does not discard any small
% singular directions. It is used when calculating a reliable true
% residual. The intermediate Tucker ranks can therefore be larger.
%
% Inputs:
%   X, Y
%       Tensor Toolbox ttensor objects with the same dimensions.
%
%   alpha, beta
%       Scalar coefficients.
%
% Output:
%   Z
%       An exact Tucker representation, up to floating-point roundoff.
%
% Thesis notation (Section 5.5):
%   X, Y, Z                    <->  \mathcal X, \mathcal Y, \mathcal W
%   alpha                      <->  \alpha
%   beta                       <->  \gamma (the AXPBY coefficient)
%   combinedCore               <->  \widehat{\mathcal G}
%   combinedFactors{mode}      <->  \widehat U^(n)
%   orthogonalFactors{mode}    <->  Q^(n)
%   orthogonalCore             <->  \mathcal G_Q
% Here beta is an AXPBY coefficient; it is not the GMRES quantity
% \beta=||\widetilde r_0||.


%% 1. Check the inputs

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must both be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(X), size(Y))
    error('X and Y must have the same tensor dimensions.');
end

if ~isscalar(alpha) || ~isscalar(beta)
    error('alpha and beta must be scalar numbers.');
end


%% 2. Read the ranks and prepare the larger block core

numberOfModes = ndims(X);

rankX = size(X.core);
rankY = size(Y.core);
combinedCoreSize = rankX + rankY;

combinedCoreValues = zeros(combinedCoreSize);

indexX = cell(numberOfModes, 1);
indexY = cell(numberOfModes, 1);

for mode = 1:numberOfModes

    indexX{mode} = 1:rankX(mode);
    indexY{mode} = rankX(mode) + (1:rankY(mode));

end

combinedCoreValues(indexX{:}) = alpha * double(X.core);
combinedCoreValues(indexY{:}) = beta * double(Y.core);

combinedCore = tensor(combinedCoreValues);


%% 3. Join the two sets of factor matrices

combinedFactors = cell(numberOfModes, 1);

for mode = 1:numberOfModes
    combinedFactors{mode} = [X.u{mode}, Y.u{mode}];
end


%% 4. Orthonormalise the joined factors

% QR avoids reconstructing the full N-by-N-by-... tensor. The R matrices
% are absorbed into the core, so the represented tensor does not change.
orthogonalFactors = cell(numberOfModes, 1);
orthogonalCore = combinedCore;

for mode = 1:numberOfModes

    [Q, R] = qr(combinedFactors{mode}, 0);

    orthogonalFactors{mode} = Q;
    orthogonalCore = ttm(orthogonalCore, R, mode);

end


%% 5. Return the exact sum

Z = ttensor(orthogonalCore, orthogonalFactors);

end
