function Z = tucker_hadamard_exact(X, Y)
%TUCKER_HADAMARD_EXACT Represent the exact Hadamard product in Tucker form.
%
% For three-dimensional Tucker tensors
%
%   X = F x_1 A1 x_2 A2 x_3 A3,
%   Y = G x_1 B1 x_2 B2 x_3 B3,
%
% the entrywise product is represented by the paired factor columns
%
%   A_n(:,a) .* B_n(:,b)
%
% and the paired core entries F(a1,a2,a3)*G(b1,b2,b3). The resulting
% multilinear ranks are the products of the input representation ranks.
% No truncation is performed here.
%
% Thesis/experiment notation (Section 6.2):
%   X, Y, Z                    <->  Tucker tensors \mathcal X, \mathcal Y,
%                                   and their exact Hadamard product
%   pairedCore                 <->  product core before recompression
%   pairedFactors{mode}        <->  paired mode-n factor columns
% In the fast-diagonalization preconditioner, X is the compressed inverse-
% eigenvalue tensor D_r and Y is the sine-transformed input.


%% 1. Check the inputs

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must be Tensor Toolbox ttensor objects.');
end

if ndims(X) ~= 3 || ndims(Y) ~= 3
    error('This educational implementation currently supports order 3.');
end

if any(size(X) ~= size(Y))
    error('X and Y must have the same tensor size.');
end


%% 2. Form the paired core

ranksX = size(X.core);
ranksY = size(Y.core);

coreX = double(X.core);
coreY = double(Y.core);

expandedCoreX = reshape( ...
    coreX, [ranksX(1), 1, ranksX(2), 1, ranksX(3), 1]);
expandedCoreY = reshape( ...
    coreY, [1, ranksY(1), 1, ranksY(2), 1, ranksY(3)]);

pairedCoreValues = expandedCoreX .* expandedCoreY;
% Pass the core dimensions explicitly. MATLAB numeric arrays drop trailing
% singleton dimensions, so tensor(reshape(...)) would turn a 1 by 1 by 1
% core into an order two tensor. The explicit size keeps the Tucker order.
pairedCore = tensor(pairedCoreValues(:), ranksX .* ranksY);


%% 3. Form the paired factor columns

pairedFactors = cell(3, 1);

for mode = 1:3

    factorX = X.u{mode};
    factorY = Y.u{mode};
    pairedFactor = zeros( ...
        size(factorX, 1), ranksX(mode) * ranksY(mode));

    columnIndex = 0;

    for columnY = 1:ranksY(mode)
        for columnX = 1:ranksX(mode)

            columnIndex = columnIndex + 1;
            pairedFactor(:, columnIndex) = ...
                factorX(:, columnX) .* factorY(:, columnY);

        end
    end

    pairedFactors{mode} = pairedFactor;

end


%% 4. Return the exact representation

Z = ttensor(pairedCore, pairedFactors);

end
