function outputVector = ...
    apply_poisson_fast_diagonalization_full_preconditioner( ...
        inputVector, inverseEigenvalueMultiplier)
%APPLY_POISSON_FAST_DIAGONALIZATION_FULL_PRECONDITIONER Apply M_r.
%
% The input is a full vectorization of an N-by-N-by-N tensor. The
% preconditioner performs
%
%   output = IDST( inverseEigenvalueMultiplier .* DST(input) )
%
% with a DST-I in every mode. The multiplier can be the exact inverse
% eigenvalue tensor or a full reconstruction of its Tucker approximation.
%
% Thesis/experiment notation (Section 6.2 full baseline):
%   inputVector, outputVector       <->  y, M_r y
%   inverseEigenvalueMultiplier     <->  full reconstruction of D_r
%   spectralTensor                  <->  DST(y), then D_r .* DST(y)
% Thus outputVector=IDST(D_r .* DST(y)).


%% 1. Check the dimensions

if ndims(inverseEigenvalueMultiplier) ~= 3
    error('inverseEigenvalueMultiplier must be a three-dimensional array.');
end

multiplierSize = size(inverseEigenvalueMultiplier);

if any(multiplierSize ~= multiplierSize(1))
    error('The inverse-eigenvalue multiplier must have equal mode sizes.');
end

N = multiplierSize(1);
inputVector = inputVector(:);

if length(inputVector) ~= N^3
    error('inputVector must contain exactly N^3 entries.');
end


%% 2. Transform, multiply, and transform back

inputTensor = reshape(inputVector, [N, N, N]);
spectralTensor = apply_transform_in_all_modes(inputTensor, @dst);
spectralTensor = spectralTensor .* inverseEigenvalueMultiplier;
outputTensor = apply_transform_in_all_modes(spectralTensor, @idst);
outputVector = outputTensor(:);

end


function Y = apply_transform_in_all_modes(X, transformFunction)
%APPLY_TRANSFORM_IN_ALL_MODES Apply a column transform to every mode.

Y = X;
numberOfModes = 3;

for mode = 1:numberOfModes

    permutation = [mode, setdiff(1:numberOfModes, mode, 'stable')];
    inversePermutation = zeros(1, numberOfModes);
    inversePermutation(permutation) = 1:numberOfModes;

    permutedY = permute(Y, permutation);
    transformedMatrix = transformFunction( ...
        reshape(permutedY, size(Y, mode), []));
    permutedY = reshape(transformedMatrix, size(permutedY));
    Y = permute(permutedY, inversePermutation);

end

end
