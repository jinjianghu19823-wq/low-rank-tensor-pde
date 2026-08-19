function outputVector = poisson_action_full_vector_4d( ...
    inputVector, A1, N)
%POISSON_ACTION_FULL_VECTOR_4D Apply the 4D Poisson operator matrix-free.
%
% The full Kronecker-sum matrix would have N^4 rows and N^4 columns. This
% function never constructs that matrix. Instead, it reshapes the vector as
% an N-by-N-by-N-by-N array and applies A1 along one tensor direction at a
% time.
%
% Inputs:
%   inputVector
%       Vectorization of an N-by-N-by-N-by-N array.
%
%   A1
%       N-by-N one-dimensional finite-difference matrix.
%
%   N
%       Number of interior grid points in every coordinate direction.
%
% Output:
%   outputVector
%       Vectorization of the four-dimensional Poisson action.


%% 1. Check the dimensions

inputVector = inputVector(:);

if length(inputVector) ~= N^4
    error('inputVector must contain exactly N^4 entries.');
end

if ~isequal(size(A1), [N, N])
    error('A1 must have size N-by-N.');
end


%% 2. Reshape the vector as a four-dimensional array

inputTensor = reshape(inputVector, [N, N, N, N]);
outputTensor = zeros(N, N, N, N);


%% 3. Apply A1 along each of the four modes

for mode = 1:4

    % Move the active mode to the first array dimension.
    otherModes = 1:4;
    otherModes(mode) = [];
    permutation = [mode, otherModes];

    permutedInput = permute(inputTensor, permutation);

    % Every column is now one mode fibre, so one matrix multiplication
    % applies A1 to all fibres in the active mode.
    inputMatrix = reshape(permutedInput, N, []);
    outputMatrix = A1 * inputMatrix;

    permutedOutput = reshape(outputMatrix, size(permutedInput));
    oneModeContribution = ipermute(permutedOutput, permutation);

    outputTensor = outputTensor + oneModeContribution;

end


%% 4. Return to vector form

outputVector = outputTensor(:);

end
