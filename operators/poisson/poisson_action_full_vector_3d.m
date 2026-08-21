function outputVector = poisson_action_full_vector_3d( ...
    inputVector, A1, N)
%POISSON_ACTION_FULL_VECTOR_3D Apply the 3D Poisson operator matrix-free.
%
% The full Kronecker-sum matrix has N^3 rows and N^3 columns and is never
% formed. This routine reshapes the vector as an N-by-N-by-N array and
% applies the standard seven-point finite-difference stencil.
%
% Inputs:
%   inputVector
%       Vectorization of an N-by-N-by-N array.
%
%   A1
%       N-by-N matrix h^(-2)*tridiag(-1,2,-1).
%
%   N
%       Number of interior grid points in each coordinate direction.
%
% Output:
%   outputVector
%       Vectorization of the three-dimensional Poisson action.
%
% Thesis/experiment notation (Section 6.2 full baseline):
%   inputVector, outputVector  <->  x, A_0 x
%   A1                          <->  A_1
%   N                           <->  interior points per coordinate
%   inputTensor                 <->  tensor reshaping of x


%% 1. Check the dimensions

inputVector = inputVector(:);

if length(inputVector) ~= N^3
    error('inputVector must contain exactly N^3 entries.');
end

if N < 2 || ~isequal(size(A1), [N, N])
    error('N must be at least 2 and A1 must have size N-by-N.');
end


%% 2. Read the stencil coefficients

diagonalCoefficient = full(A1(1,1));
offDiagonalCoefficient = full(A1(1,2));


%% 3. Apply the seven-point stencil

inputTensor = reshape(inputVector, [N, N, N]);

% Each of the three one-dimensional operators contributes its diagonal.
outputTensor = 3 * diagonalCoefficient * inputTensor;

% Neighbours in mode 1.
outputTensor(1:end-1,:,:) = outputTensor(1:end-1,:,:) + ...
    offDiagonalCoefficient * inputTensor(2:end,:,:);
outputTensor(2:end,:,:) = outputTensor(2:end,:,:) + ...
    offDiagonalCoefficient * inputTensor(1:end-1,:,:);

% Neighbours in mode 2.
outputTensor(:,1:end-1,:) = outputTensor(:,1:end-1,:) + ...
    offDiagonalCoefficient * inputTensor(:,2:end,:);
outputTensor(:,2:end,:) = outputTensor(:,2:end,:) + ...
    offDiagonalCoefficient * inputTensor(:,1:end-1,:);

% Neighbours in mode 3.
outputTensor(:,:,1:end-1) = outputTensor(:,:,1:end-1) + ...
    offDiagonalCoefficient * inputTensor(:,:,2:end);
outputTensor(:,:,2:end) = outputTensor(:,:,2:end) + ...
    offDiagonalCoefficient * inputTensor(:,:,1:end-1);


%% 4. Return to vector form

outputVector = outputTensor(:);

end
