function preconditioner = build_poisson_exponential_sum_preconditioner( ...
    A1, numberOfModes, quadratureHalfWidth)
%BUILD_POISSON_EXPONENTIAL_SUM_PRECONDITIONER Build the separable M_q data.
%
% For the d-dimensional Kronecker-sum Poisson matrix A_d, the approximate
% inverse is
%
%     M_q = sum_{k=-q}^q omega_k
%           exp(-tau_k A1) kron ... kron exp(-tau_k A1),
%
% where
%
%     zeta_q = pi / sqrt(q),
%     tau_k  = exp(k*zeta_q),
%     omega_k = zeta_q*tau_k.
%
% The returned structure stores the eigendecomposition of A1. The matrix
% exponentials are therefore applied to Tucker factor matrices without
% forming d-dimensional matrices.


%% 1. Check the inputs

if size(A1, 1) ~= size(A1, 2)
    error('A1 must be a square matrix.');
end

if norm(A1 - A1.', 'fro') > 1e-12 * max(1, norm(A1, 'fro'))
    error('A1 must be symmetric.');
end

if ~isscalar(numberOfModes) || numberOfModes < 1 || ...
        numberOfModes ~= floor(numberOfModes)
    error('numberOfModes must be a positive integer.');
end

if ~isscalar(quadratureHalfWidth) || quadratureHalfWidth < 1 || ...
        quadratureHalfWidth ~= floor(quadratureHalfWidth)
    error('quadratureHalfWidth q must be a positive integer.');
end


%% 2. Diagonalise the one-dimensional Poisson matrix

[eigenvectors, eigenvalueMatrix] = eig(full(A1), 'vector');
[oneDimensionalEigenvalues, permutation] = sort(eigenvalueMatrix, 'ascend');
eigenvectors = eigenvectors(:, permutation);

if any(oneDimensionalEigenvalues <= 0)
    error('A1 must be positive definite.');
end


%% 3. Construct the sinc-quadrature nodes and weights

q = quadratureHalfWidth;
zeta = pi / sqrt(q);
indices = (-q:q).';
tau = exp(indices * zeta);
omega = zeta * tau;


%% 4. Return all reusable data

preconditioner.A1 = A1;
preconditioner.N = size(A1, 1);
preconditioner.number_of_modes = numberOfModes;
preconditioner.q = q;
preconditioner.number_of_terms = 2 * q + 1;
preconditioner.zeta = zeta;
preconditioner.indices = indices;
preconditioner.tau = tau;
preconditioner.omega = omega;
preconditioner.eigenvectors = eigenvectors;
preconditioner.one_dimensional_eigenvalues = ...
    oneDimensionalEigenvalues;

end
