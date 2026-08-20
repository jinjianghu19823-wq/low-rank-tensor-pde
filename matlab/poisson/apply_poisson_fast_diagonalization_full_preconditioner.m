function y = ...
    apply_poisson_fast_diagonalization_full_preconditioner( ...
        x, inverse_eigenvalue_multiplier)
%APPLY_POISSON_FAST_DIAGONALIZATION_FULL_PRECONDITIONER Apply M_r.

if ndims(inverse_eigenvalue_multiplier) ~= 3
    error('inverse_eigenvalue_multiplier must be a three-dimensional array.');
end

multiplier_size = size(inverse_eigenvalue_multiplier);

if any(multiplier_size ~= multiplier_size(1))
    error('The inverse-eigenvalue multiplier must have equal mode sizes.');
end

N = multiplier_size(1);
x = x(:);

if length(x) ~= N^3
    error('x must contain exactly N^3 entries.');
end

input_tensor = reshape(x, [N, N, N]);
spectral_tensor = apply_transform_in_all_modes(input_tensor, @dst);
spectral_tensor = spectral_tensor .* inverse_eigenvalue_multiplier;
output_tensor = apply_transform_in_all_modes(spectral_tensor, @idst);
y = output_tensor(:);

end

function Y = apply_transform_in_all_modes(X, transform)
%APPLY_TRANSFORM_IN_ALL_MODES Apply a column transform to every mode.

Y = X;
d = 3;

for mode = 1:d

    permutation = [mode, setdiff(1:d, mode, 'stable')];
    inverse_permutation = zeros(1, d);
    inverse_permutation(permutation) = 1:d;

    Y_perm = permute(Y, permutation);
    transformed_matrix = transform( ...
        reshape(Y_perm, size(Y, mode), []));
    Y_perm = reshape(transformed_matrix, size(Y_perm));
    Y = permute(Y_perm, inverse_permutation);

end

end
