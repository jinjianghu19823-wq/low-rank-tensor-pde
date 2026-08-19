function outputVector = ...
    apply_poisson_fast_diagonalization_full_preconditioner_4d( ...
    inputVector, inverseEigenvalueMultiplier)
%APPLY_POISSON_FAST_DIAGONALIZATION_FULL_PRECONDITIONER_4D Apply M_r.

if ndims(inverseEigenvalueMultiplier) ~= 4
    error('inverseEigenvalueMultiplier must be a four dimensional array.');
end
multiplierSize = size(inverseEigenvalueMultiplier);
if any(multiplierSize ~= multiplierSize(1))
    error('The multiplier must have equal mode sizes.');
end
N = multiplierSize(1);
inputVector = inputVector(:);
if numel(inputVector) ~= N^4
    error('inputVector must contain exactly N^4 entries.');
end

output = reshape(inputVector, [N, N, N, N]);
for mode = 1:4
    permutation = [mode, setdiff(1:4, mode, 'stable')];
    permuted = permute(output, permutation);
    transformed = dst(reshape(permuted, N, []));
    output = ipermute(reshape(transformed, size(permuted)), permutation);
end
output = output .* inverseEigenvalueMultiplier;
for mode = 1:4
    permutation = [mode, setdiff(1:4, mode, 'stable')];
    permuted = permute(output, permutation);
    transformed = idst(reshape(permuted, N, []));
    output = ipermute(reshape(transformed, size(permuted)), permutation);
end
outputVector = output(:);

end
