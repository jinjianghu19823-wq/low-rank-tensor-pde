function Z = tucker_hadamard_exact(X, Y)
%TUCKER_HADAMARD_EXACT Represent the exact Hadamard product in Tucker form.

if ~isa(X, 'ttensor') || ~isa(Y, 'ttensor')
    error('X and Y must be Tensor Toolbox ttensor objects.');
end

if ndims(X) ~= 3 || ndims(Y) ~= 3
    error('This educational implementation currently supports order 3.');
end

if any(size(X) ~= size(Y))
    error('X and Y must have the same tensor size.');
end

ranks_x = size(X.core);
ranks_y = size(Y.core);

core_x = double(X.core);
core_y = double(Y.core);

expanded_core_x = reshape( ...
    core_x, [ranks_x(1), 1, ranks_x(2), 1, ranks_x(3), 1]);
expanded_core_y = reshape( ...
    core_y, [1, ranks_y(1), 1, ranks_y(2), 1, ranks_y(3)]);

paired_core_values = expanded_core_x .* expanded_core_y;
paired_core = tensor(paired_core_values(:), ranks_x .* ranks_y);

paired_factors = cell(3, 1);

for mode = 1:3

    factor_x = X.u{mode};
    factor_y = Y.u{mode};
    paired_factor = zeros( ...
        size(factor_x, 1), ranks_x(mode) * ranks_y(mode));

    col_idx = 0;

    for column_y = 1:ranks_y(mode)
        for column_x = 1:ranks_x(mode)

            col_idx = col_idx + 1;
            paired_factor(:, col_idx) = ...
                factor_x(:, column_x) .* factor_y(:, column_y);

        end
    end

    paired_factors{mode} = paired_factor;

end

Z = ttensor(paired_core, paired_factors);

end
