function [A, s] = controlled_svd_matrix_generator(m, n, spectrum_type, param, seed)
% Inputs:
% m,n: matrix size m-by-n
% spectrum_type "exp", "power", or "scaled_power"
% param: alpha for exp, beta for power, scale for scaled_power
% seed: random seed
%
% Outputs:
% A: m-by-n test matrix
% s: prescribed singular values
    rng(seed);

    assert(m >= n, 'assumes m >= n');

    [U, ~] = qr(randn(m, n), "econ");
    [V, ~] = qr(randn(n, n), "econ");

    j = (1:n)';

    switch spectrum_type
        case "exp"
            alpha = param;
            s = exp(-(j-1) / alpha);

        case "power"
            beta = param;
            s = j.^(-beta);
            s = s / s(1);

        case "scaled_power"
            scale = param;
            s = (1 + (j - 1) / scale).^(-7);

        otherwise
            error("Unknown spectrum type.");
    end

    A = (U .* s') * V';
end
