function [X, s, Qcell] = controlled_tucker_tensor_generator(N, scale, seed)
% Construct a controlled third-order tensor
% X = S x_1 Q1 x_2 Q2 x_3 Q3,
% Output:
% X: Tensor Toolbox tensor object, size N x N x N
% s: prescribed mode-unfolding singular values
% Qcell: random orthogonal factor matrices

    rng(seed);

    i = (1:N)';
    s = (1 + (i - 1) / scale).^(-7);
    S_array = zeros(N, N, N);
    for i = 1:N
        S_array(i,i,i) = s(i);
    end

    [Q1,~] = qr(randn(N,N), 0);
    [Q2,~] = qr(randn(N,N), 0);
    [Q3,~] = qr(randn(N,N), 0);

    X = tensor(S_array);
    X = ttm(X, Q1, 1);
    X = ttm(X, Q2, 2);
    X = ttm(X, Q3, 3);

    Qcell = {Q1, Q2, Q3};
end
