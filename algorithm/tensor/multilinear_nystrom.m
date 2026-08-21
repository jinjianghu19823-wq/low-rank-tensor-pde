function [G, Lcell, Xhat, info, elapsed] = multilinear_nystrom(X, ranks, l_vec)
%   This implements the QR-based MLN construction:
%
%       W_n = Y_n' * A_(n) * X_n = Z_n R_n,
%       L_n = A_(n) * X_n / R_n,
%
%   and then
%
%       G = X x_1 Y_1' ... x_d Y_d' x_1 Z_1' ... x_d Z_d',
%       Xhat = G x_1 L_1 ... x_d L_d.
%
% Inputs:
%   X: Tensor Toolbox tensor object
%   ranks: target multilinear rank, e.g. [r r r]
%   l_vec: oversampling vector, e.g. [ceil(r/2), ceil(r/2), ceil(r/2)]
%
% Outputs:
%   G: core tensor
%   Lcell: Tucker factor matrices
%   Xhat: reconstructed Tucker approximation
%   info: diagnostic structure
%   elapsed: runtime

    tic;

    d = ndims(X);
    Lcell = cell(1, d);
    Ycell = cell(1, d);
    Zcell = cell(1, d);

    info.condW = zeros(1, d);
    info.l = l_vec;

    for n = 1:d
        A_n = double(tenmat(X, n));

        rn = ranks(n);
        l = l_vec(n);

        [In, Jn] = size(A_n);

        Xn = randn(Jn, rn);
        Yn = randn(In, rn + l);

        AX = A_n * Xn;
        YtA = Yn' * A_n;
        W = YtA * Xn;

        [Z, R] = qr(W, 0);

        Lcell{n} = AX / R;
        Ycell{n} = Yn;
        Zcell{n} = Z;

        info.condW(n) = local_cond(W);
    end

    G = X;

    for n = 1:d
        G = ttm(G, Ycell{n}', n);
    end

    for n = 1:d
        G = ttm(G, Zcell{n}', n);
    end

    Xhat = G;

    for n = 1:d
        Xhat = ttm(Xhat, Lcell{n}, n);
    end

    elapsed = toc;
end

function c = local_cond(W)
    s = svd(W, "econ");

    if isempty(s) || s(end) == 0
        c = Inf;
    else
        c = s(1) / s(end);
    end
end