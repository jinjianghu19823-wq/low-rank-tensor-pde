function [G, Ucell, Xhat, elapsed] = truncated_hosvd(X, ranks)

    tic;

    d = ndims(X);
    Ucell = cell(1, d);

    for n = 1:d
        Xn = double(tenmat(X, n));
        [U,~,~] = svd(Xn, "econ");
        Ucell{n} = U(:, 1:ranks(n));
    end

    G = X;
    for n = 1:d
        G = ttm(G, Ucell{n}', n);
    end

    Xhat = G;
    for n = 1:d
        Xhat = ttm(Xhat, Ucell{n}, n);
    end

    elapsed = toc;
end