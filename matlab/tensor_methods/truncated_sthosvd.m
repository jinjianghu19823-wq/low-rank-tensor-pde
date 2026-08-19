function [G, Ucell, Xhat, elapsed] = truncated_sthosvd(X, ranks, order)

    tic;

    d = ndims(X);
    Ucell = cell(1,d);
    G = X;

    for k = 1:d
        n = order(k);

        Gn = double(tenmat(G, n));
        [U,~,~] = svd(Gn, "econ");

        Ucell{n} = U(:,1:ranks(n));

        % Compress immediately in mode n
        G = ttm(G, Ucell{n}', n);
    end

    Xhat = G;
    for n = 1:d
        Xhat = ttm(Xhat, Ucell{n}, n);
    end

    elapsed = toc;
end