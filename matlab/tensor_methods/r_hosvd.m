function [G, Ucell, Xhat, elapsed] = r_hosvd(X, ranks, p)

    tic;

    d = ndims(X);
    Ucell = cell(1,d);

    for n = 1:d
        Xn = double(tenmat(X, n));
        rn = ranks(n);

        if rn + p > min(size(Xn))
            error("Mode %d: r+p exceeds min(size(unfolding)).", n);
        end

        [Urand,~,~] = rsvd(Xn, rn, p);
        Ucell{n} = Urand(:,1:rn);
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