function [G, Ucell, Xhat, elapsed] = r_sthosvd(X, ranks, p, order)

    tic;

    d = ndims(X);
    Ucell = cell(1,d);
    G = X;

    for k = 1:d
        n = order(k);

        Gn = double(tenmat(G, n));
        rn = ranks(n);

        if rn + p > min(size(Gn))
            error("Mode %d: r+p exceeds min(size(current unfolding)).", n);
        end

        [Urand,~,~] = rsvd(Gn, rn, p);
        Ucell{n} = Urand(:,1:rn);

        % Compress immediately in mode n
        G = ttm(G, Ucell{n}', n);
    end

    Xhat = G;
    for n = 1:d
        Xhat = ttm(Xhat, Ucell{n}, n);
    end

    elapsed = toc;
end