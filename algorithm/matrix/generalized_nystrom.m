function [L, M, info, elapsed] = generalized_nystrom(A, r, l, stabilized)
% Ahat = L*M approximates A
% Ahat = A*X * (Y'*A*X)^\dagger * Y'*A
% Stable QR implementation: W = Y'*A*X = Z*R. Ahat = (A*X/R) * (Z'*(Y'*A)).
%
% Inputs:
% A: matrix to approximate
% r: target rank
% l: oversampling parameter for Y
% stablized false for GN, true for SGN
%
% Outputs:
% L,M: low-rank factors, Ahat = L*M
% info: diagnostic structure
% elapsed: runtime

    tic;
    [m, n] = size(A);
    X = randn(n, r);
    Y = randn(m, r+l);

    AX = A * X;
    YtA = Y' * A;
    W = YtA * X;

    info.condW = local_cond(W);
    info.rankkept = r;
    info.l = l;

    if ~stabilized
        [Z, R] = qr(W,'econ');
        L = AX / R;
        M = Z' * YtA;
    else
        [Uw, Sw, Vw] = svd(W, 'econ');
        sig = diag(Sw);
        tol = 10 * eps * norm(A, "fro");
        idx = find(sig > tol);

        if isempty(idx)
            idx = 1;
        end

        info.rankKept = numel(idx);

        L = AX * Vw(:, idx) * diag(1 ./ sig(idx));
        M = Uw(:, idx)' * YtA;
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
