function test_sgmres_trunc_arnoldi
%TEST_SGMRES_TRUNC_ARNOLDI Deterministic smoke tests for sGMRES.

    rng(1, 'twister');

    s = 20;
    m = 30;
    zeta = 4;
    S = sparse_sign_embedding(s, m, zeta);

    assert(isequal(size(S), [s, m]));
    assert(all(full(sum(spones(S), 1)) == zeta));
    assert(norm(full(sum(abs(S).^2, 1)) - 1, Inf) < 1e-14);

    n = 8;
    A = diag(linspace(1, 3, n));
    xTrue = (1:n).';
    b = A * xTrue;
    x0 = zeros(n, 1);

    rng(2, 'twister');
    [x, info] = sgmres_trunc_arnoldi( ...
        A, b, x0, n, n, 20, 4);

    assert(info.actual_dim == n);
    assert(~info.rank_deficient);
    assert(info.true_relres < 1e-10);
    assert(norm(x - xTrue) / norm(xTrue) < 1e-10);
    assert(info.solver_time_sec >= 0);
    assert(info.diagnostic_time_sec >= 0);

    checkpoints = [0, 1, 3, 6];
    rng(3, 'twister');
    [~, historyInfo] = sgmres_trunc_arnoldi( ...
        A, b, x0, 6, 2, 20, 4, checkpoints);

    assert(isequal(historyInfo.checkpoint_iterations, checkpoints(:)));
    assert(historyInfo.checkpoint_used_dimensions(1) == 0);
    assert(abs(historyInfo.checkpoint_true_relres(1) - 1) < 1e-14);

    for checkpointIndex = 2:numel(checkpoints)
        checkpoint = checkpoints(checkpointIndex);
        rng(3, 'twister');
        [~, checkpointInfo] = sgmres_trunc_arnoldi( ...
            A, b, x0, checkpoint, min(2, checkpoint), 20, 4);

        assert(abs( ...
            historyInfo.checkpoint_true_relres(checkpointIndex) - ...
            checkpointInfo.true_relres) < 1e-13);
    end

    [xExact, exactInfo] = sgmres_trunc_arnoldi( ...
        A, b, xTrue, 3, 2, 8, 1);

    assert(isequal(xExact, xTrue));
    assert(exactInfo.actual_dim == 0);
    assert(exactInfo.true_relres == 0);

    fprintf('test_sgmres_trunc_arnoldi passed.\n');
end
