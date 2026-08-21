function [x, info] = sgmres_trunc_arnoldi( ...
    A, b, x0, ell, q, s, zeta, checkpoints)
%SGMRES_TRUNC_ARNOLDI Basic implementation of sGMRES.
%
% A           is the matrix in A*x=b
% b           is the right-hand side
% x0          is the initial guess
% ell         is the number of Krylov iterations
% q           is the truncation parameter
% s           is the number of rows in the sketch
% zeta        is the number of nonzeros in each column of the sketch
% checkpoints is an optional vector of Krylov dimensions at which the true
%             relative residual is evaluated. These diagnostic calculations
%             are excluded from info.solver_time_sec.

    if nargin < 8
        checkpoints = [];
    end

    checkpoints = checkpoints(:);

    if any(~isfinite(checkpoints)) || ...
            any(checkpoints < 0) || ...
            any(checkpoints > ell) || ...
            any(checkpoints ~= floor(checkpoints))
        error('checkpoints must contain integers between 0 and ell.');
    end

    % Make b and x0 column vectors.
    b = b(:);
    x0 = x0(:);

    m = length(b);

    if length(x0) ~= m
        error('b and x0 must have the same length.');
    end

    if size(A, 1) ~= m || size(A, 2) ~= m
        error('A must be a square matrix with the same size as b.');
    end

    if q > ell
        error('q must not be larger than ell.');
    end

    if s < ell + 1
        error('s must be at least ell + 1.');
    end

    if zeta > s
        error('zeta must not be larger than s.');
    end

    normb = norm(b);

    if normb == 0
        error('b must be nonzero.');
    end

    solverTimer = tic;

    %% Step 1: Compute the initial residual.

    r0 = b - A * x0;
    beta = norm(r0);

    if beta == 0
        x = x0;

        info.iter = 0;
        info.actual_dim = 0;
        info.initial_relres = 0;
        info.sketch_resnorm = 0;
        info.sketch_relres = 0;
        info.true_resnorm = 0;
        info.true_relres = 0;
        info.cond_sketch = NaN;
        info.rank_sketch = 0;
        info.rank_deficient = false;
        info.breakdown = false;
        info.solver_time_sec = toc(solverTimer);
        info.diagnostic_time_sec = 0;
        info.checkpoint_iterations = checkpoints;
        info.checkpoint_used_dimensions = zeros(size(checkpoints));
        info.checkpoint_true_relres = zeros(size(checkpoints));
        info.q = q;
        info.s = s;
        info.zeta = zeta;
        return
    end

    %% Step 2: Construct the sketching matrix.

    S = sparse_sign_embedding(s, m, zeta);
    g = S * r0;

    %% Step 3: Prepare the basis matrix B and sketched matrix C.

    B = zeros(m, ell);
    C = zeros(s, ell);

    B(:, 1) = r0 / beta;

    ellUsed = ell;
    breakdown = false;

    %% Step 4: Generate the q-truncated Arnoldi basis.

    for j = 1:ell
        % Apply A to the current basis vector.
        v = A * B(:, j);

        % Store the corresponding column of S*A*B.
        C(:, j) = S * v;

        % The last basis vector does not need a successor.
        if j == ell
            continue
        end

        w = v;

        % Orthogonalise only against the most recent q vectors.
        firstIndex = max(1, j - q + 1);

        for i = firstIndex:j
            coefficient = B(:, i)' * w;
            w = w - coefficient * B(:, i);
        end

        wNorm = norm(w);

        % Stop if a new basis vector cannot be generated.
        if wNorm <= 1e-14 * max(1, norm(v))
            ellUsed = j;
            breakdown = true;
            break
        end

        B(:, j + 1) = w / wNorm;
    end

    % Remove unused columns if Arnoldi stopped early.
    B = B(:, 1:ellUsed);
    C = C(:, 1:ellUsed);

    %% Step 5: Solve the small sketched least-squares problem.

    % Thin QR factorisation: C = Q*R.
    [Q, R] = qr(C, 'econ');

    % Solve R*y = Q'*g by back substitution.
    y = R \ (Q' * g);

    %% Step 6: Form the approximate solution.

    x = x0 + B * y;

    %% Step 7: Compute the cheap sketched residual.

    sketchResidual = g - C * y;
    sketchResnorm = norm(sketchResidual);

    if norm(g) == 0
        sketchRelres = NaN;
    else
        sketchRelres = sketchResnorm / norm(g);
    end

    solverTime = toc(solverTimer);

    %% Step 8: Compute diagnostics separately.

    diagnosticTimer = tic;

    trueResidual = b - A * x;
    trueResnorm = norm(trueResidual);
    trueRelres = trueResnorm / normb;

    sketchRank = rank(C);
    sketchCondition = cond(C);

    %% Step 9: Evaluate requested true-residual checkpoints.

    checkpointUsedDimensions = zeros(size(checkpoints));
    checkpointTrueRelres = zeros(size(checkpoints));

    for checkpointIndex = 1:numel(checkpoints)
        checkpoint = checkpoints(checkpointIndex);

        if checkpoint == 0
            checkpointTrueRelres(checkpointIndex) = beta / normb;
            continue
        end

        usedDimension = min(checkpoint, ellUsed);
        checkpointUsedDimensions(checkpointIndex) = usedDimension;

        Cj = C(:, 1:usedDimension);
        [Qj, Rj] = qr(Cj, 'econ');
        yj = Rj \ (Qj' * g);
        xj = x0 + B(:, 1:usedDimension) * yj;

        checkpointTrueResidual = b - A * xj;
        checkpointTrueRelres(checkpointIndex) = ...
            norm(checkpointTrueResidual) / normb;
    end

    diagnosticTime = toc(diagnosticTimer);

    %% Store the results.

    info.iter = ellUsed;
    info.actual_dim = ellUsed;
    info.initial_relres = beta / normb;
    info.sketch_resnorm = sketchResnorm;
    info.sketch_relres = sketchRelres;
    info.true_resnorm = trueResnorm;
    info.true_relres = trueRelres;
    info.cond_sketch = sketchCondition;
    info.rank_sketch = sketchRank;
    info.rank_deficient = sketchRank < ellUsed;
    info.breakdown = breakdown;
    info.solver_time_sec = solverTime;
    info.diagnostic_time_sec = diagnosticTime;
    info.checkpoint_iterations = checkpoints;
    info.checkpoint_used_dimensions = checkpointUsedDimensions;
    info.checkpoint_true_relres = checkpointTrueRelres;
    info.q = q;
    info.s = s;
    info.zeta = zeta;
end
