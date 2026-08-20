function [x, info] = gmres_left_preconditioned_full_fixed_cycle( ...
    A, P, b, x0, ...
    m, tol, verbose)
%GMRES_LEFT_PRECONDITIONED_FULL_FIXED_CYCLE Delayed-assembly GMRES cycle.

if nargin < 7
    verbose = false;
end

if m < 1 || m ~= round(m)
    error('m must be a positive integer.');
end

if tol <= 0
    error('tol must be positive.');
end

b = b(:);
x0 = x0(:);

if length(b) ~= length(x0)
    error('b and x0 must have the same number of entries.');
end

nrmb = norm(b);

if nrmb == 0
    error('The right-hand side b must be nonzero.');
end

wall_timer = tic;
solver_timer = tic;

Pb = P(b);
norm_pb = norm(Pb);

if norm_pb == 0
    error('The preconditioned right-hand side must be nonzero.');
end

r0 = b - A(x0);
Pr0 = P(r0);
beta = norm(Pr0);

true_res0 = norm(r0) / nrmb;
precond_res0 = beta / norm_pb;

if beta <= 1e-14 * norm_pb
    error('The preconditioned starting residual is too small.');
end

% Arnoldi cycle

n = length(b);
V = zeros(n, m + 1);
V(:, 1) = Pr0 / beta;
H = zeros(m + 1, m);
res = zeros(m, 1);

y = zeros(0, 1);
breakdown = false;
stop_reason = "cycle_length";

for j = 1:m

    w = P(A(V(:, j)));

    for i = 1:j
        H(i, j) = V(:, i)' * w;
        w = w - H(i, j) * V(:, i);
    end

    H(j + 1, j) = norm(w);

    rhs = zeros(j + 1, 1);
    rhs(1) = beta;
    Hj = H(1:j + 1, 1:j);
    y = Hj \ rhs;
    res(j) = norm(rhs - Hj * y) / norm_pb;

    if verbose
        fprintf([ ...
            'Delayed full GMRES iteration %2d: ', ...
            'computed preconditioned residual %.3e\n'], ...
            j, res(j));
    end

    if H(j + 1, j) <= ...
            1e-14 * max(1, norm(Hj, 'fro'))
        breakdown = true;
        stop_reason = "breakdown";
        break
    end

    if j < m
        V(:, j + 1) = w / H(j + 1, j);
    end

end

niter = j;
x = x0 + V(:, 1:niter) * y;
solver_time = toc(solver_timer);

% Independent residuals

diagnostic_timer = tic;

r = b - A(x);
true_res = norm(r) / nrmb;
precond_res = norm(P(r)) / norm_pb;

diagnostic_time_sec = toc(diagnostic_timer);
wall_time_sec = toc(wall_timer);

info.iterations = niter;
info.converged = ...
    true_res <= tol;
info.breakdown = breakdown;
info.stop_reason = stop_reason;
info.initial_original_true_relative_residual = ...
    true_res0;
info.initial_preconditioned_true_relative_residual = ...
    precond_res0;
info.computed_preconditioned_relative_residual = ...
    res(1:niter);
info.true_preconditioned_relative_residual = ...
    precond_res;
info.original_true_relative_residual = true_res;
info.preconditioned_residual_gap = abs( ...
    precond_res - res(niter));
info.norm_preconditioned_right_hand_side = ...
    norm_pb;
info.H = H(1:niter + 1, 1:niter);
info.y = y;
info.number_of_basis_vectors = niter + 1;
info.orthogonalisation_passes = 1;
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = diagnostic_time_sec;
info.wall_time_sec = wall_time_sec;

end
