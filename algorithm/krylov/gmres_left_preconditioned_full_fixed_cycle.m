function [x, info] = gmres_left_preconditioned_full_fixed_cycle( ...
    operatorFunction, preconditionerFunction, b, x0, ...
    cycleLength, targetOriginalTolerance, displayProgress)
%GMRES_LEFT_PRECONDITIONED_FULL_FIXED_CYCLE Delayed-assembly GMRES cycle.
%
% This timing control builds a fixed left-preconditioned Arnoldi cycle,
% solves the small least-squares problem at every step, and forms the full
% solution only once after the cycle. Independent original and
% preconditioned residuals are evaluated after the solver timer stops. This
% matches the cycle-end assembly used by the Tucker-GMRES implementation.
%
% Thesis/experiment notation (Section 6.2 full baseline):
%   operatorFunction, preconditionerFunction  <->  A_0, M_r
%   b, x0, x                                 <->  b_0, x_0, x_j
%   cycleLength                              <->  four-step basis limit
%   beta                                     <->  ||r_0^P||_2
%   V(:,j), w                                <->  v_j, M_r A_0 v_j
%   H(1:j+1,1:j), y                          <->  \bar H_j, y_j
%   smallResidual                            <->  beta e_1-\bar H_j y_j
% The word cycle means one finite Arnoldi run here; this routine does not
% implement a restart loop.


%% 1. Check the inputs and form the starting residual

if nargin < 7
    displayProgress = false;
end

if cycleLength < 1 || cycleLength ~= round(cycleLength)
    error('cycleLength must be a positive integer.');
end

if targetOriginalTolerance <= 0
    error('targetOriginalTolerance must be positive.');
end

b = b(:);
x0 = x0(:);

if length(b) ~= length(x0)
    error('b and x0 must have the same number of entries.');
end

normB = norm(b);

if normB == 0
    error('The right-hand side b must be nonzero.');
end

wallTimer = tic;
solverTimer = tic;

preconditionedRightHandSide = preconditionerFunction(b);
normPreconditionedRightHandSide = ...
    norm(preconditionedRightHandSide);

if normPreconditionedRightHandSide == 0
    error('The preconditioned right-hand side must be nonzero.');
end

initialOriginalResidualVector = b - operatorFunction(x0);
initialPreconditionedResidualVector = ...
    preconditionerFunction(initialOriginalResidualVector);
beta = norm(initialPreconditionedResidualVector);

initialOriginalResidual = ...
    norm(initialOriginalResidualVector) / normB;
initialPreconditionedResidual = ...
    beta / normPreconditionedRightHandSide;

if beta <= 1e-14 * normPreconditionedRightHandSide
    error('The preconditioned starting residual is too small.');
end


%% 2. Build the fixed Arnoldi cycle

numberOfUnknowns = length(b);
V = zeros(numberOfUnknowns, cycleLength + 1);
V(:, 1) = initialPreconditionedResidualVector / beta;
H = zeros(cycleLength + 1, cycleLength);
computedResidualHistory = zeros(cycleLength, 1);

y = zeros(0, 1);
breakdown = false;
stopReason = "cycle_length";

for j = 1:cycleLength

    w = preconditionerFunction(operatorFunction(V(:, j)));

    for i = 1:j
        H(i, j) = V(:, i)' * w;
        w = w - H(i, j) * V(:, i);
    end

    H(j + 1, j) = norm(w);

    smallRightHandSide = zeros(j + 1, 1);
    smallRightHandSide(1) = beta;
    currentHessenberg = H(1:j + 1, 1:j);
    y = currentHessenberg \ smallRightHandSide;
    smallResidual = ...
        smallRightHandSide - currentHessenberg * y;
    computedResidualHistory(j) = ...
        norm(smallResidual) / normPreconditionedRightHandSide;

    if displayProgress
        fprintf([ ...
            'Delayed full GMRES iteration %2d: ', ...
            'computed preconditioned residual %.3e\n'], ...
            j, computedResidualHistory(j));
    end

    if H(j + 1, j) <= ...
            1e-14 * max(1, norm(currentHessenberg, 'fro'))
        breakdown = true;
        stopReason = "breakdown";
        break
    end

    if j < cycleLength
        V(:, j + 1) = w / H(j + 1, j);
    end

end


%% 3. Assemble the solution once and stop the solver timer

numberOfIterations = j;
x = x0 + V(:, 1:numberOfIterations) * y;
solverTimeSec = toc(solverTimer);


%% 4. Evaluate independent residuals outside the solver timer

diagnosticTimer = tic;

originalResidualVector = b - operatorFunction(x);
originalTrueResidual = norm(originalResidualVector) / normB;
truePreconditionedResidualVector = ...
    preconditionerFunction(originalResidualVector);
truePreconditionedResidual = ...
    norm(truePreconditionedResidualVector) / ...
    normPreconditionedRightHandSide;

diagnosticTimeSec = toc(diagnosticTimer);
wallTimeSec = toc(wallTimer);


%% 5. Return the compact cycle evidence

info.iterations = numberOfIterations;
info.converged = ...
    originalTrueResidual <= targetOriginalTolerance;
info.breakdown = breakdown;
info.stop_reason = stopReason;
info.initial_original_true_relative_residual = ...
    initialOriginalResidual;
info.initial_preconditioned_true_relative_residual = ...
    initialPreconditionedResidual;
info.computed_preconditioned_relative_residual = ...
    computedResidualHistory(1:numberOfIterations);
info.true_preconditioned_relative_residual = ...
    truePreconditionedResidual;
info.original_true_relative_residual = originalTrueResidual;
info.preconditioned_residual_gap = abs( ...
    truePreconditionedResidual - ...
    computedResidualHistory(numberOfIterations));
info.norm_preconditioned_right_hand_side = ...
    normPreconditionedRightHandSide;
info.H = H(1:numberOfIterations + 1, 1:numberOfIterations);
info.y = y;
info.number_of_basis_vectors = numberOfIterations + 1;
info.orthogonalisation_passes = 1;
info.solver_time_sec = solverTimeSec;
info.diagnostic_time_sec = diagnosticTimeSec;
info.wall_time_sec = wallTimeSec;

end
