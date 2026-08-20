function [U, info] = run_left_preconditioned_tucker_gmres_cycle(A, P, F0, U0, ...
    maxit, tol, fixed_tol, ...
    tol_mode, true_residual, ...
    precond_residual, verbose, ...
    max_basis_entries, max_rank, ...
    relax_budget, max_relax_tol, ...
    check_it)
%RUN_LEFT_PRECONDITIONED_TUCKER_GMRES_CYCLE Run one Tucker-GMRES cycle.

if ~isa(F0, 'ttensor') || ~isa(U0, 'ttensor')
    error('F0 and U0 must be Tensor Toolbox ttensor objects.');
end

if ~isequal(size(F0), size(U0))
    error('F0 and U0 must have the same tensor dimensions.');
end

if maxit < 1 || maxit ~= floor(maxit)
    error('maxit must be a positive integer.');
end

if tol <= 0 || tol >= 1
    error('tol must be between 0 and 1.');
end

if fixed_tol <= 0 || fixed_tol >= 1
    error('fixed_tol must be between 0 and 1.');
end

if nargin < 14
    relax_budget = tol;
end

if nargin < 15
    max_relax_tol = 1 - eps;
end
if nargin < 16
    check_it = [];
end
check_it = normalise_tucker_diagnostic_iterations( ...
    check_it, maxit);

if relax_budget <= 0 || relax_budget >= 1
    error('relax_budget must be between 0 and 1.');
end

if max_relax_tol <= 0 || max_relax_tol >= 1
    error('max_relax_tol must be between 0 and 1.');
end

tol_mode = lower(string(tol_mode));

if tol_mode ~= "fixed" && tol_mode ~= "relaxed"
    error('tol_mode must be "fixed" or "relaxed".');
end

if ~islogical(verbose) || ~isscalar(verbose)
    error('verbose must be one logical value.');
end

if max_basis_entries <= 0
    error('max_basis_entries must be positive.');
end

max_ranks = normalise_tucker_rank_cap(max_rank, size(F0));

norm_f0 = norm(F0);

if norm_f0 == 0
    error('The original right-hand side F0 must be nonzero.');
end

if tol_mode == "relaxed"
    starting_compression_tolerance = min(relax_budget, max_relax_tol);
    solution_tol = starting_compression_tolerance;
else
    solution_tol = fixed_tol;
    starting_compression_tolerance = fixed_tol;
end

% Form the preconditioned starting residual

wall_timer = tic;
solver_time = 0;

phase_timing.operator_application_time_sec = 0;
phase_timing.preconditioner_application_time_sec = 0;
phase_timing.outer_round_time_sec = 0;
phase_timing.initial_residual_formation_time_sec = 0;
phase_timing.orthogonalisation_inner_product_time_sec = 0;
phase_timing.orthogonalisation_subtraction_round_time_sec = 0;
phase_timing.small_least_squares_time_sec = 0;
phase_timing.basis_normalisation_time_sec = 0;
phase_timing.initial_basis_normalisation_time_sec = 0;
phase_timing.solution_assembly_time_sec = 0;
phase_timing.original_true_residual_time_sec = 0;
phase_timing.preconditioned_residual_diagnostic_time_sec = 0;

phase_timing.operator_application_history_sec = ...
    zeros(maxit, 1);
phase_timing.preconditioner_application_history_sec = ...
    zeros(maxit, 1);
phase_timing.outer_round_history_sec = zeros(maxit, 1);
phase_timing.orthogonalisation_inner_product_history_sec = ...
    zeros(maxit, 1);
phase_timing.orthogonalisation_subtraction_round_history_sec = ...
    zeros(maxit, 1);
phase_timing.small_least_squares_history_sec = ...
    zeros(maxit, 1);
phase_timing.basis_normalisation_history_sec = ...
    zeros(maxit, 1);

kernel_timing = empty_tucker_kernel_timing();
phase_kernels = empty_phase_kernel_timing();

solver_timer = tic;

component_timer = tic;
[AU0, initial_A_info] = A(U0, starting_compression_tolerance, max_ranks);
initial_operator_application_time = toc(component_timer);
phase_timing.operator_application_time_sec = ...
    phase_timing.operator_application_time_sec + ...
    initial_operator_application_time;
kernel_timing = add_operation_kernel_timing( ...
    kernel_timing, initial_A_info);
phase_kernels.operator_application = ...
    add_operation_kernel_timing( ...
        phase_kernels.operator_application, initial_A_info);

component_timer = tic;
original_residual = tucker_axpby_exact(F0, 1, AU0, -1);
phase_timing.initial_residual_formation_time_sec = toc(component_timer);

component_timer = tic;
[r_precond, initial_preconditioner_info] = P(original_residual, starting_compression_tolerance, max_ranks);
initial_preconditioner_application_time = toc(component_timer);
phase_timing.preconditioner_application_time_sec = ...
    phase_timing.preconditioner_application_time_sec + ...
    initial_preconditioner_application_time;
kernel_timing = add_operation_kernel_timing( ...
    kernel_timing, initial_preconditioner_info);
phase_kernels.preconditioner_application = ...
    add_operation_kernel_timing( ...
        phase_kernels.preconditioner_application, ...
        initial_preconditioner_info);

component_timer = tic;
[R0P, initial_outer_round_info] = tucker_round(r_precond, starting_compression_tolerance, max_ranks);
initial_outer_round_time = toc(component_timer);
phase_timing.outer_round_time_sec = ...
    phase_timing.outer_round_time_sec + initial_outer_round_time;
kernel_timing = add_operation_kernel_timing( ...
    kernel_timing, initial_outer_round_info);
phase_kernels.outer_round = add_operation_kernel_timing( ...
    phase_kernels.outer_round, initial_outer_round_info);

beta = norm(R0P);
solver_time = solver_time + toc(solver_timer);

if beta <= 1e-14 * norm_f0
    error(['The preconditioned starting residual is nearly zero. ', 'Use a more accurate preconditioner application or ', 'recompression tolerance.']);
end

% Evaluate the initial residuals independently

diagnostic_timer = tic;
true_res0 = true_residual(U0);
initial_original_true_residual_time = toc(diagnostic_timer);
phase_timing.original_true_residual_time_sec = ...
    phase_timing.original_true_residual_time_sec + ...
    initial_original_true_residual_time;

diagnostic_timer = tic;
beta0 = precond_residual(U0);
initial_preconditioned_residual_diagnostic_time = toc(diagnostic_timer);
phase_timing.preconditioned_residual_diagnostic_time_sec = ...
    phase_timing.preconditioned_residual_diagnostic_time_sec + ...
    initial_preconditioned_residual_diagnostic_time;
diagnostic_time = phase_timing.original_true_residual_time_sec + ...
    phase_timing.preconditioned_residual_diagnostic_time_sec;

phase_timing.initial_operator_application_time_sec = ...
    initial_operator_application_time;
phase_timing.initial_preconditioner_application_time_sec = ...
    initial_preconditioner_application_time;
phase_timing.initial_outer_round_time_sec = initial_outer_round_time;
phase_timing.initial_original_true_residual_time_sec = ...
    initial_original_true_residual_time;
phase_timing.initial_preconditioned_residual_diagnostic_time_sec = ...
    initial_preconditioned_residual_diagnostic_time;

if true_res0 <= tol

    U = U0;
    wall_time = toc(wall_timer);
    info = empty_cycle_info(tol_mode, max_ranks, beta, true_res0, beta0 / beta, ...
        solver_time, diagnostic_time, wall_time, ...
        tol, fixed_tol, ...
        relax_budget, max_relax_tol);
    info.timing = finalise_cycle_timing( ...
        phase_timing, 0, solver_time, diagnostic_time, ...
        wall_time, kernel_timing, phase_kernels);
    return

end

% Prepare the Arnoldi basis and histories

d = ndims(F0);

V = cell(maxit + 1, 1);
H = zeros(maxit + 1, maxit);

true_res_hist = true_res0;
true_res_it = 0;
true_preconditioned_relative_residual_history = beta0 / beta;
true_preconditioned_residual_iteration = 0;
res_hist = zeros(maxit, 1);
eta_history = zeros(maxit, 1);

basis_ranks = zeros(maxit + 1, d);
basis_storage = zeros(maxit + 1, 1);

solver_time_history = zeros(maxit, 1);
diagnostic_time_history = zeros(maxit, 1);
wall_time_history = zeros(maxit, 1);

cap_hist = false(maxit, 1);
operator_product_maximum_local_error_history = zeros(maxit, 1);
orthogonalisation_maximum_local_error_history = zeros(maxit, 1);
solution_maximum_local_error_history = zeros(maxit, 1);

solver_timer = tic;
V{1} = (1 / beta) * R0P;
phase_timing.initial_basis_normalisation_time_sec = toc(solver_timer);
phase_timing.basis_normalisation_time_sec = ...
    phase_timing.basis_normalisation_time_sec + ...
    phase_timing.initial_basis_normalisation_time_sec;
solver_time = solver_time + ...
    phase_timing.initial_basis_normalisation_time_sec;
basis_ranks(1, :) = size(V{1}.core);
basis_storage(1) = tucker_entries_from_rank_local(size(F0), basis_ranks(1, :));

if basis_storage(1) > max_basis_entries
    error('The initial Tucker basis exceeds the storage safety limit.');
end

computed_residual_norm_previous = beta;
basis_limit_reached = false;
cycle_stopped_on_computed_residual = false;
stop_reason = "iteration_limit";

[initial_cap_active, initial_local_error] = summarise_operation_info(initial_A_info, initial_preconditioner_info, initial_outer_round_info);

% Run the left-preconditioned Arnoldi cycle

for j = 1:maxit

    if tol_mode == "relaxed"

        previous_relative_residual = computed_residual_norm_previous / beta;

        eta = relax_budget / previous_relative_residual;

        eta = min([eta, max_relax_tol, 1 - eps]);

    else
        eta = fixed_tol;
    end

    eta_history(j) = eta;
    solver_timer = tic;

% Apply the recompressed preconditioned operator product

    component_timer = tic;
    [A0V, A_info] = A(V{j}, eta, max_ranks);
    one_operator_application_time = toc(component_timer);
    phase_timing.operator_application_time_sec = ...
        phase_timing.operator_application_time_sec + ...
        one_operator_application_time;
    phase_timing.operator_application_history_sec(j) = ...
        one_operator_application_time;
    kernel_timing = add_operation_kernel_timing( ...
        kernel_timing, A_info);
    phase_kernels.operator_application = ...
        add_operation_kernel_timing( ...
            phase_kernels.operator_application, A_info);

    component_timer = tic;
    [preconditioned_product, preconditioner_info] = P(A0V, eta, max_ranks);
    one_preconditioner_application_time = toc(component_timer);
    phase_timing.preconditioner_application_time_sec = ...
        phase_timing.preconditioner_application_time_sec + ...
        one_preconditioner_application_time;
    phase_timing.preconditioner_application_history_sec(j) = ...
        one_preconditioner_application_time;
    kernel_timing = add_operation_kernel_timing( ...
        kernel_timing, preconditioner_info);
    phase_kernels.preconditioner_application = ...
        add_operation_kernel_timing( ...
            phase_kernels.preconditioner_application, ...
            preconditioner_info);

    component_timer = tic;
    [W, product_round_info] = tucker_round(preconditioned_product, eta, max_ranks);
    one_outer_round_time = toc(component_timer);
    phase_timing.outer_round_time_sec = ...
        phase_timing.outer_round_time_sec + one_outer_round_time;
    phase_timing.outer_round_history_sec(j) = one_outer_round_time;
    kernel_timing = add_operation_kernel_timing( ...
        kernel_timing, product_round_info);
    phase_kernels.outer_round = add_operation_kernel_timing( ...
        phase_kernels.outer_round, product_round_info);

    arnoldi_product_norm = norm(W);

    [product_cap_active, product_local_error] = summarise_operation_info(A_info, preconditioner_info, product_round_info);

    cap_hist(j) = product_cap_active;
    operator_product_maximum_local_error_history(j) = product_local_error;

% Perform one modified Gram--Schmidt pass

    for i = 1:j

        component_timer = tic;
        correction = innerprod(V{i}, W);
        one_inner_product_time = toc(component_timer);
        phase_timing.orthogonalisation_inner_product_time_sec = ...
            phase_timing.orthogonalisation_inner_product_time_sec + ...
            one_inner_product_time;
        phase_timing.orthogonalisation_inner_product_history_sec(j) = ...
            phase_timing.orthogonalisation_inner_product_history_sec(j) + ...
            one_inner_product_time;

        H(i, j) = correction;

        component_timer = tic;
        [W, subtraction_info] = tucker_axpby_round(W, 1, V{i}, -correction, eta, max_ranks);
        one_subtraction_round_time = toc(component_timer);
        phase_timing.orthogonalisation_subtraction_round_time_sec = ...
            phase_timing.orthogonalisation_subtraction_round_time_sec + ...
            one_subtraction_round_time;
        phase_timing.orthogonalisation_subtraction_round_history_sec(j) = ...
            phase_timing.orthogonalisation_subtraction_round_history_sec(j) + ...
            one_subtraction_round_time;
        kernel_timing = add_operation_kernel_timing( ...
            kernel_timing, subtraction_info);
        phase_kernels.orthogonalisation_subtraction_and_round = ...
            add_operation_kernel_timing( ...
                phase_kernels.orthogonalisation_subtraction_and_round, ...
                subtraction_info);

        cap_hist(j) = cap_hist(j) || subtraction_info.rank_cap_active;

        orthogonalisation_maximum_local_error_history(j) = max(orthogonalisation_maximum_local_error_history(j), subtraction_info.relative_error_estimate);

    end

% Form the Hessenberg problem

    H(j + 1, j) = norm(W);

    breakdown_threshold = max(100 * eps, eta) * max(1, arnoldi_product_norm);
    numerical_arnoldi_breakdown = H(j + 1, j) <= breakdown_threshold;

    if numerical_arnoldi_breakdown
        H(j + 1, j) = 0;
    end

    basis_ranks(j + 1, :) = size(W.core);
    basis_storage(j + 1) = basis_storage(j) + tucker_entries_from_rank_local(size(F0), basis_ranks(j + 1, :));

    if basis_storage(j + 1) > max_basis_entries
        basis_limit_reached = true;
    end

    component_timer = tic;
    rhs = zeros(j + 1, 1);
    rhs(1) = beta;

    y = lsqminnorm(H(1:j + 1, 1:j), rhs, sqrt(eps));
    small_residual = rhs - H(1:j + 1, 1:j) * y;
    computed_residual_norm = norm(small_residual);

    one_small_least_squares_time = toc(component_timer);
    phase_timing.small_least_squares_time_sec = ...
        phase_timing.small_least_squares_time_sec + ...
        one_small_least_squares_time;
    phase_timing.small_least_squares_history_sec(j) = ...
        one_small_least_squares_time;

    res_hist(j) = computed_residual_norm / beta;

    solver_time = solver_time + toc(solver_timer);

    if any(check_it == j)
        diagnostic_timer = tic;
        U_check = U0;
        for diagnostic_basis_index = 1:j
            U_check = tucker_axpby_round( ...
                U_check, 1, V{diagnostic_basis_index}, ...
                y(diagnostic_basis_index), ...
                solution_tol, max_ranks);
        end
        check_res = ...
            true_residual(U_check);
        diagnostic_time = diagnostic_time + toc(diagnostic_timer);
        true_res_hist = ...
            [true_res_hist; check_res]; %#ok<AGROW>
        true_res_it = ...
            [true_res_it; j]; %#ok<AGROW>
        clear U_check
    end

    solver_time_history(j) = solver_time;
    diagnostic_time_history(j) = diagnostic_time;
    wall_time_history(j) = toc(wall_timer);

    if verbose
        fprintf([ ...
            'Iteration %3d: eta = %.3e, computed preconditioned ', ...
            'residual = %.3e\n'], ...
            j, eta, ...
            res_hist(j));
    end

    if res_hist(j) <= tol
        cycle_stopped_on_computed_residual = true;
        stop_reason = "computed_preconditioned_target";
        break
    end

    if res_hist(j) <= 100 * eps
        cycle_stopped_on_computed_residual = true;
        stop_reason = "computed_preconditioned_floor";
        break
    end

    if basis_limit_reached
        stop_reason = "basis_memory";
        break
    end

    if numerical_arnoldi_breakdown
        stop_reason = "arnoldi_breakdown";
        break
    end

    solver_timer = tic;
    V{j + 1} = (1 / H(j + 1, j)) * W;
    one_basis_normalisation_time = toc(solver_timer);
    phase_timing.basis_normalisation_time_sec = ...
        phase_timing.basis_normalisation_time_sec + ...
        one_basis_normalisation_time;
    phase_timing.basis_normalisation_history_sec(j) = ...
        one_basis_normalisation_time;
    solver_time = solver_time + one_basis_normalisation_time;
    solver_time_history(j) = solver_time;
    wall_time_history(j) = toc(wall_timer);
    computed_residual_norm_previous = computed_residual_norm;

end

% Assemble the Tucker solution once

niter_done = j;
solver_timer = tic;
U = U0;

for i = 1:niter_done

    [U, solution_addition_info] = tucker_axpby_round(U, 1, V{i}, y(i), solution_tol, max_ranks);
    kernel_timing = add_operation_kernel_timing( ...
        kernel_timing, solution_addition_info);
    phase_kernels.solution_assembly = ...
        add_operation_kernel_timing( ...
            phase_kernels.solution_assembly, solution_addition_info);

    cap_hist(niter_done) = cap_hist(niter_done) || solution_addition_info.rank_cap_active;

    solution_maximum_local_error_history(niter_done) = max(solution_maximum_local_error_history(niter_done), solution_addition_info.relative_error_estimate);

end

solution_ranks = size(U.core);
solution_rank_iteration = niter_done;
phase_timing.solution_assembly_time_sec = toc(solver_timer);
solver_time = solver_time + phase_timing.solution_assembly_time_sec;

% Recompute both cycle-end residuals independently

diagnostic_timer = tic;
final_true_res = true_residual(U);
final_original_true_residual_time = toc(diagnostic_timer);
phase_timing.original_true_residual_time_sec = ...
    phase_timing.original_true_residual_time_sec + ...
    final_original_true_residual_time;

diagnostic_timer = tic;
final_preconditioned_residual_norm = precond_residual(U);
final_preconditioned_residual_diagnostic_time = toc(diagnostic_timer);
phase_timing.preconditioned_residual_diagnostic_time_sec = ...
    phase_timing.preconditioned_residual_diagnostic_time_sec + ...
    final_preconditioned_residual_diagnostic_time;

final_true_preconditioned_relative_residual = final_preconditioned_residual_norm / beta;

diagnostic_time = phase_timing.original_true_residual_time_sec + ...
    phase_timing.preconditioned_residual_diagnostic_time_sec;

if true_res_it(end) == niter_done
    true_res_hist(end) = final_true_res;
else
    true_res_hist = ...
        [true_res_hist; final_true_res];
    true_res_it = ...
        [true_res_it; niter_done];
end
true_preconditioned_relative_residual_history = [true_preconditioned_relative_residual_history; final_true_preconditioned_relative_residual];
true_preconditioned_residual_iteration = [true_preconditioned_residual_iteration; niter_done];

preconditioned_residual_gap = abs(final_true_preconditioned_relative_residual - res_hist(niter_done));

solver_time_history(niter_done) = solver_time;
diagnostic_time_history(niter_done) = diagnostic_time;
wall_time_history(niter_done) = toc(wall_timer);

if verbose
    fprintf([ ...
        'Cycle complete at iteration %d: computed preconditioned ', ...
        'residual = %.3e, original true residual = %.3e\n'], ...
        niter_done, ...
        res_hist( ...
            niter_done), ...
        final_true_res);
end

info.iterations = niter_done;
info.converged = final_true_res <= tol;
info.stop_reason = stop_reason;
info.tolerance_mode = tol_mode;
info.target_tolerance = tol;
info.fixed_compression_tolerance = fixed_tol;
info.relaxation_error_budget = relax_budget;
info.maximum_relaxed_tolerance = max_relax_tol;
info.maximum_multilinear_rank = max_ranks;
info.cycle_starting_preconditioned_residual_norm = beta;

info.true_relative_residual = ...
    true_res_hist;
info.true_residual_iteration = true_res_it;
info.true_preconditioned_relative_residual = ...
    true_preconditioned_relative_residual_history;
info.true_preconditioned_residual_iteration = ...
    true_preconditioned_residual_iteration;
info.computed_preconditioned_relative_residual = ...
    res_hist( ...
        1:niter_done);
info.preconditioned_residual_gap = preconditioned_residual_gap;
info.preconditioned_residual_gap_iteration = ...
    niter_done;
info.eta = eta_history(1:niter_done);

info.solution_ranks = solution_ranks;
info.solution_rank_iteration = solution_rank_iteration;
info.basis_ranks = ...
    basis_ranks(1:niter_done + 1, :);
info.basis_storage_history_entries = ...
    basis_storage(1:niter_done + 1);
info.peak_basis_storage_entries = ...
    max(info.basis_storage_history_entries);

info.H = H(1:niter_done + 1, ...
    1:niter_done);
info.y = y;

info.rank_cap_active_history = ...
    cap_hist(1:niter_done);
info.rank_cap_active = initial_cap_active || ...
    any(info.rank_cap_active_history);
info.initial_maximum_local_recompression_error = initial_local_error;
info.operator_product_maximum_local_recompression_error = ...
    operator_product_maximum_local_error_history( ...
        1:niter_done);
info.orthogonalisation_maximum_local_recompression_error = ...
    orthogonalisation_maximum_local_error_history( ...
        1:niter_done);
info.solution_maximum_local_recompression_error = ...
    solution_maximum_local_error_history( ...
        1:niter_done);

info.stopped_for_basis_memory = basis_limit_reached;
info.cycle_stopped_on_computed_residual = ...
    cycle_stopped_on_computed_residual;
info.orthogonalisation_passes = 1;

info.solver_time_history_sec = ...
    solver_time_history(1:niter_done);
info.diagnostic_time_history_sec = ...
    diagnostic_time_history(1:niter_done);
info.wall_time_history_sec = ...
    wall_time_history(1:niter_done);
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = diagnostic_time;
wall_time = toc(wall_timer);
info.wall_time_sec = wall_time;
info.solution_assembly_count = 1;
info.original_true_residual_evaluation_count = ...
    numel(true_res_hist);
info.timing = finalise_cycle_timing( ...
    phase_timing, niter_done, solver_time, ...
    diagnostic_time, wall_time, kernel_timing, phase_kernels);

end

function timing_by_phase = empty_phase_kernel_timing()
%EMPTY_PHASE_KERNEL_TIMING Initialise caller-attributed leaf timings.

timing_by_phase.operator_application = empty_tucker_kernel_timing();
timing_by_phase.preconditioner_application = empty_tucker_kernel_timing();
timing_by_phase.outer_round = empty_tucker_kernel_timing();
timing_by_phase.orthogonalisation_subtraction_and_round = ...
    empty_tucker_kernel_timing();
timing_by_phase.solution_assembly = empty_tucker_kernel_timing();

end

function total_timing = add_operation_kernel_timing(total_timing, op_info)
%ADD_OPERATION_KERNEL_TIMING Add leaf timings returned by one operation.

if isstruct(op_info) && isfield(op_info, 'kernel_timing')
    total_timing = add_tucker_kernel_timing( ...
        total_timing, op_info.kernel_timing);
end

end

function timing = finalise_cycle_timing( ...
    timing, completed_iterations, solver_time, diagnostic_time, ...
    wall_time, kernel_timing, phase_kernels)
%FINALISE_CYCLE_TIMING Trim histories and close the timing accounts.

history_fields = { ...
    'operator_application_history_sec', ...
    'preconditioner_application_history_sec', ...
    'outer_round_history_sec', ...
    'orthogonalisation_inner_product_history_sec', ...
    'orthogonalisation_subtraction_round_history_sec', ...
    'small_least_squares_history_sec', ...
    'basis_normalisation_history_sec'};

for field_idx = 1:numel(history_fields)
    field_name = history_fields{field_idx};
    timing.(field_name) = timing.(field_name)(1:completed_iterations);
end

timing.accounted_arnoldi_iteration_history_sec = ...
    timing.operator_application_history_sec + ...
    timing.preconditioner_application_history_sec + ...
    timing.outer_round_history_sec + ...
    timing.orthogonalisation_inner_product_history_sec + ...
    timing.orthogonalisation_subtraction_round_history_sec + ...
    timing.small_least_squares_history_sec + ...
    timing.basis_normalisation_history_sec;

timing.accounted_solver_time_sec = ...
    timing.operator_application_time_sec + ...
    timing.preconditioner_application_time_sec + ...
    timing.outer_round_time_sec + ...
    timing.initial_residual_formation_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.orthogonalisation_subtraction_round_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solver_time - timing.accounted_solver_time_sec;

timing.accounted_diagnostic_time_sec = ...
    timing.original_true_residual_time_sec + ...
    timing.preconditioned_residual_diagnostic_time_sec;
timing.unclassified_diagnostic_time_sec = ...
    diagnostic_time - timing.accounted_diagnostic_time_sec;

timing.solver_time_sec = solver_time;
timing.diagnostic_time_sec = diagnostic_time;
timing.wall_time_sec = wall_time;
timing.kernel = kernel_timing;
timing.kernel_by_phase = phase_kernels;

end

function [cap_active, maximum_local_error] = summarise_operation_info(varargin)
%SUMMARISE_OPERATION_INFO Combine diagnostics from nested operations.

cap_active = false;
maximum_local_error = 0;

for argument_index = 1:nargin

    op_info = varargin{argument_index};

    if isempty(op_info)
        continue
    end

    if isfield(op_info, 'rank_cap_active')
        cap_active = cap_active || op_info.rank_cap_active;
    end

    scalar_fields = { ...
        'relative_error_estimate', ...
        'maximum_local_recompression_error', ...
        'final_relative_error_estimate'};

    for field_idx = 1:numel(scalar_fields)

        field_name = scalar_fields{field_idx};

        if isfield(op_info, field_name)
            maximum_local_error = max( ...
                maximum_local_error, max(op_info.(field_name), [], 'all'));
        end

    end

    vector_fields = { ...
        'addition_relative_error_estimate'};

    for field_idx = 1:numel(vector_fields)

        field_name = vector_fields{field_idx};

        if isfield(op_info, field_name) && ...
                ~isempty(op_info.(field_name))
            maximum_local_error = max( ...
                maximum_local_error, ...
                max(op_info.(field_name), [], 'all'));
        end

    end

end

end

function number_of_entries = tucker_entries_from_rank_local(n, ranks)
%TUCKER_ENTRIES_FROM_RANK_LOCAL Count Tucker core and factor entries.

number_of_entries = prod(ranks) + ...
    sum(n .* ranks);

end

function info = empty_cycle_info( ...
    tol_mode, max_ranks, beta, original_residual, ...
    r_precond, solver_time, diagnostic_time, wall_time, ...
    tol, fixed_tol, ...
    relax_budget, max_relax_tol)
%EMPTY_CYCLE_INFO Return consistent fields for an accurate initial guess.

info.iterations = 0;
info.converged = true;
info.stop_reason = "initial_guess";
info.tolerance_mode = tol_mode;
info.target_tolerance = tol;
info.fixed_compression_tolerance = fixed_tol;
info.relaxation_error_budget = relax_budget;
info.maximum_relaxed_tolerance = max_relax_tol;
info.maximum_multilinear_rank = max_ranks;
info.cycle_starting_preconditioned_residual_norm = beta;
info.true_relative_residual = original_residual;
info.true_residual_iteration = 0;
info.true_preconditioned_relative_residual = r_precond;
info.true_preconditioned_residual_iteration = 0;
info.computed_preconditioned_relative_residual = zeros(0, 1);
info.preconditioned_residual_gap = zeros(0, 1);
info.preconditioned_residual_gap_iteration = zeros(0, 1);
info.eta = zeros(0, 1);
info.solution_ranks = zeros(0, numel(max_ranks));
info.solution_rank_iteration = zeros(0, 1);
info.basis_ranks = zeros(0, numel(max_ranks));
info.basis_storage_history_entries = zeros(0, 1);
info.peak_basis_storage_entries = 0;
info.H = [];
info.y = [];
info.rank_cap_active_history = false(0, 1);
info.rank_cap_active = false;
info.initial_maximum_local_recompression_error = 0;
info.operator_product_maximum_local_recompression_error = zeros(0, 1);
info.orthogonalisation_maximum_local_recompression_error = zeros(0, 1);
info.solution_maximum_local_recompression_error = zeros(0, 1);
info.stopped_for_basis_memory = false;
info.cycle_stopped_on_computed_residual = false;
info.orthogonalisation_passes = 1;
info.solver_time_history_sec = zeros(0, 1);
info.diagnostic_time_history_sec = zeros(0, 1);
info.wall_time_history_sec = zeros(0, 1);
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = diagnostic_time;
info.wall_time_sec = wall_time;
info.solution_assembly_count = 0;
info.original_true_residual_evaluation_count = 1;

end
