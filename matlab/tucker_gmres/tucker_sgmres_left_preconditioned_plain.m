function [U, info] = tucker_sgmres_left_preconditioned_plain( ...
    A, P, ...
    F0, U0, ...
    maxit, sketch, sketch_tol, ...
    ktrunc, basis_tol, ...
    solution_tol, true_residual, ...
    precond_residual, verbose, ...
    max_basis_entries, max_rank, ...
    check_it)
%TUCKER_SGMRES_LEFT_PRECONDITIONED_PLAIN Run Algorithm 5.7.

if nargin < 13
    verbose = false;
end
if nargin < 14
    max_basis_entries = Inf;
end
if nargin < 15
    max_rank = [];
end
if nargin < 16
    check_it = [];
end

if ~isa(F0, 'ttensor') || ~isa(U0, 'ttensor')
    error('F0 and U0 must be Tensor Toolbox ttensor objects.');
end
if ~isequal(size(F0), size(U0))
    error('F0 and U0 must have the same tensor dimensions.');
end
if maxit < 1 || maxit ~= floor(maxit)
    error('maxit must be a positive integer.');
end
if ktrunc < 1 || ...
        ktrunc ~= floor(ktrunc) || ...
        ktrunc > maxit
    error('ktrunc must be between 1 and maxit.');
end
if sketch_tol <= 0 || sketch_tol >= 1
    error('sketch_tol must be between 0 and 1.');
end
if basis_tol <= 0 || basis_tol >= 1
    error('basis_tol must be between 0 and 1.');
end
if solution_tol <= 0 || ...
        solution_tol >= 1
    error('solution_tol must be between 0 and 1.');
end
if ~islogical(verbose) || ~isscalar(verbose)
    error('verbose must be one logical value.');
end
if max_basis_entries <= 0
    error('max_basis_entries must be positive.');
end
if ~isequal(double(size(F0)), double(sketch.tensor_dimensions))
    error('The residual sketch dimensions do not match F0.');
end

n = double(size(F0));
d = ndims(F0);
max_ranks = normalise_tucker_rank_cap( ...
    max_rank, n);
check_it = normalise_tucker_diagnostic_iterations( ...
    check_it, maxit);
norm_f0 = norm(F0);

if norm_f0 == 0
    error('The original right hand side F0 must be nonzero.');
end

% Initialise timing and form the starting residual

wall_timer = tic;
solver_timer = tic;
timing.initial_residual_time_sec = 0;
timing.operator_term_time_sec = 0;
timing.preconditioner_term_time_sec = 0;
timing.operator_exact_sum_time_sec = 0;
timing.residual_sketch_time_sec = 0;
timing.small_least_squares_time_sec = 0;
timing.orthogonalisation_inner_product_time_sec = 0;
timing.orthogonalisation_exact_sum_time_sec = 0;
timing.basis_round_time_sec = 0;
timing.basis_normalisation_time_sec = 0;
timing.solution_assembly_time_sec = 0;

component_timer = tic;
[A_terms0, A_coeff0, initial_A_info] = A(U0);
check_unrounded_operator_info(initial_A_info);
A_terms0 = A_terms0(:);
A_coeff0 = double(A_coeff0(:));
timing.operator_term_time_sec = toc(component_timer);

res_terms = [{F0}; A_terms0];
res_coeff = [1; -A_coeff0];
component_timer = tic;
[initial_original_residual_norm, ~] = tucker_weighted_sum_norm( ...
    res_terms, res_coeff);
initial_original_residual_scale = formal_sum_scale( ...
    res_terms, res_coeff);
timing.initial_residual_time_sec = toc(component_timer);

if initial_original_residual_norm <= ...
        100 * eps * max(1, initial_original_residual_scale)
    error(['The initial residual is numerically zero. Return U0 ', ...
           'without starting the Tucker sGMRES cycle.']);
end

component_timer = tic;
[precond_terms, ~] = ...
    apply_fixed_linear_map_to_terms( ...
        res_terms, P);
timing.preconditioner_term_time_sec = toc(component_timer);
precond_coeff = res_coeff;
[beta0, ~] = tucker_weighted_sum_norm( ...
    precond_terms, precond_coeff);
initial_preconditioned_residual_scale = formal_sum_scale( ...
    precond_terms, precond_coeff);

if beta0 <= ...
        100 * eps * max(1, initial_preconditioned_residual_scale)
    error(['The fixed left preconditioner annihilated the nonzero ', ...
           'initial residual.']);
end

component_timer = tic;
g = apply_tucker_row_khatri_rao_sketch_sum( ...
    precond_terms, ...
    precond_coeff, sketch);
timing.residual_sketch_time_sec = ...
    timing.residual_sketch_time_sec + toc(component_timer);
norm_g = norm(g);

if norm_g <= 100 * eps * max(1, beta0)
    error('The starting residual sketch is numerically zero.');
end

component_timer = tic;
[r_precond] = tucker_exact_sum_from_terms( ...
    precond_terms, precond_coeff);
timing.operator_exact_sum_time_sec = ...
    timing.operator_exact_sum_time_sec + toc(component_timer);

component_timer = tic;
[R0P, initial_round_info] = tucker_round( ...
    r_precond, basis_tol, max_ranks);
timing.basis_round_time_sec = ...
    timing.basis_round_time_sec + toc(component_timer);
beta = norm(R0P);

if beta <= 1e-14 * norm_f0
    error('The rounded starting preconditioned residual is nearly zero.');
end

component_timer = tic;
V = cell(maxit, 1);
V{1} = (1 / beta) * R0P;
timing.basis_normalisation_time_sec = ...
    timing.basis_normalisation_time_sec + toc(component_timer);

% Prepare the stored basis and residual problem

C = zeros(sketch.sketch_size, maxit);
sketch_res = NaN(maxit, 1);
sketch_cond = NaN(maxit, 1);
sketch_rank = NaN(maxit, 1);
basis_ranks = NaN(maxit, d);
basis_storage = NaN(maxit, 1);
basis_ranks(1, :) = size(V{1}.core);
basis_storage(1) = tucker_storage_entries( ...
    n, basis_ranks(1, :));
cap_active = operation_cap_active(initial_round_info);
basis_limit_reached = ...
    basis_storage(1) > max_basis_entries;
breakdown = false;
stop_reason = "iteration_limit";
y = zeros(0, 1);
ninner = 0;
orthogonalisation_subtraction_count = 0;
basis_round_count = 1;
check_time = 0;
check_res = NaN(numel(check_it), 1);
check_res_it = NaN(numel(check_it), 1);
check_idx = 0;

% Build the local Arnoldi basis and sketched residual problem

for j = 1:maxit

    component_timer = tic;
    [A_terms_value, A_coeff, A_info] = ...
        A(V{j});
    check_unrounded_operator_info(A_info);
    A_terms_value = A_terms_value(:);
    A_coeff = double(A_coeff(:));
    timing.operator_term_time_sec = ...
        timing.operator_term_time_sec + toc(component_timer);

    component_timer = tic;
    [PA_terms, ~] = ...
        apply_fixed_linear_map_to_terms( ...
            A_terms_value, P);
    timing.preconditioner_term_time_sec = ...
        timing.preconditioner_term_time_sec + toc(component_timer);

    component_timer = tic;
    C(:, j) = apply_tucker_row_khatri_rao_sketch_sum( ...
        PA_terms, A_coeff, sketch);
    timing.residual_sketch_time_sec = ...
        timing.residual_sketch_time_sec + toc(component_timer);

    component_timer = tic;
    Cj = C(:, 1:j);
    y = lsqminnorm(Cj, g, sqrt(eps));
    sketch_residual = g - Cj * y;
    sketch_res(j) = norm(sketch_residual) / norm_g;
    timing.small_least_squares_time_sec = ...
        timing.small_least_squares_time_sec + toc(component_timer);

    svals = svd(Cj, 'econ');
    if isempty(svals) || svals(1) == 0
        sketch_cond(j) = Inf;
        sketch_rank(j) = 0;
    else
        numerical_threshold = max(size(Cj)) * ...
            eps(svals(1));
        sketch_rank(j) = sum(svals > numerical_threshold);
        sketch_cond(j) = ...
            svals(1) / svals(end);
    end

    if verbose
        fprintf([ ...
            'Plain Tucker sGMRES iteration %2d, ', ...
            'sketched residual %.3e\n'], ...
            j, sketch_res(j));
    end

    if any(check_it == j)
        diagnostic_timer = tic;
        U_check = U0;
        for diagnostic_basis_index = 1:j
            U_check = tucker_axpby_round( ...
                U_check, 1, V{diagnostic_basis_index}, ...
                y(diagnostic_basis_index), ...
                solution_tol, max_ranks);
        end
        check_idx = check_idx + 1;
        check_res(check_idx) = ...
            true_residual(U_check);
        check_res_it(check_idx) = j;
        check_time = check_time + ...
            toc(diagnostic_timer);
        clear U_check
    end

    if sketch_res(j) <= sketch_tol
        stop_reason = "sketched_residual";
        break
    end

    if j == maxit
        break
    end

    component_timer = tic;
    Z = tucker_exact_sum_from_terms( ...
        PA_terms, A_coeff);
    timing.operator_exact_sum_time_sec = ...
        timing.operator_exact_sum_time_sec + toc(component_timer);

    first_idx = max(1, j - ktrunc + 1);
    W = Z;
    operator_product_norm = norm(Z);

    for i = first_idx:j
        component_timer = tic;
        coefficient = innerprod(W, V{i});
        timing.orthogonalisation_inner_product_time_sec = ...
            timing.orthogonalisation_inner_product_time_sec + ...
            toc(component_timer);
        ninner = ...
            ninner + 1;

        component_timer = tic;
        W = tucker_axpby_exact(W, 1, V{i}, -coefficient);
        timing.orthogonalisation_exact_sum_time_sec = ...
            timing.orthogonalisation_exact_sum_time_sec + ...
            toc(component_timer);
        orthogonalisation_subtraction_count = ...
            orthogonalisation_subtraction_count + 1;
    end

    component_timer = tic;
    [W, round_info] = tucker_round( ...
        W, basis_tol, max_ranks);
    timing.basis_round_time_sec = ...
        timing.basis_round_time_sec + toc(component_timer);
    basis_round_count = basis_round_count + 1;
    cap_active = cap_active || operation_cap_active(round_info);

    Wnorm = norm(W);
    breakdown_threshold = max( ...
        100 * eps, basis_tol) * ...
        max(1, operator_product_norm);

    if Wnorm <= breakdown_threshold
        breakdown = true;
        stop_reason = "arnoldi_breakdown";
        break
    end

    component_timer = tic;
    V{j + 1} = (1 / Wnorm) * W;
    timing.basis_normalisation_time_sec = ...
        timing.basis_normalisation_time_sec + toc(component_timer);

    basis_ranks(j + 1, :) = size(V{j + 1}.core);
    basis_storage(j + 1) = basis_storage(j) + ...
        tucker_storage_entries( ...
            n, basis_ranks(j + 1, :));

    if basis_storage(j + 1) > max_basis_entries
        basis_limit_reached = true;
        stop_reason = "basis_memory";
        break
    end

end

niter = j;
check_res = ...
    check_res(1:check_idx);
check_res_it = ...
    check_res_it(1:check_idx);

% Assemble the rounded Tucker solution

component_timer = tic;
U = U0;
solution_rank_cap_active = false;
solution_maximum_local_error = 0;

for i = 1:niter
    [U, add_info] = tucker_axpby_round( ...
        U, 1, V{i}, y(i), solution_tol, ...
        max_ranks);
    solution_rank_cap_active = solution_rank_cap_active || ...
        operation_cap_active(add_info);
    if isfield(add_info, 'relative_error_estimate')
        solution_maximum_local_error = max( ...
            solution_maximum_local_error, ...
            add_info.relative_error_estimate);
    end
end
timing.solution_assembly_time_sec = toc(component_timer);
cap_active = cap_active || solution_rank_cap_active;
raw_solver_time_sec = toc(solver_timer);
solver_time = raw_solver_time_sec - check_time;

diagnostic_timer = tic;
true_res0 = true_residual(U0);
initial_preconditioned_relative_residual = ...
    precond_residual(U0) / ...
    beta0;
final_true_res = true_residual(U);
final_preconditioned_relative_residual = ...
    precond_residual(U) / ...
    beta0;

linear_candidate = U0;
for i = 1:niter
    linear_candidate = tucker_axpby_exact( ...
        linear_candidate, 1, V{i}, y(i));
end
linear_candidate_preconditioned_relative_residual = ...
    precond_residual(linear_candidate) / ...
    beta0;
linear_candidate_original_true_residual = ...
    true_residual(linear_candidate);

basis_gram = zeros(niter);
for row_idx = 1:niter
    for col_idx = 1:niter
        basis_gram(row_idx, col_idx) = ...
            innerprod(V{row_idx}, V{col_idx});
    end
end
basis_orthogonality_error = ...
    norm(basis_gram - eye(niter), 'fro');
independent_diagnostic_time_sec = toc(diagnostic_timer);
diagnostic_time_sec = independent_diagnostic_time_sec + ...
    check_time;
wall_time_sec = toc(wall_timer);

if ~isempty(check_res_it) && ...
        check_res_it(end) == niter
    check_res(end) = final_true_res;
else
    check_res(end + 1, 1) = final_true_res;
    check_res_it(end + 1, 1) = niter;
end

basis_rows = 1:niter;
info.iterations = niter;
info.stop_reason = stop_reason;
info.breakdown = breakdown;
info.stopped_for_basis_memory = basis_limit_reached;
info.rank_cap_active = cap_active;
info.exact_fixed_linear_sketch_before_rounding = true;
info.fixed_linear_preconditioner_rounding_performed = false;
info.sketch_size = sketch.sketch_size;
info.sketch_seed = sketch.random_seed;
info.orthogonalisation_window = ktrunc;
info.basis_compression_tolerance = basis_tol;
info.solution_compression_tolerance = solution_tol;
info.sketch_stopping_tolerance = sketch_tol;
info.cycle_starting_preconditioned_residual_norm = beta;
info.computed_sketch_relative_residual = ...
    sketch_res(basis_rows);
info.sketch_matrix_condition = ...
    sketch_cond(basis_rows);
info.sketch_matrix_rank = sketch_rank(basis_rows);
info.independent_preconditioned_relative_residual = ...
    final_preconditioned_relative_residual;
info.linear_candidate_preconditioned_relative_residual = ...
    linear_candidate_preconditioned_relative_residual;
info.linear_candidate_original_true_relative_residual = ...
    linear_candidate_original_true_residual;
info.true_relative_residual = ...
    [true_res0; check_res];
info.true_residual_iteration = ...
    [0; check_res_it];
info.initial_preconditioned_relative_residual = ...
    initial_preconditioned_relative_residual;
info.sketch_to_linear_residual_ratio = ...
    sketch_res(niter) / ...
    linear_candidate_preconditioned_relative_residual;
info.rounded_solution_residual_gap = abs( ...
    final_preconditioned_relative_residual - ...
    linear_candidate_preconditioned_relative_residual);
info.basis_ranks = basis_ranks(basis_rows, :);
info.basis_storage_history_entries = ...
    basis_storage(basis_rows);
info.peak_basis_storage_entries = ...
    max(info.basis_storage_history_entries);
info.solution_ranks = size(U.core);
info.basis_gram = basis_gram;
info.basis_orthogonality_error_fro = basis_orthogonality_error;
info.C = C(:, basis_rows);
info.y = y;
info.orthogonalisation_inner_product_count = ...
    ninner;
info.orthogonalisation_subtraction_count = ...
    orthogonalisation_subtraction_count;
info.basis_round_count = basis_round_count;
info.solution_assembly_count = 1;
info.original_true_residual_evaluation_count = ...
    numel(info.true_relative_residual);
info.solution_maximum_local_recompression_error = ...
    solution_maximum_local_error;
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = diagnostic_time_sec;
info.wall_time_sec = wall_time_sec;
timing.solver_time_sec = solver_time;
timing.diagnostic_time_sec = diagnostic_time_sec;
timing.wall_time_sec = wall_time_sec;
timing.checkpoint_diagnostic_time_sec = check_time;
timing.independent_residual_diagnostic_time_sec = ...
    independent_diagnostic_time_sec;
timing.accounted_solver_time_sec = ...
    timing.initial_residual_time_sec + ...
    timing.operator_term_time_sec + ...
    timing.preconditioner_term_time_sec + ...
    timing.operator_exact_sum_time_sec + ...
    timing.residual_sketch_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.orthogonalisation_exact_sum_time_sec + ...
    timing.basis_round_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solver_time - timing.accounted_solver_time_sec;
info.timing = timing;

end

function active = operation_cap_active(op_info)
%OPERATION_CAP_ACTIVE Read an optional rank cap diagnostic.

active = isstruct(op_info) && ...
    isfield(op_info, 'rank_cap_active') && ...
    logical(op_info.rank_cap_active);

end

function check_unrounded_operator_info(op_info)
%CHECK_UNROUNDED_OPERATOR_INFO Enforce the fixed linear operator contract.

if ~isstruct(op_info) || ...
        ~isfield(op_info, 'rounding_performed') || ...
        logical(op_info.rounding_performed)
    error(['The operator callback must return an unrounded formal sum ', ...
           'and report rounding_performed=false.']);
end

end

function [mapped_terms, op_info] = ...
    apply_fixed_linear_map_to_terms(terms, map_function)
%APPLY_FIXED_LINEAR_MAP_TO_TERMS Apply P to every exact operator term.

mapped_terms = cell(size(terms));
op_info = cell(size(terms));

for term_idx = 1:numel(terms)
    [mapped_terms{term_idx}, op_info{term_idx}] = ...
        map_function(terms{term_idx});

    if ~isstruct(op_info{term_idx}) || ...
            ~isfield(op_info{term_idx}, 'rounding_performed') || ...
            logical(op_info{term_idx}.rounding_performed)
        error(['The preconditioner callback must be fixed and linear. ', ...
               'It must report rounding_performed=false.']);
    end
end

end

function scale = formal_sum_scale(terms, coeff)
%FORMAL_SUM_SCALE Return a cancellation free scale for zero checks.

scale = 0;

for term_idx = 1:numel(terms)
    scale = scale + abs(coeff(term_idx)) * norm(terms{term_idx});
end

end

function number_of_entries = tucker_storage_entries( ...
    n, ranks)
%TUCKER_STORAGE_ENTRIES Count core and factor entries.

number_of_entries = prod(ranks) + ...
    sum(n .* ranks);

end
