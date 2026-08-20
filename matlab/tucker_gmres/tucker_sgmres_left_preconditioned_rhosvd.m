function [U, info] = tucker_sgmres_left_preconditioned_rhosvd( ...
    A, P, ...
    F0, U0, maxit, sketch, ...
    sketch_tol, ktrunc, ...
    basis_opts, solution_opts, ...
    true_residual, precond_residual, ...
    verbose, max_basis_entries, check_it)
%TUCKER_SGMRES_LEFT_PRECONDITIONED_RHOSVD Run Algorithm 5.8.

if nargin < 13
    verbose = false;
end
if nargin < 14
    max_basis_entries = Inf;
end
if nargin < 15
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
if ~islogical(verbose) || ~isscalar(verbose)
    error('verbose must be one logical value.');
end
if max_basis_entries <= 0
    error('max_basis_entries must be positive.');
end

n = double(size(F0));
d = ndims(F0);
norm_f0 = norm(F0);
check_it = normalise_tucker_diagnostic_iterations( ...
    check_it, maxit);

if norm_f0 == 0
    error('The original right hand side F0 must be nonzero.');
end
if ~isequal(n, double(sketch.tensor_dimensions))
    error('The residual sketch dimensions do not match F0.');
end

basis_opts = check_roundsum_settings( ...
    basis_opts, n, 'basis_round_sum_settings');
solution_opts = check_roundsum_settings( ...
    solution_opts, n, ...
    'solution_round_sum_settings');
paper_order = basis_opts.algorithm_variant == "paper_order";

if basis_opts.random_seed == sketch.random_seed || ...
        solution_opts.random_seed == sketch.random_seed
    error(['The residual sketch and RoundSum maps must use ', ...
           'independent random seeds.']);
end

% Form the exact initial residual as a formal Tucker sum

wall_timer = tic;
solver_timer = tic;
timing.initial_residual_time_sec = 0;
timing.operator_term_time_sec = 0;
timing.preconditioner_term_time_sec = 0;
timing.residual_sketch_time_sec = 0;
timing.roundsum_time_sec = 0;
timing.small_least_squares_time_sec = 0;
timing.orthogonalisation_inner_product_time_sec = 0;
timing.basis_normalisation_time_sec = 0;
timing.solution_assembly_time_sec = 0;
timing.roundsum_error_diagnostic_time_sec = 0;

component_timer = tic;
[A_terms0, A_coeff0, initial_A_info] = A(U0);
check_unrounded_operator_info(initial_A_info);
A_terms0 = A_terms0(:);
A_coeff0 = double(A_coeff0(:));
timing.operator_term_time_sec = toc(component_timer);

res_terms = [{F0}; A_terms0];
res_coeff = ...
    [1; -A_coeff0];
[initial_original_residual_norm, ~] = tucker_weighted_sum_norm( ...
    res_terms, res_coeff);
original_residual_scale = formal_sum_scale( ...
    res_terms, res_coeff);

if initial_original_residual_norm <= ...
        100 * eps * max(1, original_residual_scale)
    U = U0;
    info = zero_initial_residual_info( ...
        U0, true_residual, sketch, ...
        basis_opts, solution_opts, wall_timer, solver_timer);
    return
end

component_timer = tic;
[precond_terms, initial_preconditioner_info] = ...
    apply_fixed_linear_map_to_terms( ...
    res_terms, P);
timing.preconditioner_term_time_sec = toc(component_timer);
precond_coeff = res_coeff;

[beta0, ~] = tucker_weighted_sum_norm( ...
    precond_terms, precond_coeff);
preconditioned_residual_scale = formal_sum_scale( ...
    precond_terms, precond_coeff);

if beta0 <= ...
        100 * eps * max(1, preconditioned_residual_scale)
    error(['The fixed left preconditioner annihilated the nonzero ', ...
           'initial residual.']);
end

% Create the first basis tensor with a fresh RoundSum map

basis_round_calls = 1;
initial_round_seed = basis_opts.random_seed;
[R0P, initial_round_info] = apply_configured_roundsum( ...
    precond_terms, precond_coeff, ...
    basis_opts, initial_round_seed);
timing.roundsum_time_sec = initial_round_info.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    initial_round_info.error_diagnostic_time_sec;
beta = norm(R0P);

if beta <= max(100 * eps, basis_opts.compression_tolerance) * ...
        max(1, beta0)
    error(['RoundSum produced a numerically zero first basis tensor ', ...
           'from a nonzero initial residual.']);
end

component_timer = tic;
V = cell(maxit, 1);
V{1} = (1 / beta) * R0P;
timing.basis_normalisation_time_sec = toc(component_timer);

component_timer = tic;
if paper_order
    g = apply_tucker_row_khatri_rao_sketch(R0P, sketch);
else
    g = apply_tucker_row_khatri_rao_sketch_sum( ...
        precond_terms, ...
        precond_coeff, sketch);
end
timing.residual_sketch_time_sec = toc(component_timer);
norm_g = norm(g);

if norm_g <= 100 * eps * max(1, beta0)
    error(['The residual sketch g is numerically zero although the ', ...
           'preconditioned residual is nonzero. Draw a new sketch.']);
end

% Build the local basis and the exact sketched residual problem

C = zeros(sketch.sketch_size, maxit);
sketch_res = NaN(maxit, 1);
sketch_cond = NaN(maxit, 1);
sketch_rank = NaN(maxit, 1);
basis_ranks = NaN(maxit, d);
basis_storage = NaN(maxit, 1);
basis_ranks(1, :) = size(V{1}.core);
basis_storage(1) = tucker_storage_entries( ...
    n, basis_ranks(1, :));

round_seeds = NaN(1 + 2 * maxit, 1);
round_seeds(1) = initial_round_seed;
round_info = cell(size(round_seeds));
round_info{1} = initial_round_info;
round_idx = 1;

stop_reason = "iteration_limit";
breakdown = false;
basis_limit_reached = false;
y = zeros(0, 1);
ninner = 0;
exact_sketch_before_round_sum = ~paper_order;
rounded_operator_sketch_after_round_sum = paper_order;
check_time = 0;
check_res = NaN(numel(check_it), 1);
check_res_it = NaN(numel(check_it), 1);
check_idx = 0;

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

    if paper_order
        basis_round_calls = basis_round_calls + 1;
        operator_round_seed = basis_opts.random_seed + ...
            basis_round_calls - 1;
        [Zhat, A_round_info] = apply_configured_roundsum( ...
            PA_terms, A_coeff, ...
            basis_opts, operator_round_seed);
        timing.roundsum_time_sec = timing.roundsum_time_sec + ...
            A_round_info.algorithm_time_sec;
        timing.roundsum_error_diagnostic_time_sec = ...
            timing.roundsum_error_diagnostic_time_sec + ...
            A_round_info.error_diagnostic_time_sec;
        round_idx = round_idx + 1;
        round_seeds(round_idx) = operator_round_seed;
        round_info{round_idx} = A_round_info;

        component_timer = tic;
        C(:, j) = apply_tucker_row_khatri_rao_sketch( ...
            Zhat, sketch);
    else
        component_timer = tic;
        C(:, j) = apply_tucker_row_khatri_rao_sketch_sum( ...
            PA_terms, A_coeff, ...
            sketch);
    end
    timing.residual_sketch_time_sec = ...
        timing.residual_sketch_time_sec + toc(component_timer);

    component_timer = tic;
    Cj = C(:, 1:j);
    y = lsqminnorm(Cj, g, sqrt(eps));
    sketch_residual = g - Cj * y;
    sketch_res(j) = norm(sketch_residual) / norm_g;
    timing.small_least_squares_time_sec = ...
        timing.small_least_squares_time_sec + toc(component_timer);

    [sketch_rank(j), sketch_cond(j)] = ...
        small_matrix_diagnostics(Cj);

    if verbose
        fprintf([ ...
            'RHOSVD Tucker sGMRES iteration %2d, ', ...
            'sketched residual %.3e\n'], ...
            j, sketch_res(j));
    end

    if any(check_it == j)
        diagnostic_timer = tic;
        diagnostic_terms = [{U0}; V(1:j)];
        diagnostic_coefficients = [1; y(1:j)];
        [U_check, ~] = apply_configured_roundsum( ...
            diagnostic_terms, diagnostic_coefficients, ...
            solution_opts, solution_opts.random_seed);
        check_idx = check_idx + 1;
        check_res(check_idx) = ...
            true_residual(U_check);
        check_res_it(check_idx) = j;
        check_time = check_time + ...
            toc(diagnostic_timer);
        clear U_check diagnostic_terms diagnostic_coefficients
    end

    if sketch_res(j) <= sketch_tol
        stop_reason = "sketched_residual";
        break
    end

    if j == maxit
        break
    end

    if ~paper_order
        basis_round_calls = basis_round_calls + 1;
        operator_round_seed = basis_opts.random_seed + ...
            basis_round_calls - 1;
        [Zhat, A_round_info] = apply_configured_roundsum( ...
            PA_terms, A_coeff, ...
            basis_opts, operator_round_seed);
        timing.roundsum_time_sec = timing.roundsum_time_sec + ...
            A_round_info.algorithm_time_sec;
        timing.roundsum_error_diagnostic_time_sec = ...
            timing.roundsum_error_diagnostic_time_sec + ...
            A_round_info.error_diagnostic_time_sec;
        round_idx = round_idx + 1;
        round_seeds(round_idx) = operator_round_seed;
        round_info{round_idx} = A_round_info;
    end

    first_idx = max(1, j - ktrunc + 1);
    local_coefficients = zeros(j - first_idx + 1, 1);

    for i = first_idx:j
        component_timer = tic;
        coefficient = innerprod(Zhat, V{i});

        if ~paper_order
            for earlier_index = first_idx:i - 1
                earlier_local_index = earlier_index - first_idx + 1;
                coefficient = coefficient - ...
                    local_coefficients(earlier_local_index) * ...
                    innerprod(V{earlier_index}, V{i});
                ninner = ...
                    ninner + 1;
            end
        end

        local_coefficients(i - first_idx + 1) = coefficient;
        ninner = ...
            ninner + 1;
        timing.orthogonalisation_inner_product_time_sec = ...
            timing.orthogonalisation_inner_product_time_sec + ...
            toc(component_timer);
    end

    Wterms = [{Zhat}; V(first_idx:j)];
    Wcoefficients = [1; -local_coefficients];
    basis_round_calls = basis_round_calls + 1;
    basis_round_seed = basis_opts.random_seed + ...
        basis_round_calls - 1;
    [Wtilde, one_basis_round_info] = apply_configured_roundsum( ...
        Wterms, Wcoefficients, basis_opts, basis_round_seed);
    timing.roundsum_time_sec = timing.roundsum_time_sec + ...
        one_basis_round_info.algorithm_time_sec;
    timing.roundsum_error_diagnostic_time_sec = ...
        timing.roundsum_error_diagnostic_time_sec + ...
        one_basis_round_info.error_diagnostic_time_sec;
    round_idx = round_idx + 1;
    round_seeds(round_idx) = basis_round_seed;
    round_info{round_idx} = one_basis_round_info;

    Wnorm = norm(Wtilde);
    breakdown_threshold = max( ...
        100 * eps, basis_opts.compression_tolerance) * ...
        max(1, norm(Zhat));

    if Wnorm <= breakdown_threshold
        breakdown = true;
        stop_reason = "arnoldi_breakdown";
        break
    end

    component_timer = tic;
    V{j + 1} = (1 / Wnorm) * Wtilde;
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
round_seeds = round_seeds(1:round_idx);
round_info = round_info(1:round_idx);

% Form the final solution with one independent RoundSum call

solution_terms = [{U0}; V(1:niter)];
solution_coefficients = [1; y(1:niter)];
solution_round_seed = solution_opts.random_seed;

if any(round_seeds == solution_round_seed)
    error(['The solution RoundSum seed must differ from every basis ', ...
           'RoundSum seed used in this cycle.']);
end

[U, solution_round_info] = apply_configured_roundsum( ...
    solution_terms, solution_coefficients, solution_opts, ...
    solution_round_seed);
timing.solution_assembly_time_sec = ...
    solution_round_info.algorithm_time_sec;
timing.roundsum_error_diagnostic_time_sec = ...
    timing.roundsum_error_diagnostic_time_sec + ...
    solution_round_info.error_diagnostic_time_sec;
raw_solver_time_sec = toc(solver_timer);
solver_time = raw_solver_time_sec - ...
    timing.roundsum_error_diagnostic_time_sec - ...
    check_time;

% Evaluate the two independent residual diagnostics

diagnostic_timer = tic;
true_res0 = true_residual(U0);
final_true_res = true_residual(U);
initial_preconditioned_relative_residual = ...
    precond_residual(U0) / ...
    beta0;
final_preconditioned_relative_residual = ...
    precond_residual(U) / ...
    beta0;

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
    timing.roundsum_error_diagnostic_time_sec + ...
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
info.sketch_size = sketch.sketch_size;
info.sketch_seed = sketch.random_seed;
info.orthogonalisation_window = ktrunc;
info.computed_sketch_relative_residual = ...
    sketch_res(basis_rows);
info.sketch_matrix_condition = ...
    sketch_cond(basis_rows);
info.sketch_matrix_rank = sketch_rank(basis_rows);
info.independent_preconditioned_relative_residual = ...
    final_preconditioned_relative_residual;
info.true_relative_residual = ...
    [true_res0; check_res];
info.true_residual_iteration = [0; check_res_it];
info.initial_preconditioned_relative_residual = ...
    initial_preconditioned_relative_residual;
info.initial_original_residual_norm = initial_original_residual_norm;
info.initial_preconditioned_residual_norm = ...
    beta0;
info.basis_ranks = basis_ranks(basis_rows, :);
info.basis_storage_history_entries = ...
    basis_storage(basis_rows);
info.solution_ranks = size(U.core);
info.basis_gram = basis_gram;
info.basis_orthogonality_error_fro = basis_orthogonality_error;
info.C = C(:, basis_rows);
info.g = g;
info.y = y(1:niter);
info.orthogonalisation_inner_product_count = ...
    ninner;
info.exact_fixed_linear_sketch_before_roundsum = ...
    exact_sketch_before_round_sum;
info.rounded_operator_sketch_after_roundsum = ...
    rounded_operator_sketch_after_round_sum;
info.fixed_linear_preconditioner_rounding_performed = false;
info.local_mgs_is_batch_formula_adaptation = ~paper_order;
info.batch_local_projection_coefficients = paper_order;
info.algorithm_variant = basis_opts.algorithm_variant;
info.basis_roundsum_settings = basis_opts;
info.solution_roundsum_settings = solution_opts;
info.basis_roundsum_seeds = round_seeds;
info.basis_roundsum_info = round_info;
info.solution_roundsum_seed = solution_round_seed;
info.solution_roundsum_info = solution_round_info;
info.original_true_residual_evaluation_count = ...
    numel(info.true_relative_residual);
info.initial_operator_info = initial_A_info;
info.initial_preconditioner_info = initial_preconditioner_info;
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = diagnostic_time_sec;
info.wall_time_sec = wall_time_sec;
timing.solver_time_sec = solver_time;
timing.diagnostic_time_sec = diagnostic_time_sec;
timing.wall_time_sec = wall_time_sec;
timing.independent_residual_diagnostic_time_sec = ...
    independent_diagnostic_time_sec;
timing.checkpoint_diagnostic_time_sec = check_time;
timing.accounted_diagnostic_time_sec = ...
    timing.independent_residual_diagnostic_time_sec + ...
    timing.roundsum_error_diagnostic_time_sec + ...
    timing.checkpoint_diagnostic_time_sec;
timing.unclassified_diagnostic_time_sec = ...
    diagnostic_time_sec - timing.accounted_diagnostic_time_sec;
timing.accounted_solver_time_sec = ...
    timing.initial_residual_time_sec + ...
    timing.operator_term_time_sec + ...
    timing.preconditioner_term_time_sec + ...
    timing.residual_sketch_time_sec + ...
    timing.roundsum_time_sec + ...
    timing.small_least_squares_time_sec + ...
    timing.orthogonalisation_inner_product_time_sec + ...
    timing.basis_normalisation_time_sec + ...
    timing.solution_assembly_time_sec;
timing.unclassified_solver_time_sec = ...
    solver_time - timing.accounted_solver_time_sec;
info.timing = timing;

end

function [Z, info] = apply_configured_roundsum( ...
    terms, coeff, settings, seed)
%APPLY_CONFIGURED_ROUNDSUM Apply fixed or adaptive RHOSVD RoundSum.

if settings.adaptive_rank_selection
    [Z, info] = tucker_roundsum_rhosvd_adaptive( ...
        terms, coeff, settings.oversampling, ...
        settings.compression_tolerance, settings.maximum_rank, ...
        seed, settings.measure_error_diagnostics);
else
    [Z, info] = tucker_roundsum_rhosvd( ...
        terms, coeff, settings.range_sketch_sizes, ...
        settings.compression_tolerance, settings.maximum_ranks, ...
        seed, settings.measure_error_diagnostics);
    info.adaptive_rank_selection = false;
end

end

function settings = check_roundsum_settings( ...
    settings, n, arg_name)
%CHECK_ROUNDSUM_SETTINGS Validate one basis or solution setting group.

if ~isstruct(settings)
    error('%s must be a structure.', arg_name);
end

required_fields = {'compression_tolerance', 'random_seed'};
for field_idx = 1:numel(required_fields)
    if ~isfield(settings, required_fields{field_idx})
        error('%s is missing field %s.', ...
            arg_name, required_fields{field_idx});
    end
end

if ~isfield(settings, 'measure_error_diagnostics')
    settings.measure_error_diagnostics = true;
end
if ~isfield(settings, 'algorithm_variant')
    settings.algorithm_variant = "paper_order";
end
if ~isfield(settings, 'adaptive_rank_selection')
    settings.adaptive_rank_selection = false;
end

if ~islogical(settings.measure_error_diagnostics) || ...
        ~isscalar(settings.measure_error_diagnostics)
    error('%s measure_error_diagnostics must be one logical value.', ...
        arg_name);
end
if ~islogical(settings.adaptive_rank_selection) || ...
        ~isscalar(settings.adaptive_rank_selection)
    error('%s adaptive_rank_selection must be one logical value.', ...
        arg_name);
end

settings.algorithm_variant = string(settings.algorithm_variant);
valid_variants = ["exact_sketch_before_roundsum", "paper_order"];
if ~isscalar(settings.algorithm_variant) || ...
        ~any(settings.algorithm_variant == valid_variants)
    error('%s contains an invalid algorithm_variant.', arg_name);
end

d = numel(n);

if settings.adaptive_rank_selection
    adaptive_fields = {'oversampling', 'maximum_rank'};
    for field_idx = 1:numel(adaptive_fields)
        if ~isfield(settings, adaptive_fields{field_idx})
            error('%s is missing field %s.', ...
                arg_name, adaptive_fields{field_idx});
        end
    end

    if ~isscalar(settings.oversampling) || ...
            settings.oversampling < 0 || ...
            settings.oversampling ~= floor(settings.oversampling)
        error('%s oversampling must be a nonnegative integer.', ...
            arg_name);
    end
    if ~isscalar(settings.maximum_rank) || ...
            settings.maximum_rank < 1 || ...
            settings.maximum_rank ~= floor(settings.maximum_rank) || ...
            settings.maximum_rank > min(n)
        error('%s maximum_rank is invalid.', arg_name);
    end
    settings.range_sketch_sizes = [];
    settings.maximum_ranks = [];
else
    fixed_fields = {'range_sketch_sizes', 'maximum_ranks'};
    for field_idx = 1:numel(fixed_fields)
        if ~isfield(settings, fixed_fields{field_idx})
            error('%s is missing field %s.', ...
                arg_name, fixed_fields{field_idx});
        end
    end

    settings.range_sketch_sizes = ...
        double(settings.range_sketch_sizes(:).');

    if isscalar(settings.range_sketch_sizes)
        settings.range_sketch_sizes = repmat( ...
            settings.range_sketch_sizes, 1, d);
    elseif numel(settings.range_sketch_sizes) ~= d
        error('%s range_sketch_sizes must have one value per mode.', ...
            arg_name);
    end

    if any(settings.range_sketch_sizes < 1) || ...
            any(settings.range_sketch_sizes ~= ...
            floor(settings.range_sketch_sizes)) || ...
            any(settings.range_sketch_sizes > n)
        error('%s contains an invalid range sketch size.', arg_name);
    end

    if isempty(settings.maximum_ranks)
        settings.maximum_ranks = settings.range_sketch_sizes;
    else
        settings.maximum_ranks = normalise_tucker_rank_cap( ...
            settings.maximum_ranks, n);
    end

    if any(settings.maximum_ranks > settings.range_sketch_sizes)
        error('%s must satisfy R_n <= k_n in every mode.', arg_name);
    end
end

if ~isscalar(settings.compression_tolerance) || ...
        settings.compression_tolerance <= 0 || ...
        settings.compression_tolerance >= 1
    error('%s compression_tolerance must be between 0 and 1.', ...
        arg_name);
end

if ~isscalar(settings.random_seed) || ...
        ~isfinite(settings.random_seed) || ...
        settings.random_seed < 0 || ...
        settings.random_seed ~= floor(settings.random_seed)
    error('%s random_seed must be a nonnegative integer.', arg_name);
end

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
%APPLY_FIXED_LINEAR_MAP_TO_TERMS Apply P to each exact operator term.

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
%FORMAL_SUM_SCALE A cancellation-free norm scale for zero checks.

scale = 0;

for term_idx = 1:numel(terms)
    scale = scale + abs(coeff(term_idx)) * norm(terms{term_idx});
end

end

function [numerical_rank, condition_estimate] = ...
    small_matrix_diagnostics(matrix)
%SMALL_MATRIX_DIAGNOSTICS Rank and condition of the residual matrix.

svals = svd(matrix, 'econ');

if isempty(svals) || svals(1) == 0
    numerical_rank = 0;
    condition_estimate = Inf;
else
    threshold = max(size(matrix)) * eps(svals(1));
    numerical_rank = sum(svals > threshold);
    if svals(end) <= threshold
        condition_estimate = Inf;
    else
        condition_estimate = svals(1) / svals(end);
    end
end

end

function number_of_entries = tucker_storage_entries( ...
    n, ranks)
%TUCKER_STORAGE_ENTRIES Count core and factor entries.

number_of_entries = prod(ranks) + ...
    sum(n .* ranks);

end

function info = zero_initial_residual_info( ...
    U0, true_residual, sketch, ...
    basis_opts, solution_opts, wall_timer, solver_timer)
%ZERO_INITIAL_RESIDUAL_INFO Return safely without a Krylov division.

solver_time = toc(solver_timer);
true_residual = true_residual(U0);

info.iterations = 0;
info.stop_reason = "zero_initial_residual";
info.breakdown = false;
info.stopped_for_basis_memory = false;
info.sketch_size = sketch.sketch_size;
info.sketch_seed = sketch.random_seed;
info.computed_sketch_relative_residual = zeros(0, 1);
info.independent_preconditioned_relative_residual = 0;
info.true_relative_residual = true_residual;
info.true_residual_iteration = 0;
info.initial_original_residual_norm = 0;
info.basis_ranks = zeros(0, ndims(U0));
info.solution_ranks = size(U0.core);
info.C = zeros(sketch.sketch_size, 0);
info.g = zeros(sketch.sketch_size, 1);
info.y = zeros(0, 1);
info.exact_fixed_linear_sketch_before_roundsum = true;
info.fixed_linear_preconditioner_rounding_performed = false;
info.basis_roundsum_settings = basis_opts;
info.solution_roundsum_settings = solution_opts;
info.basis_roundsum_seeds = zeros(0, 1);
info.solution_roundsum_seed = NaN;
info.solver_time_sec = solver_time;
info.diagnostic_time_sec = 0;
info.wall_time_sec = toc(wall_timer);

end
