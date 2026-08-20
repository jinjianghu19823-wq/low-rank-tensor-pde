function run_poisson_fast_diagonalization_fixed_iteration_scaling_case( ...
    N, method_index, repeat_index, order_position)
%RUN_POISSON_FAST_DIAGONALIZATION_FIXED_ITERATION_SCALING_CASE Warm and measure one protected case from the frozen four-step study.

config = ...
    poisson_fast_diagonalization_fixed_iteration_scaling_config();

run_poisson_fast_diagonalization_equal_accuracy_scaling_case( ...
    N, method_index, repeat_index, order_position, config);

end
