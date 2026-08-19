function run_poisson_fast_diagonalization_fixed_iteration_scaling_case( ...
    N, methodIndex, repeatIndex, orderPosition)
%RUN_POISSON_FAST_DIAGONALIZATION_FIXED_ITERATION_SCALING_CASE
% Warm and measure one protected case from the frozen four-step study.
%
% Thesis/experiment notation (Section 6.2):
%   N              <->  mode size and interior points per coordinate
%   methodIndex    <->  1 full, 2 fixed Tucker, 3 capped relaxed Tucker
%   repeatIndex    <->  measured repetition in the balanced design
%   orderPosition  <->  execution position within that repetition

config = ...
    poisson_fast_diagonalization_fixed_iteration_scaling_config();

run_poisson_fast_diagonalization_equal_accuracy_scaling_case( ...
    N, methodIndex, repeatIndex, orderPosition, config);

end
