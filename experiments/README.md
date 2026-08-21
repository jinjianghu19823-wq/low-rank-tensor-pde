# Reproducing the thesis experiments

Run `setup('/path/to/tensor_toolbox')` from the repository root before an
experiment. Generated files are written under `experiments/raw/`,
`experiments/processed/`, and `experiments/figures/`; these outputs are not
tracked by Git.

## Section 3.6: controlled compression

Run the scripts in this order:

```matlab
run('experiments/section3_6/experiment1_generate_data.m')
run('experiments/section3_6/experiment1_compare_svd_rsvd.m')
run('experiments/section3_6/experiment2_controlled_tensor.m')
run('experiments/section3_6/plot_controlled_randomized_compression.m')
```

## Section 6.1: GMRES and sGMRES

The full configuration includes the large `N=1000` case and may require
substantial memory and time.

```matlab
run('experiments/section6_1/experiment3_gmres_sgmres_poisson_scaling.m')
run('experiments/section6_1/plot_gmres_sgmres_poisson_scaling.m')
```

## Section 6.2: accurate fast-diagonalisation preconditioner

Each protected timing case was run in a fresh MATLAB process. For every grid,
repeat, and execution position, obtain the corresponding method index from
`config.balancedOrder` and call:

```matlab
run_poisson_fast_diagonalization_fixed_iteration_scaling_case( ...
    N, methodIndex, repeatIndex, orderPosition)
```

The frozen design is returned by
`poisson_fast_diagonalization_fixed_iteration_scaling_config`. After all
timing cases, run the independent histories for `N=150`, `300`, and `400`,
then consolidate and plot:

```matlab
run_poisson_fast_diagonalization_fixed_iteration_histories(150)
run_poisson_fast_diagonalization_fixed_iteration_histories(300)
run_poisson_fast_diagonalization_fixed_iteration_histories(400)
plot_poisson_fast_diagonalization_fixed_iteration_scaling
```

## Section 6.3: weak-preconditioner four-method study

The main experiment is resumable and writes a checkpoint after each case.
It is intentionally expensive.

```matlab
run_weak_preconditioned_tucker_four_method_scaling
run_weak_preconditioned_tucker_four_method_residual_histories
plot_weak_preconditioned_tucker_four_method_scaling
```

Timing outputs exclude independent residual diagnostics and plot generation.
Do not compare generated timings across machines without rerunning every
method under the same environment.
