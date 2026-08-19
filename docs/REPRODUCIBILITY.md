# Reproducibility guide

## Environment

The reported results used MATLAB R2025b Update 5, Tensor Toolbox 3.8 at commit
`531c5fe84aee270d0a6793e142682556ef9b2fc5`, macOS 26.5.2 and double
precision. The local machine used an Apple M5 processor with 10 cores and
16 GB of unified memory.

Set `TENSOR_TOOLBOX_ROOT` to a Tensor Toolbox 3.8 checkout. Then start MATLAB
from the repository root and run

```matlab
addpath("matlab")
add_toolboxes
```

## Checked processed evidence

The repository includes the processed CSV tables used to create every public
figure. Plotting scripts read these tables and do not rerun a solver.

| Comparison | Plot script | Main output prefix |
| --- | --- | --- |
| Controlled randomised compression | `plot_controlled_randomized_compression` | `controlled_*_randomized_compression` |
| Vector GMRES and sGMRES | `plot_gmres_sgmres_poisson_scaling` | `gmres_sgmres_poisson_*` |
| Four step Tucker GMRES | `plot_poisson_fast_diagonalization_fixed_iteration_scaling` | `poisson_fast_diagonalization_fixed_iteration_scaling_*` |
| Long Tucker sGMRES cycle | `plot_weak_preconditioned_tucker_four_method_scaling` | `weak_preconditioned_tucker_four_method_scaling_*` |

## Complete experiment runs

The controlled compression and vector GMRES comparisons can be run directly.

```matlab
experiment1_generate_data
experiment1_compare_svd_rsvd
experiment2_controlled_tensor
experiment3_gmres_sgmres_poisson_scaling
```

The long Tucker cycle comparison is started with

```matlab
run_weak_preconditioned_tucker_four_method_scaling
```

It uses mode sizes 150, 200 and 250, exactly 100 Arnoldi products, five
accuracy seeds for the randomised methods and four protected timing blocks.
This experiment can take several hours. It writes restartable checkpoints to
`experiments/raw`, which is intentionally excluded from version control.

The four step preconditioned Tucker GMRES comparison uses independent cases.
For a selected grid size, repetition and method, call
`run_poisson_fast_diagonalization_fixed_iteration_scaling_case` with the
arguments documented in that file. The separate residual histories are built
by `run_poisson_fast_diagonalization_fixed_iteration_histories`.

## Timing policy

Solver timers include the initial residual, random map construction, operator
products, inner products, small least squares problems, Tucker rounding and
final solution assembly. They exclude preconditioner construction,
independent residual histories, plotting and table processing. Each method is
warmed before a protected timing run, and execution order is rotated.

Timing values should not be expected to reproduce exactly on a different
machine. Residuals, ranks, fixed parameters and safety checks are the primary
reproducibility evidence.

## Data policy

Processed tables are included because they are small and sufficient to audit
the reported values and rebuild the figures. Large raw MATLAB checkpoints are
not included. Copyrighted papers and the dissertation source are also not part
of this software repository.
