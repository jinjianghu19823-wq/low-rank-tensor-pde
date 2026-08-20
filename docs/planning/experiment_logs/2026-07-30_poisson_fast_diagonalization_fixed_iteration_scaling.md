# Four-Step Preconditioned GMRES Scaling Study

## Experiment ID

`2026-07-30_poisson_fast_diagonalization_fixed_iteration_scaling`

## Question

After the same four preconditioned Arnoldi products, how do the original true
residual, solver time and maximum Tucker basis rank change with grid size for
full GMRES, fixed Tucker GMRES and relaxed Tucker GMRES?

## Fixed setup

- Three-dimensional finite-difference Poisson problem.
- Grid sizes `N=150,200,250,300,350,400`.
- Smooth rank-one Gaussian right-hand side and zero initial guess.
- Fast-diagonalisation left preconditioner selected by a complete spectral
  defect test.
- Four Arnoldi products for every method.
- Fixed Tucker tolerance `1e-10`.
- Relaxation error budget `1e-10` and upper tolerance `1e-2`.
- Three balanced measured repetitions per method and grid, each after an
  unmeasured warm-up.
- Original true residual evaluated independently of the computed Hessenberg
  residual.

## Result and boundary

Full GMRES and fixed Tucker GMRES reach the original relative residual
reference `1e-6` at every grid. Relaxed Tucker GMRES reaches it at five grids.
At `N=350`, all three relaxed runs finish at `1.107e-6`; this miss is retained
without retuning. Relaxed Tucker GMRES has the lowest four-step solver time and
lowest Tucker ranks in the tested cases, but the ratios are fixed-work cost
ratios rather than time-to-equal-accuracy speedups.

The experiment is limited to one smooth right-hand side, one preconditioner
selection rule and four unrestarted Arnoldi products. Independent residual
diagnostics and plotting are excluded from solver time.

## Public implementation

- Configuration:
  `matlab/experiments/poisson_fast_diagonalization_fixed_iteration_scaling_config.m`
- Case runner:
  `matlab/experiments/run_poisson_fast_diagonalization_fixed_iteration_scaling_case.m`
- Residual histories:
  `matlab/experiments/run_poisson_fast_diagonalization_fixed_iteration_histories.m`
- Plot routine:
  `matlab/experiments/plot_poisson_fast_diagonalization_fixed_iteration_scaling.m`

The focused preconditioner test is
`matlab/tests/test_poisson_fast_diagonalization_tucker_preconditioner.m`.
