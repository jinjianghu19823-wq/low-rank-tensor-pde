# 100-Product Tucker Krylov Scaling Study

## Experiment ID

`2026-08-19_weak_preconditioned_tucker_four_method_scaling`

## Question

Do incomplete orthogonalisation and RHOSVD RoundSum reduce solver time and
Tucker basis ranks when fixed, relaxed, plain sketched and RHOSVD sketched
Tucker GMRES all complete exactly 100 Arnoldi products?

## Fixed setup

- Three-dimensional finite-difference Poisson problem.
- Grid sizes `N=150,200,250`.
- The same positive rank-one left preconditioner for every method.
- Exactly 100 Arnoldi products. Internal stopping is set below all observed
  residuals, so no saved run stops early.
- Five paired residual-sketch seeds for accuracy and four balanced protected
  timing positions after an unmeasured warm-up.
- Local orthogonalisation window `q=2`.
- Residual sketch size `s=336`.
- Independent original true residual at the endpoint and in separate
  diagnostic runs at iterations `0,20,40,60,80,100`.

## Why `s=336` was used

The value is an empirical calibration choice. It is not a theorem, a fixed
parameter from the source paper or an optimum claim. An earlier 20-product
calibration transferred the ratio `s/(m+1)=16`, which gives `s=336`. A later
small-sketch check saved little time and produced worse residual accuracy and
conditioning. The value `s=336` was therefore kept fixed for this study and
was not tuned separately for each grid.

## Result and boundary

Plain and RHOSVD Tucker sGMRES keep their median original true residuals within
1.3 percent of fixed Tucker GMRES on all three grids. Their fixed-work solver
times are between 8.99 and 11.84 times smaller than the fixed method times.
Their maximum Tucker basis ranks are also lower. RHOSVD is slower than plain at
`N=150`, then faster at `N=200` and `N=250`. Relaxed Tucker GMRES is fast but
less accurate because the deliberately weak preconditioner does not satisfy
the well-conditioned-operator assumption used by its practical relaxation
rule.

This is a fixed-work comparison, not a time-to-prescribed-residual experiment.
The result is limited to the stated Poisson problem, rank-one right-hand side,
weak preconditioner, sketch family, random seeds and tested grids.

## Public implementation

- Configuration:
  `matlab/experiments/weak_preconditioned_tucker_four_method_scaling_config.m`
- Shared problem builder:
  `matlab/experiments/build_weak_preconditioned_tucker_scaling_problem.m`
- Shared method driver:
  `matlab/experiments/execute_weak_preconditioned_tucker_scaling_method.m`
- Main runner:
  `matlab/experiments/run_weak_preconditioned_tucker_four_method_scaling.m`
- Plot routine:
  `matlab/experiments/plot_weak_preconditioned_tucker_four_method_scaling.m`

The RHOSVD correctness regression is
`matlab/tests/test_tucker_roundsum_rhosvd.m`.
