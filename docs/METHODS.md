# Method and code map

| Dissertation method | Main MATLAB routine | Main feature |
| --- | --- | --- |
| Fixed Tucker GMRES | `tucker_gmres_left_preconditioned_fixed` | Full Arnoldi and fixed STHOSVD tolerance |
| Relaxed Tucker GMRES | `tucker_gmres_left_preconditioned_relaxed` | Full Arnoldi and residual based tolerance |
| Plain Tucker sGMRES | `tucker_sgmres_left_preconditioned_plain` | Local orthogonalisation and row Khatri Rao residual sketch |
| RHOSVD Tucker sGMRES | `tucker_sgmres_left_preconditioned_rhosvd` | Local orthogonalisation and randomised `RoundSum` |

## Shared operations

- `run_left_preconditioned_tucker_gmres_cycle` implements the common fixed and
  relaxed Tucker GMRES cycle.
- `sthosvd_round_tensor` performs deterministic Tucker rounding.
- `tucker_roundsum_rhosvd_adaptive` performs adaptive randomised summation and
  rounding.
- `create_tucker_row_khatri_rao_sketch` and
  `apply_tucker_row_khatri_rao_sketch` implement the residual sketch without
  forming its full matrix.
- `true_residual_tucker_poisson` evaluates the original Poisson residual
  independently.

The implementations use left preconditioning. The basis therefore belongs to
the Krylov space of the preconditioned operator, while the returned correction
is added directly to the original unknown. The original true residual remains
the final accuracy check.
