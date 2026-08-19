# MATLAB source guide

Run `add_toolboxes` before calling the solvers or experiments. The folder
contains the following groups.

- `krylov_methods` contains vector GMRES and sGMRES routines.
- `tucker_gmres` contains Tucker addition, rounding, GMRES, sGMRES and
  RHOSVD `RoundSum` routines.
- `poisson` contains finite difference Poisson actions and left
  preconditioners.
- `tensor_methods` and `matrix_methods` contain the compression routines used
  in the controlled examples.
- `experiments` contains final experiment configurations, run scripts and
  plot only scripts.
- `tests` contains small correctness checks rather than timing experiments.

The solvers distinguish three residual quantities where applicable.

- The computed residual comes from the small Hessenberg problem.
- The sketched residual comes from the sketched least squares problem.
- The original true residual is evaluated independently from the original
  operator and is the final accuracy measure.

Diagnostic residual histories are evaluated outside protected solver timers.
