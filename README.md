# Tucker sGMRES

MATLAB implementations and reproducibility files for low rank Tucker GMRES
and sketched Tucker GMRES methods. The code accompanies the MSc dissertation
*Low Rank Tensor Decomposition for the Solution of Partial Differential
Equations* by Jinjiang Hu.

The repository follows the compact organisation of the
[TT sGMRES reference implementation](https://github.com/numpi/tt-sgmres).
It contains the solver routines, focused correctness tests and the scripts and
processed tables used for the reported numerical comparisons.

## Implemented methods

- fixed tolerance left preconditioned Tucker GMRES
- relaxed left preconditioned Tucker GMRES
- plain left preconditioned Tucker sGMRES
- RHOSVD Tucker sGMRES with randomised `RoundSum`
- vector GMRES, restarted GMRES and sGMRES reference routines
- deterministic and randomised Tucker compression routines

The Tucker methods always report the original true residual separately from
the computed or sketched residual. This distinction is important because
Tucker rounding perturbs the Arnoldi relation.

## Requirements

- MATLAB R2025b or a compatible recent release
- [Tensor Toolbox for MATLAB 3.8](https://gitlab.com/tensors/tensor_toolbox)

The reported experiments used MATLAB R2025b in double precision on an Apple
M5 processor with 16 GB of unified memory. Solver timings will depend on the
machine and MATLAB version.

## Installation

Clone this repository and Tensor Toolbox. Then define the toolbox location
before starting MATLAB.

```bash
git clone https://github.com/jinjianghu19823-wq/tucker-sgmres.git
git clone --branch v3.8 https://gitlab.com/tensors/tensor_toolbox.git
export TENSOR_TOOLBOX_ROOT=/absolute/path/to/tensor_toolbox
```

In MATLAB, change to the repository root and run

```matlab
addpath("matlab")
add_toolboxes
```

## Quick correctness check

```matlab
run_all_tests
```

The focused Tucker sGMRES checks can also be run separately.

```matlab
test_tucker_sgmres_foundation
test_tucker_roundsum_rhosvd
test_poisson_preconditioned_tucker_gmres
```

## Reproduce the figures

The processed tables are included, so the figures can be rebuilt without
rerunning the expensive solvers.

```matlab
plot_controlled_randomized_compression
plot_gmres_sgmres_poisson_scaling
plot_poisson_fast_diagonalization_fixed_iteration_scaling
plot_weak_preconditioned_tucker_four_method_scaling
```

The generated PDF, PNG and MATLAB figure files are written to
`experiments/figures`.

The complete solver runs are larger and should be started separately. Their
fixed settings are stored in configuration functions.

```matlab
experiment3_gmres_sgmres_poisson_scaling
run_weak_preconditioned_tucker_four_method_scaling
```

The four step preconditioned Tucker GMRES study is divided into independent
grid and method cases so that a failed large case does not remove earlier
results. See [the reproducibility guide](docs/REPRODUCIBILITY.md) for the
commands and memory notes.

## Repository layout

```text
matlab/
  krylov_methods/   vector GMRES and sGMRES routines
  tucker_gmres/     Tucker GMRES, Tucker sGMRES and RoundSum
  poisson/          tensor structured Poisson operators and preconditioners
  tensor_methods/   deterministic and randomised Tucker compression
  experiments/      fixed configurations, solver drivers and plot scripts
  tests/            focused correctness tests
experiments/
  processed/        checked numerical tables
  figures/          figures rebuilt from the checked tables
docs/               method map and reproduction instructions
```

## Main references

- A. Bucci, M. Iannacito, M. Pasha and R. Smith, *Randomized Tucker-Sketched
  GMRES*, arXiv:2608.11091, 2026.
- A. Bucci, D. Palitta and L. Robol, *Randomized Sketched TT-GMRES for Linear
  Systems with Tensor Structure*, SIAM Journal on Scientific Computing, 2025.
- Y. Nakatsukasa and J. A. Tropp, *Fast and Accurate Randomized Algorithms for
  Linear Systems and Eigenvalue Problems*, SIAM Journal on Matrix Analysis and
  Applications, 2022.

## Citation

Please use the metadata in [`CITATION.cff`](CITATION.cff) when citing this
software. The dissertation citation can be added after the University of
Edinburgh record becomes available.

## License

The original code in this repository is available under the MIT License. The
Tensor Toolbox dependency has its own licence and is not included here.
