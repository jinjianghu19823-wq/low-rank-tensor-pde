# Low-Rank Tensor Decomposition for PDEs

MATLAB code accompanying the MSc dissertation *Low-Rank Tensor
Decomposition for the Solution of Partial Differential Equations*.

This repository contains the complete code path used in the thesis: every
numbered algorithm, the supporting Tucker arithmetic and sketches, the
Poisson operators and fast-diagonalisation preconditioners, the final
experiment/configuration/plot scripts, and focused correctness tests.
Generated raw data, processed tables, and figures are intentionally excluded;
the experiment scripts recreate them locally.

## Requirements

- MATLAB R2025b or a compatible recent release.
- [Tensor Toolbox for MATLAB](https://www.tensortoolbox.org/) 3.8 for tensor
  algorithms and experiments. Matrix and vector routines use MATLAB only.

From the repository root, initialise the paths with either:

```matlab
setup('/path/to/tensor_toolbox')
```

or set `TENSOR_TOOLBOX_ROOT` before starting MATLAB and run:

```matlab
setup
```

## Repository layout

```text
algorithm/              Numbered thesis algorithms
  matrix/               Randomised SVD and generalised Nystrom
  tensor/               HOSVD, STHOSVD, randomised variants, MLN
  krylov/               GMRES, restarted GMRES, and sGMRES
  tucker/               Tucker GMRES, Tucker sGMRES, and RoundSum
operators/poisson/      Poisson actions and fast-diagonalisation preconditioners
utilities/              Tucker, sketching, plotting, generator, and diagnostic helpers
experiments/            Final scripts for Sections 3.6 and 6.1--6.3
tests/                  Focused correctness and smoke tests
docs/                   Reference list and supporting documentation
```

## Numbered algorithm map

| Thesis algorithm | Implementation |
|---|---|
| 3.1 Truncated HOSVD | `algorithm/tensor/truncated_hosvd.m` |
| 3.2 Sequentially truncated HOSVD | `algorithm/tensor/truncated_sthosvd.m` |
| 3.3 Randomised SVD | `algorithm/matrix/rsvd.m` |
| 3.4 Randomised HOSVD | `algorithm/tensor/r_hosvd.m` |
| 3.5 Randomised sequentially truncated HOSVD | `algorithm/tensor/r_sthosvd.m` |
| 3.6 Stabilised generalised Nystrom approximation | `algorithm/matrix/generalized_nystrom.m` |
| 3.7 Multilinear Nystrom approximation | `algorithm/tensor/multilinear_nystrom.m` |
| 5.1 GMRES | `algorithm/krylov/gmres_unrestarted_custom.m` |
| 5.2 Restarted GMRES | `algorithm/krylov/gmres_restarted_custom.m` |
| 5.3 sGMRES with truncated Arnoldi orthogonalisation | `algorithm/krylov/sgmres_trunc_arnoldi.m` |
| 5.4 Tucker AXPBY with recompression | `algorithm/tucker/tucker_axpby_round.m` |
| 5.5 Fixed Tucker GMRES | `algorithm/tucker/tucker_gmres_left_preconditioned_fixed.m` |
| 5.6 Relaxed Tucker GMRES | `algorithm/tucker/tucker_gmres_left_preconditioned_capped_relaxed.m` |
| 5.7 Plain Tucker sGMRES | `algorithm/tucker/tucker_sgmres_left_preconditioned_plain.m` |
| 5.8 RHOSVD Tucker sGMRES | `algorithm/tucker/tucker_sgmres_left_preconditioned_rhosvd.m` |

Algorithm 5.6 and the reported Sections 6.2 and 6.3 use the capped relaxed
implementation, which exposes the internal stopping tolerance, relaxation
budget, and maximum relaxed tolerance separately. The adjacent
`tucker_gmres_left_preconditioned_relaxed.m` implements the theoretical
special case in which one tolerance is used as both the stopping threshold
and relaxation budget and the upper cap is inactive.

## Experiments

The final experiment scripts are grouped by thesis section. See
[`experiments/README.md`](experiments/README.md) for execution order and
outputs.

- `experiments/section3_6/`: controlled matrix and tensor compression.
- `experiments/section6_1/`: GMRES, restarted GMRES, and sGMRES for 2D Poisson.
- `experiments/section6_2/`: full and Tucker GMRES with an accurate
  fast-diagonalisation preconditioner.
- `experiments/section6_3/`: four Tucker Krylov methods with a weak
  preconditioner and 100 Arnoldi products.

All reported solver residuals are independently evaluated original true
relative residuals, not only internal or sketched residual estimates.

## Tests

Run the focused release suite from MATLAB:

```matlab
setup('/path/to/tensor_toolbox')
run_all_tests
```

## Citation and licence

Citation metadata are in [`CITATION.cff`](CITATION.cff), and the dissertation
reference list is in [`docs/REFERENCES.md`](docs/REFERENCES.md). The code is
released under the MIT License.
