# Tucker sGMRES

This repository contains the MATLAB implementation of the Tucker GMRES and
Tucker sGMRES methods used in the MSc dissertation *Low-Rank Tensor
Decomposition for the Solution of Partial Differential Equations* by Jinjiang
Hu. The included experiment drivers reproduce the Poisson studies reported in
Sections 6.2 and 6.3.

The public repository has one canonical address:

```bash
git clone https://github.com/jinjianghu19823-wq/tucker-sgmres.git
```

## Requirements

- MATLAB R2025b or a compatible recent release
- [Tensor Toolbox for MATLAB 3.8](https://gitlab.com/tensors/tensor_toolbox)

Set `TENSOR_TOOLBOX_ROOT` to the Tensor Toolbox directory. Then start MATLAB
from the repository root and run

```matlab
addpath("matlab")
add_toolboxes
run_all_tests
```

The main solver routines are in `matlab/tucker_gmres`. The Poisson operators
and preconditioners are in `matlab/poisson`.

## Reproducibility records

The cleaned, dated records for the two reported Poisson studies are stored in
[`docs/planning/experiment_logs`](docs/planning/experiment_logs):

- [four-step preconditioned GMRES scaling](docs/planning/experiment_logs/2026-07-30_poisson_fast_diagonalization_fixed_iteration_scaling.md)
- [100-product Tucker Krylov scaling](docs/planning/experiment_logs/2026-08-19_weak_preconditioned_tucker_four_method_scaling.md)

The long-cycle experiment uses residual sketch size `s=336`. This value came
from an empirical calibration, not from a theorem and not as a fixed value in
the source paper. The earlier 20-product calibration used the transfer rule
`s/(m+1)=16`, giving `s=336`. A follow-up with smaller sketches saved little
time and gave worse residual accuracy and conditioning, so `s=336` was kept
for the reported 100-product comparison. It is not claimed to be optimal.

## Main references

- A. Bucci, M. Iannacito, M. Pasha and R. Smith, *Randomized Tucker-Sketched
  GMRES*, arXiv:2608.11091, 2026.
- A. Bucci, D. Palitta and L. Robol, *Randomized Sketched TT-GMRES for Linear
  Systems with Tensor Structure*, SIAM Journal on Scientific Computing, 2025.
- Y. Nakatsukasa and J. A. Tropp, *Fast and Accurate Randomized Algorithms for
  Linear Systems and Eigenvalue Problems*, SIAM Journal on Matrix Analysis and
  Applications, 45(2), 1183-1214, 2024,
  [doi:10.1137/23M1565413](https://doi.org/10.1137/23M1565413).

Citation metadata is provided in `CITATION.cff`. The code is released under
the MIT License.
