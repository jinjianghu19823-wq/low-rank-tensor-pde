function config = poisson_fast_diagonalization_fixed_iteration_scaling_config()
%POISSON_FAST_DIAGONALIZATION_FIXED_ITERATION_SCALING_CONFIG Frozen four-step comparison with an original-residual reference of 1e-6.

config.experimentId = ...
    "2026-07-30_poisson_fast_diagonalization_fixed_iteration_scaling";
config.outputPrefix = ...
    "poisson_fast_diagonalization_fixed_iteration_scaling";

config.candidateN = [150, 200, 250, 300, 350, 400];
config.representativeN = [150, 300, 400];
config.d = 3;
config.gaussianExponent = 5;

config.preconditionerTuckerRanks = [7, 8, 9, 9, 9, 10];
config.maximumPreconditionerSpectralDelta = 0.1;

config.maxit = 4;
config.originalTrueResidualTarget = 1e-6;
config.internalPreconditionedStoppingTolerance = 1e-14;
config.diagnosticTolerance = 1e-13;

config.targetToBaseToleranceRatio = 1e4;
config.fixedRecompressionTolerance = ...
    config.originalTrueResidualTarget / ...
    config.targetToBaseToleranceRatio;
config.relax_budget = ...
    config.fixedRecompressionTolerance;
config.max_relax_tol = 1e-2;

config.methodIds = ["full"; "fixed"; "relaxed"];
config.methodNames = [ ...
    "Preconditioned full GMRES"; ...
    "Fixed Tucker-GMRES"; ...
    "Relaxed Tucker-GMRES"];

% Three cyclic orders place every method once in every execution position.
config.balancedOrder = [ ...
    1, 2, 3; ...
    2, 3, 1; ...
    3, 1, 2];
config.numberOfMeasuredRepeats = size(config.balancedOrder, 1);
config.numberOfWarmupsPerMeasuredProcess = 1;

config.reusedN = NaN;
config.numberOfReusedN300Repeats = 0;

config.maximumTuckerBasisGiB = 2.0;
config.maximumTuckerBasisEntries = ...
    floor(config.maximumTuckerBasisGiB * 2^30 / 8);

config.maximumPreallocatedFullBasisGiB = 2.5;
config.estimatedPeakFullArrayCopies = 18;
config.maximumEstimatedFullArrayGiB = 9.0;

end
