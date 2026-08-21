function config = ...
    poisson_fast_diagonalization_equal_accuracy_scaling_config()
%POISSON_FAST_DIAGONALIZATION_EQUAL_ACCURACY_SCALING_CONFIG
% Frozen multi-grid comparison for the main Section 6.2 evidence.

config.experimentId = ...
    "2026-07-30_poisson_fast_diagonalization_equal_accuracy_scaling";
config.outputPrefix = ...
    "poisson_fast_diagonalization_equal_accuracy_scaling";

config.candidateN = [150, 200, 250, 300, 350, 400];
config.numberOfModes = 3;
config.gaussianExponent = 5;

% These are the first ranks that met delta <= 0.1 in the protected
% preconditioned full-GMRES scale-up study. Every new case rebuilds the
% selected preconditioner and checks the same spectral condition.
config.preconditionerTuckerRanks = [7, 8, 9, 9, 9, 10];
config.maximumPreconditionerSpectralDelta = 0.1;

config.maximumIteration = 4;
config.originalTrueResidualTarget = 1e-8;
config.internalPreconditionedStoppingTolerance = 1e-14;
config.diagnosticTolerance = 1e-13;

config.fixedRecompressionTolerance = 1e-12;
config.relaxationErrorBudget = 1e-12;
config.maximumRelaxedTolerance = 1e-2;

config.methodIds = ["full"; "fixed"; "relaxed"];
config.methodNames = [ ...
    "Preconditioned full GMRES"; ...
    "Fixed Tucker-GMRES"; ...
    "Capped relaxed Tucker-GMRES"];

% Three cyclic orders place every method once in every execution position.
config.balancedOrder = [ ...
    1, 2, 3; ...
    2, 3, 1; ...
    3, 1, 2];
config.numberOfMeasuredRepeats = size(config.balancedOrder, 1);
config.numberOfWarmupsPerMeasuredProcess = 1;

% The N=300 central comparison already has all six permutations and uses the
% same mathematical and solver settings. It is reused rather than rerun.
config.reusedN = 300;
config.reusedN300OutputPrefix = ...
    "poisson_fast_diagonalization_equal_accuracy_timing_n300";
config.numberOfReusedN300Repeats = 6;

config.maximumTuckerBasisGiB = 2.0;
config.maximumTuckerBasisEntries = ...
    floor(config.maximumTuckerBasisGiB * 2^30 / 8);

% The delayed full solver stores five basis vectors. The conservative array
% estimate is retained from the protected full-GMRES scale-up preflight.
config.maximumPreallocatedFullBasisGiB = 2.5;
config.estimatedPeakFullArrayCopies = 18;
config.maximumEstimatedFullArrayGiB = 9.0;

end
