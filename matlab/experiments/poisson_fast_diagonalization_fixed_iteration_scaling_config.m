function config = poisson_fast_diagonalization_fixed_iteration_scaling_config()
%POISSON_FAST_DIAGONALIZATION_FIXED_ITERATION_SCALING_CONFIG
% Frozen four-step comparison with an original-residual reference of 1e-6.
%
% Thesis/experiment notation (Section 6.2):
%   candidateN                       <->  N
%   numberOfModes                   <->  d=3
%   preconditionerTuckerRanks       <->  selected rank r of D_r
%   maximumPreconditionerSpectralDelta
%                                    <->  maximum spectral defect delta
%   maximumIteration                <->  four-step basis limit
%   originalTrueResidualTarget      <->  external reference 10^(-6)
%   internalPreconditionedStoppingTolerance
%                                    <->  internal threshold 10^(-14)
%   fixedRecompressionTolerance     <->  fixed eta=10^(-10)
%   relaxationErrorBudget           <->  relaxed base budget 10^(-10)
%   maximumRelaxedTolerance         <->  eta_max=10^(-2)

config.experimentId = ...
    "2026-07-30_poisson_fast_diagonalization_fixed_iteration_scaling";
config.outputPrefix = ...
    "poisson_fast_diagonalization_fixed_iteration_scaling";

config.candidateN = [150, 200, 250, 300, 350, 400];
config.representativeN = [150, 300, 400];
config.numberOfModes = 3;
config.gaussianExponent = 5;

% These are the first ranks that met delta <= 0.1 in the protected
% preconditioner-selection study.
config.preconditionerTuckerRanks = [7, 8, 9, 9, 9, 10];
config.maximumPreconditionerSpectralDelta = 0.1;

% Every method completes the same four Arnoldi steps. The small
% preconditioned-residual threshold is deliberately below the values
% reached in this window, so it does not shorten a cycle.
config.maximumIteration = 4;
config.originalTrueResidualTarget = 1e-6;
config.internalPreconditionedStoppingTolerance = 1e-14;
config.diagnosticTolerance = 1e-13;

% The earlier 1e-8 study used a 1e-12 base Tucker error budget. The new
% protocol preserves the same target-to-base ratio of 1e4.
config.targetToBaseToleranceRatio = 1e4;
config.fixedRecompressionTolerance = ...
    config.originalTrueResidualTarget / ...
    config.targetToBaseToleranceRatio;
config.relaxationErrorBudget = ...
    config.fixedRecompressionTolerance;
config.maximumRelaxedTolerance = 1e-2;

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

% No previous timing rows are reused. The new tolerance changes both
% Tucker paths, and all three methods are rerun in the same balanced study.
config.reusedN = NaN;
config.numberOfReusedN300Repeats = 0;

config.maximumTuckerBasisGiB = 2.0;
config.maximumTuckerBasisEntries = ...
    floor(config.maximumTuckerBasisGiB * 2^30 / 8);

config.maximumPreallocatedFullBasisGiB = 2.5;
config.estimatedPeakFullArrayCopies = 18;
config.maximumEstimatedFullArrayGiB = 9.0;

end
