function config = weak_preconditioned_tucker_four_method_scaling_config()
%WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING_CONFIG Frozen design.
%
% The scaling study uses exactly 100 Arnoldi operator products. The first
% N=150 case is a feasibility gate. Larger grids are enabled only after the
% measured time, rank, and retained basis storage have been inspected.

config.experimentId = ...
    "2026-08-19_weak_preconditioned_tucker_four_method_scaling";
config.outputPrefix = ...
    "weak_preconditioned_tucker_four_method_scaling";

config.candidateN = [150; 200; 250];
config.pilotN = 150;
config.representativeN = [150; 200; 250];
config.numberOfModes = 3;
config.gaussianExponent = 5;
config.maximumIterations = 100;
config.diagnosticCheckpoints = [0; 20; 40; 60; 80; 100];
config.internalStoppingTolerance = 1e-14;
config.diagnosticTolerance = 1e-12;

% A common rank one approximation is used to keep a long Krylov process.
% Its complete preconditioned spectrum is checked separately at every N.
config.preconditionerTuckerRank = 1;

config.fixedCompressionTolerance = 1e-10;
config.relaxationErrorBudget = 1e-10;
config.maximumRelaxedTolerance = 1e-2;

config.orthogonalisationWindow = 2;
config.residualSketchSize = 336;
config.residualSketchSeeds = [101; 202; 303; 404; 505];
config.timingResidualSketchSeed = 101;
config.sketchStoppingTolerance = 1e-14;
config.plainCompressionTolerance = 1e-10;

config.roundSumCompressionTolerance = 1e-10;
config.solutionRoundSumTolerance = 1e-10;
config.roundSumOversampling = 5;
config.basisRoundSumSeedOffset = 1400000;
config.solutionRoundSumSeedOffset = 1500000;
config.measureRoundSumErrorDiagnostics = false;

config.maximumTuckerBasisGiB = 2;
config.maximumTuckerBasisEntries = ...
    floor(config.maximumTuckerBasisGiB * 2^30 / 8);

config.methodIds = ["fixed"; "relaxed"; "plain"; "rhosvd_sgmres"];
config.methodNames = [ ...
    "Fixed Tucker GMRES"; ...
    "Relaxed Tucker GMRES"; ...
    "Plain Tucker sGMRES"; ...
    "RHOSVD Tucker sGMRES"];

config.timingMethodOrders = [ ...
    1, 2, 3, 4; ...
    2, 3, 4, 1; ...
    3, 4, 1, 2; ...
    4, 1, 2, 3];
config.numberOfMeasuredTimingBlocks = ...
    size(config.timingMethodOrders, 1);
config.numberOfWarmupsPerMethod = 1;
config.maximumTimingEndpointRelativeDrift = 1e-8;
config.maximumTimingClosureRelativeError = 1e-8;

% The pilot is allowed to continue only if every method completes without
% a rank or memory safeguard. The measured pilot time is used to choose the
% final safe N rather than an extrapolated runtime claim.
config.pilotResidualSketchSeed = 101;

end
