function config = weak_preconditioned_tucker_four_method_scaling_config()
%WEAK_PRECONDITIONED_TUCKER_FOUR_METHOD_SCALING_CONFIG Frozen design.

config.experimentId = ...
    "2026-08-19_weak_preconditioned_tucker_four_method_scaling";
config.outputPrefix = ...
    "weak_preconditioned_tucker_four_method_scaling";

config.candidateN = [150; 200; 250];
config.pilotN = 150;
config.representativeN = [150; 200; 250];
config.d = 3;
config.gaussianExponent = 5;
config.maximumIterations = 100;
config.diagnosticCheckpoints = [0; 20; 40; 60; 80; 100];
config.internalStoppingTolerance = 1e-14;
config.diagnosticTolerance = 1e-12;

config.preconditionerTuckerRank = 1;

config.fixed_tol = 1e-10;
config.relax_budget = 1e-10;
config.max_relax_tol = 1e-2;

config.ktrunc = 2;
config.residualSketchSize = 336;
config.residualSketchSeeds = [101; 202; 303; 404; 505];
config.timingResidualSketchSeed = 101;
config.sketch_tol = 1e-14;
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

config.pilotResidualSketchSeed = 101;

end
