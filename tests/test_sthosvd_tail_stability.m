function test_sthosvd_tail_stability
%TEST_STHOSVD_TAIL_STABILITY Check tight-tolerance rank selection.
%
% The spectrum below has one dominant singular value followed by a long,
% small tail. At a relative tolerance of 1e-8, calculating the tail energy
% as total energy minus retained energy is cancellation-prone. This test
% checks that STHOSVD instead selects the rank from backward tail norms.


%% 0. Prepare MATLAB and Tensor Toolbox

add_toolboxes();

fprintf('\nStarting the STHOSVD tail-stability test.\n\n');


%% 1. Construct a matrix with a known singular-value tail

matrixSize = 150;
singularValuesSquared = [ ...
    1; ...
    logspace(-2, -18, matrixSize - 1).' ...
];
singularValues = sqrt(singularValuesSquared);

X = tensor(diag(singularValues));

relativeTolerance = 1e-8;
numberOfModes = 2;
inputNorm = norm(singularValues);
modeErrorThreshold = ...
    relativeTolerance * inputNorm / sqrt(numberOfModes);


%% 2. Calculate the reference tail norms without subtraction

referenceTailNorm = zeros(matrixSize, 1);

for rankIndex = matrixSize-1:-1:1
    referenceTailNorm(rankIndex) = hypot( ...
        singularValues(rankIndex + 1), ...
        referenceTailNorm(rankIndex + 1));
end

expectedRank = find( ...
    referenceTailNorm <= modeErrorThreshold, 1, 'first');

assert(~isempty(expectedRank) && expectedRank < matrixSize, ...
    'The test spectrum must have a nontrivial tolerance-selected rank.');


%% 3. Check the STHOSVD rank and error diagnostics

[T, info] = sthosvd_round_tensor( ...
    X, relativeTolerance, [], 1:numberOfModes);

actualRelativeError = norm(X - full(T)) / norm(X);
reportedFirstModeTail = sqrt(info.discarded_energy(1));

assert(info.tolerance_ranks(1) == expectedRank, ...
    'STHOSVD did not select the rank from the stable tail norm.');
assert(info.retained_ranks(1) == expectedRank, ...
    'The inactive rank cap changed the tolerance-selected rank.');
assert(~info.rank_cap_active, ...
    'No hard rank cap was requested in this test.');
assert(abs(reportedFirstModeTail - ...
    referenceTailNorm(expectedRank)) <= ...
    100 * eps(max(1, referenceTailNorm(expectedRank))), ...
    'The reported discarded norm is inconsistent with the stable tail.');
assert(actualRelativeError <= ...
    relativeTolerance * (1 + 100 * eps), ...
    'The recompressed tensor exceeds the requested relative tolerance.');

fprintf('  Matrix size: %d\n', matrixSize);
fprintf('  Stable tolerance-selected rank: %d\n', expectedRank);
fprintf('  Actual relative error: %.3e\n', actualRelativeError);
fprintf('  Reported relative error estimate: %.3e\n', ...
    info.relative_error_estimate);
fprintf('  STHOSVD tail-stability test passed.\n');

end
