function [Z, info] = apply_poisson_exponential_sum_preconditioner( ...
    Y, preconditioner, compressionTolerance, maximumMultilinearRank)
%APPLY_POISSON_EXPONENTIAL_SUM_PRECONDITIONER Apply M_q in Tucker form.
%
% Each exponential term acts separately on the Tucker factor matrices:
%
%     exp(-tau_k A1) U
%       = Q diag(exp(-tau_k*lambda)) Q' U.
%
% The 2q+1 resulting Tucker tensors are added one at a time. The partial sum
% is recompressed after every addition to control temporary ranks. This is
% the practical sequential-rounding implementation discussed after
% Algorithm 5.7. Its several local truncations are recorded in info; they
% are not claimed to equal one final truncation of the complete sum.


%% 1. Check the inputs

if ~isa(Y, 'ttensor')
    error('Y must be a Tensor Toolbox ttensor object.');
end

if ~isstruct(preconditioner) || ...
        ~isfield(preconditioner, 'number_of_terms')
    error('preconditioner must come from the build function.');
end

if ~isscalar(compressionTolerance) || compressionTolerance <= 0 || ...
        compressionTolerance >= 1
    error('compressionTolerance must be one number between 0 and 1.');
end

if nargin < 4
    maximumMultilinearRank = [];
end

numberOfModes = ndims(Y);

if numberOfModes ~= preconditioner.number_of_modes
    error('The tensor order does not match the preconditioner.');
end

if any(size(Y) ~= preconditioner.N)
    error('Every tensor mode must have size preconditioner.N.');
end

maximumRanks = normalise_tucker_rank_cap( ...
    maximumMultilinearRank, size(Y));


%% 2. Prepare diagnostics for the sequential sum

numberOfTerms = preconditioner.number_of_terms;

% There are numberOfTerms-1 rounded additions and one final round. Allocate
% an equal part of the requested complete-application budget to each local
% recompression. This conservative division follows the triangle-inequality
% worst case; the independently recomputed preconditioned residual still
% checks the accumulated error in practice.
localCompressionTolerance = ...
    compressionTolerance / numberOfTerms;

additionRankCapActive = false(max(numberOfTerms - 1, 0), 1);
additionRelativeErrorEstimate = zeros(max(numberOfTerms - 1, 0), 1);
rankBeforeRecompression = zeros(max(numberOfTerms - 1, 0), numberOfModes);

Q = preconditioner.eigenvectors;
lambda = preconditioner.one_dimensional_eigenvalues;


%% 3. Build and add each separable exponential term

for termIndex = 1:numberOfTerms

    tau = preconditioner.tau(termIndex);
    omega = preconditioner.omega(termIndex);
    exponentialEigenvalues = exp(-tau * lambda);

    transformedFactors = cell(numberOfModes, 1);

    for mode = 1:numberOfModes

        % Apply exp(-tau*A1) to the current mode factor without forming the
        % dense matrix exponential.
        transformedFactors{mode} = Q * ( ...
            exponentialEigenvalues .* (Q.' * Y.u{mode}));

    end

    termTensor = ttensor(omega * Y.core, transformedFactors);

    if termIndex == 1
        partialSum = termTensor;
    else

        rankBeforeRecompression(termIndex - 1, :) = ...
            size(partialSum.core) + size(termTensor.core);

        [partialSum, additionInfo] = tucker_axpby_round( ...
            partialSum, 1, termTensor, 1, ...
            localCompressionTolerance, ...
            maximumRanks);

        additionRankCapActive(termIndex - 1) = ...
            additionInfo.rank_cap_active;
        additionRelativeErrorEstimate(termIndex - 1) = ...
            additionInfo.relative_error_estimate;

    end

end


%% 4. Apply one final recompression to the represented partial sum

[Z, finalRoundInfo] = tucker_round( ...
    partialSum, localCompressionTolerance, maximumRanks);


%% 5. Return diagnostics

info.q = preconditioner.q;
info.number_of_terms = numberOfTerms;
info.requested_tolerance = compressionTolerance;
info.local_recompression_tolerance = localCompressionTolerance;
info.maximum_ranks = maximumRanks;
info.rank_before_recompression = rankBeforeRecompression;
info.addition_rank_cap_active = additionRankCapActive;
info.addition_relative_error_estimate = ...
    additionRelativeErrorEstimate;
info.final_relative_error_estimate = ...
    finalRoundInfo.relative_error_estimate;
info.final_rank_cap_active = finalRoundInfo.rank_cap_active;
info.rank_cap_active = ...
    any(additionRankCapActive) || finalRoundInfo.rank_cap_active;
info.maximum_local_recompression_error = max( ...
    [0; additionRelativeErrorEstimate; ...
     finalRoundInfo.relative_error_estimate]);
info.output_ranks = size(Z.core);

end
