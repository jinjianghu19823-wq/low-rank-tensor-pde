function diagnostic = diagnose_poisson_exponential_sum_preconditioner( ...
    preconditioner, useExactDiscreteSpectrum, intervalSampleCount)
%DIAGNOSE_POISSON_EXPONENTIAL_SUM_PRECONDITIONER Diagnose M_q A_d.
%
% The inexpensive diagnostic samples the full spectral interval of A_d.
% For d=2 or d=3, the optional exact diagnostic then visits every discrete
% Kronecker-sum eigenvalue in small blocks without forming A_d.
%
% The outputs include
%
%     delta_q = max |1 - lambda*mu_q(lambda)|
%
% and the observed condition number
%
%     max(lambda*mu_q(lambda)) / min(lambda*mu_q(lambda)).


%% 1. Check the inputs

if nargin < 2
    useExactDiscreteSpectrum = false;
end

if nargin < 3
    intervalSampleCount = 20001;
end

if ~islogical(useExactDiscreteSpectrum) || ...
        ~isscalar(useExactDiscreteSpectrum)
    error('useExactDiscreteSpectrum must be one logical value.');
end

if intervalSampleCount < 3 || intervalSampleCount ~= floor(intervalSampleCount)
    error('intervalSampleCount must be an integer of at least 3.');
end

lambda1 = preconditioner.one_dimensional_eigenvalues;
numberOfModes = preconditioner.number_of_modes;

lambdaMinimum = numberOfModes * min(lambda1);
lambdaMaximum = numberOfModes * max(lambda1);


%% 2. Sample the complete spectral interval

intervalLambda = linspace( ...
    lambdaMinimum, lambdaMaximum, intervalSampleCount).';
intervalValues = evaluate_poisson_preconditioner_spectrum( ...
    preconditioner, intervalLambda);

intervalMinimum = min(intervalValues);
intervalMaximum = max(intervalValues);
intervalDelta = max(abs(1 - intervalValues));


%% 3. Optionally visit every discrete Kronecker-sum eigenvalue

exactMinimum = NaN;
exactMaximum = NaN;
exactDelta = NaN;
numberOfDiscreteEigenvalues = preconditioner.N^numberOfModes;

if useExactDiscreteSpectrum

    if numberOfModes == 1

        exactValues = evaluate_poisson_preconditioner_spectrum( ...
            preconditioner, lambda1);
        exactMinimum = min(exactValues);
        exactMaximum = max(exactValues);
        exactDelta = max(abs(1 - exactValues));

    elseif numberOfModes == 2

        pairSums = lambda1 + lambda1.';
        exactValues = evaluate_poisson_preconditioner_spectrum( ...
            preconditioner, pairSums(:));
        exactMinimum = min(exactValues);
        exactMaximum = max(exactValues);
        exactDelta = max(abs(1 - exactValues));

    elseif numberOfModes == 3

        pairSums = lambda1 + lambda1.';
        exactMinimum = Inf;
        exactMaximum = -Inf;
        exactDelta = 0;

        % Each block contains all combinations of the final two modes for
        % one fixed eigenvalue in the first mode.
        for firstModeIndex = 1:numel(lambda1)

            spectrumBlock = lambda1(firstModeIndex) + pairSums(:);
            valueBlock = evaluate_poisson_preconditioner_spectrum( ...
                preconditioner, spectrumBlock);

            exactMinimum = min(exactMinimum, min(valueBlock));
            exactMaximum = max(exactMaximum, max(valueBlock));
            exactDelta = max( ...
                exactDelta, max(abs(1 - valueBlock)));

        end

    else
        error(['Exact discrete diagnosis is implemented only for ', ...
               'one, two, or three modes.']);
    end

end


%% 4. Return the condition diagnostics

diagnostic.N = preconditioner.N;
diagnostic.number_of_modes = numberOfModes;
diagnostic.q = preconditioner.q;
diagnostic.number_of_terms = preconditioner.number_of_terms;
diagnostic.zeta = preconditioner.zeta;
diagnostic.lambda_minimum = lambdaMinimum;
diagnostic.lambda_maximum = lambdaMaximum;
diagnostic.interval_sample_count = intervalSampleCount;
diagnostic.interval_minimum_preconditioned_eigenvalue = intervalMinimum;
diagnostic.interval_maximum_preconditioned_eigenvalue = intervalMaximum;
diagnostic.interval_delta = intervalDelta;
diagnostic.interval_condition_number = ...
    intervalMaximum / intervalMinimum;
diagnostic.used_exact_discrete_spectrum = useExactDiscreteSpectrum;
diagnostic.number_of_discrete_eigenvalues = ...
    numberOfDiscreteEigenvalues;
diagnostic.exact_minimum_preconditioned_eigenvalue = exactMinimum;
diagnostic.exact_maximum_preconditioned_eigenvalue = exactMaximum;
diagnostic.exact_delta = exactDelta;
diagnostic.exact_condition_number = exactMaximum / exactMinimum;

if useExactDiscreteSpectrum
    selectedDelta = exactDelta;
else
    selectedDelta = intervalDelta;
end

diagnostic.condition_bound_from_delta = ...
    (1 + selectedDelta) / (1 - selectedDelta);
diagnostic.delta_is_below_one = selectedDelta < 1;

end
