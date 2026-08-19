function values = evaluate_poisson_preconditioner_spectrum( ...
    preconditioner, lambda)
%EVALUATE_POISSON_PRECONDITIONER_SPECTRUM Evaluate lambda*mu_q(lambda).
%
% For an eigenvalue lambda of A_d, mu_q(lambda) is the corresponding
% eigenvalue of M_q. The returned value
%
%     lambda * mu_q(lambda)
%
% is therefore the eigenvalue of the left-preconditioned matrix M_q A_d.

lambda = double(lambda);
mu = zeros(size(lambda));

for termIndex = 1:preconditioner.number_of_terms
    mu = mu + preconditioner.omega(termIndex) * ...
        exp(-preconditioner.tau(termIndex) * lambda);
end

values = lambda .* mu;

end
