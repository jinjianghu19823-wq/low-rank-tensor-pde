function y = poisson_action_full_vector_3d( ...
    x, A1, N)
%POISSON_ACTION_FULL_VECTOR_3D Apply the 3D Poisson operator matrix-free.

x = x(:);

if length(x) ~= N^3
    error('x must contain exactly N^3 entries.');
end

if N < 2 || ~isequal(size(A1), [N, N])
    error('N must be at least 2 and A1 must have size N-by-N.');
end

diag_coeff = full(A1(1,1));
offdiag = full(A1(1,2));

X = reshape(x, [N, N, N]);

Y = 3 * diag_coeff * X;

Y(1:end-1,:,:) = Y(1:end-1,:,:) + offdiag * X(2:end,:,:);
Y(2:end,:,:) = Y(2:end,:,:) + offdiag * X(1:end-1,:,:);

Y(:,1:end-1,:) = Y(:,1:end-1,:) + offdiag * X(:,2:end,:);
Y(:,2:end,:) = Y(:,2:end,:) + offdiag * X(:,1:end-1,:);

Y(:,:,1:end-1) = Y(:,:,1:end-1) + offdiag * X(:,:,2:end);
Y(:,:,2:end) = Y(:,:,2:end) + offdiag * X(:,:,1:end-1);

y = Y(:);

end
