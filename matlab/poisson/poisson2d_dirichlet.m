function A = poisson2d_dirichlet(N)
% Discretises -Delta u = f on (0,1)^2 using N interior points per direction.

    h = 1 / (N + 1);
    e = ones(N, 1);

    A1 = spdiags([-e, 2*e, -e], -1:1, N, N) / h^2;
    I  = speye(N);

    A = kron(I, A1) + kron(A1, I);
end