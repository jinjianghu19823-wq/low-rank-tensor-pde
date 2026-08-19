function [U,S,V] = rsvd(A,r,p)
% A: m-by-n matrix
% r: target rank
% p: oversampling parameter

    [m,n] = size(A);
    l = r + p;
    % Generate a random matrix for projection
    Omega = randn(n, l);
    Y = A * Omega;

    [Q,~] = qr(Y,"econ");
    B = Q' * A;

    [Ub,Sb,Vb] = svd(B, "econ");

    U = Q * Ub(:, 1:r);
    S = Sb(1:r, 1:r);
    V = Vb(:, 1:r);
end
