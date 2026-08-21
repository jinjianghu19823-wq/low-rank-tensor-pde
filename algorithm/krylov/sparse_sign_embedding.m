function S = sparse_sign_embedding(s, m, zeta)
%SPARSE_SIGN_EMBEDDING Construct a basic sparse sign matrix.
%
% S has s rows and m columns.
% Each column has exactly zeta nonzero entries.
% Every nonzero is either +1/sqrt(zeta) or -1/sqrt(zeta).

    if zeta > s
        error('zeta must not be larger than s.');
    end

    numberOfEntries = zeta * m;

    rowIndices = zeros(numberOfEntries, 1);
    columnIndices = zeros(numberOfEntries, 1);
    values = zeros(numberOfEntries, 1);

    entry = 0;

    for j = 1:m
        % Choose zeta different rows for column j.
        selectedRows = randperm(s, zeta);

        for k = 1:zeta
            entry = entry + 1;

            rowIndices(entry) = selectedRows(k);
            columnIndices(entry) = j;

            if rand < 0.5
                randomSign = -1;
            else
                randomSign = 1;
            end

            values(entry) = randomSign / sqrt(zeta);
        end
    end

    S = sparse(rowIndices, columnIndices, values, s, m);
end
