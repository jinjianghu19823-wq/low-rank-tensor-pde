function T = compare_svd_rsvd(A, matrix_name, ranks, p, ntrials)

    [m, n] = size(A);
    normA = norm(A, "fro");

    fprintf("\n=====================================\n");
    fprintf("Matrix: %s, size %d x %d\n", matrix_name, m, n);
    fprintf("=====================================\n");

    tic;
    [U,S,V] = svd(A, 'econ');
    time_svd_full = toc;

    singular_values = diag(S);

    fprintf("Economy SVD time: %.4f seconds\n", time_svd_full);

    num_ranks = length(ranks);
    Matrix = strings(num_ranks, 1);
    Rank = zeros(num_ranks,1);
    Oversampling = zeros(num_ranks,1);

    SVD_RelErr = zeros(num_ranks,1);
    RSVD_RelErrMean = zeros(num_ranks,1);
    RSVD_RelErrStd = zeros(num_ranks,1);
    ErrorRatio = zeros(num_ranks,1);
    GN_RelErrMean = zeros(num_ranks,1);
    GN_RelErrStd = zeros(num_ranks,1);
    GN_ErrorRatio = zeros(num_ranks,1);

    SVD_Time = zeros(num_ranks,1);
    RSVD_TimeMean = zeros(num_ranks,1);
    RSVD_TimeStd = zeros(num_ranks,1);
    Speedup = zeros(num_ranks,1);
    GN_TimeMean = zeros(num_ranks,1);
    GN_TimeStd = zeros(num_ranks,1);
    GN_Speedup = zeros(num_ranks,1);
    GN_CondWMean = zeros(num_ranks,1);
    GN_CondWStd = zeros(num_ranks,1);

    CompressionFactor = zeros(num_ranks,1);

    for k = 1:num_ranks
        r = ranks(k);

        if r+p > min(m,n)
            error("r + p must be <= min(m,n)");
        end

        tail = singular_values(r+1:end);
        svd_relerr = norm(tail) / normA;

        rsvd_errors = zeros(ntrials, 1);
        rsvd_times = zeros(ntrials, 1);
        gn_errors = zeros(ntrials, 1);
        gn_times = zeros(ntrials, 1);
        gn_condW = zeros(ntrials, 1);

        for trial = 1:ntrials
            rng(1000 + 100*k + trial);

            tic;
            [Ur,Sr,Vr] = rsvd(A, r, p);
            rsvd_times(trial) = toc;

            A_rsvd = Ur * Sr * Vr';
            rsvd_errors(trial) = norm(A - A_rsvd, "fro") / normA;

            rng(3000 + 100*k + trial);

            [Lgn, Mgn, info_gn, t_gn] = generalized_nystrom(A, r, p, false);
            A_gn = Lgn * Mgn;
            gn_times(trial) = t_gn;
            gn_errors(trial) = norm(A - A_gn, "fro") / normA;
            gn_condW(trial) = info_gn.condW;
        end

        full_storage = m*n;
        lowrank_storage = m*r + r + n*r;
        compression_factor = full_storage / lowrank_storage;

        Matrix(k) = matrix_name;
        Rank(k) = r;
        Oversampling(k) = p;

        SVD_RelErr(k) = svd_relerr;
        RSVD_RelErrMean(k) = mean(rsvd_errors);
        RSVD_RelErrStd(k) = std(rsvd_errors);
        ErrorRatio(k) = mean(rsvd_errors) / svd_relerr;
        GN_RelErrMean(k) = mean(gn_errors);
        GN_RelErrStd(k) = std(gn_errors);
        GN_ErrorRatio(k) = mean(gn_errors) / svd_relerr;

        SVD_Time(k) = time_svd_full;
        RSVD_TimeMean(k) = mean(rsvd_times);
        RSVD_TimeStd(k) = std(rsvd_times);
        Speedup(k) = time_svd_full / mean(rsvd_times);
        GN_TimeMean(k) = mean(gn_times);
        GN_TimeStd(k) = std(gn_times);
        GN_Speedup(k) = time_svd_full / mean(gn_times);
        GN_CondWMean(k) = mean(gn_condW);
        GN_CondWStd(k) = std(gn_condW);

        CompressionFactor(k) = compression_factor;

        fprintf("r = %4d | SVD err = %.3e | RSVD err = %.3e ± %.1e | GN err = %.3e ± %.1e | RSVD speedup = %.2fx | GN speedup = %.2fx\n", ...
            r, svd_relerr, mean(rsvd_errors), std(rsvd_errors), mean(gn_errors), std(gn_errors), ...
            time_svd_full/mean(rsvd_times), time_svd_full/mean(gn_times));
    end

    T = table(Matrix, Rank, Oversampling, ...
        SVD_RelErr, ...
        RSVD_RelErrMean, RSVD_RelErrStd, ErrorRatio, ...
        GN_RelErrMean, GN_RelErrStd, GN_ErrorRatio, ...
        SVD_Time, ...
        RSVD_TimeMean, RSVD_TimeStd, Speedup, ...
        GN_TimeMean, GN_TimeStd, GN_Speedup, GN_CondWMean, GN_CondWStd, ...
        CompressionFactor);
end
