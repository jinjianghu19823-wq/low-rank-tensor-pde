function timing = empty_tucker_kernel_timing()
%EMPTY_TUCKER_KERNEL_TIMING Return zeroed leaf-kernel timing fields.
%
% These fields describe leaf operations inside Tucker arithmetic. They are
% intentionally separate from the non-overlapping solver-phase timings in
% run_left_preconditioned_tucker_gmres_cycle. For example, STHOSVD SVD time
% is already contained in a recompression call, so it must not be added to
% the recompression phase again when reconstructing total solver time.

timing.poisson_mode_product_time_sec = 0;
timing.preconditioner_forward_dst_time_sec = 0;
timing.preconditioner_exact_hadamard_time_sec = 0;
timing.preconditioner_inverse_dst_time_sec = 0;
timing.exact_sum_core_time_sec = 0;
timing.factor_concatenation_time_sec = 0;
timing.factor_qr_time_sec = 0;
timing.core_transform_time_sec = 0;
timing.sthosvd_unfolding_time_sec = 0;
timing.sthosvd_svd_time_sec = 0;
timing.sthosvd_rank_selection_time_sec = 0;
timing.sthosvd_projection_time_sec = 0;
timing.factor_reconstruction_time_sec = 0;

end
