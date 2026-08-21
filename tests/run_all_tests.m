function run_all_tests(tensorToolboxRoot)
%RUN_ALL_TESTS Run the focused release test suite.

if nargin < 1
    add_toolboxes();
else
    add_toolboxes(tensorToolboxRoot);
end

tests = { ...
    @test_matrix_tensor_algorithms, ...
    @test_sgmres_trunc_arnoldi, ...
    @test_sthosvd_tail_stability, ...
    @test_poisson_fast_diagonalization_tucker_preconditioner, ...
    @test_poisson_fast_diagonalization_full_gmres, ...
    @test_tucker_gmres_algorithms, ...
    @test_tucker_roundsum_rhosvd, ...
    @test_tucker_sgmres_foundation};

for index = 1:numel(tests)
    testFunction = tests{index};
    fprintf('\n=== %s ===\n', func2str(testFunction));
    testFunction();
end

fprintf('\nAll %d focused tests passed.\n', numel(tests));

end
