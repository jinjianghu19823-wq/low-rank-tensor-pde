function run_all_tests()
%RUN_ALL_TESTS Run the focused correctness tests for the public release.

add_toolboxes();

tests = { ...
    @test_tucker_sgmres_foundation, ...
    @test_tucker_roundsum_rhosvd, ...
    @test_poisson_fast_diagonalization_tucker_preconditioner};

for test_index = 1:numel(tests)
    test_function = tests{test_index};
    fprintf("\nRunning %s\n", func2str(test_function));
    test_function();
end

fprintf("\nAll %d focused tests passed.\n", numel(tests));
end
