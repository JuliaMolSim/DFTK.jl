@testitem "split_evenly" begin
    using DFTK

    function run_test(itr, n)
        splitted = DFTK.split_evenly(itr, n)
        @test collect(itr) == reduce(vcat, splitted)
    end

    run_test(1:12, 4)
    run_test(1:31, 7)
    run_test(1:168, 16)
    run_test(1:14, 16)
end
