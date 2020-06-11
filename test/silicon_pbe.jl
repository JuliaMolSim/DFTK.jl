include("silicon_runners.jl")

@testset "Silicon PBE (small, Float64)" begin
    run_silicon_pbe(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17)
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Silicon PBE (large, Float64)" begin
        run_silicon_pbe(Float64, Ecut=25, test_tol=1e-5, n_ignored=0,
                        grid_size=33, scf_tol=1e-7)
    end
end
