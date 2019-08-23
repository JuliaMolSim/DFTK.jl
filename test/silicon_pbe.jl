include("silicon_runners.jl")

@testset "Silicon PBE (small, Float64)" begin
    run_silicon_pbe(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17)
end
