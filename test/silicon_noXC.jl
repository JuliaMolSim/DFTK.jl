include("silicon_runners.jl")

@testset "SCF of silicon without exchange-correlation (small)" begin
    run_silicon_noXC(Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15)
end
