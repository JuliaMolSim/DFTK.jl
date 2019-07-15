include("../silicon_runners.jl")

@testset "SCF of silicon without exchange-correlation (medium)" begin
    run_silicon_noXC(Ecut=15, test_tol=0.0005, n_ignored=5, grid_size=25)
end

@testset "SCF of silicon without exchange-correlation (large)" begin
    run_silicon_noXC(Ecut=25, test_tol=2e-7, n_ignored=0, grid_size=33, scf_tol=1e-7)
end
