include("../silicon_runners.jl")

@testset "SCF of silicon without exchange-correlation (large)" begin
    run_silicon_redHF(Float64, Ecut=25, test_tol=1e-6, n_ignored=0, grid_size=33,
                      scf_tol=1e-7)
end
