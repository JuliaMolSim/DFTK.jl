include("silicon_runners.jl")
using DoubleFloats

@testset "SCF of silicon without exchange-correlation (small)" begin
    run_silicon_redHF(Float64, Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15)
end

@testset "Silicon LDA (small, Double32)" begin
    run_silicon_redHF(Double32, Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15)
end
