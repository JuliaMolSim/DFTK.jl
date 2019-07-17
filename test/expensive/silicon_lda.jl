include("../silicon_runners.jl")

@testset "Silicon LDA (large, Float64)" begin
    run_silicon_lda(Float64, Ecut=25, test_tol=4e-7, n_ignored=0, grid_size=33, scf_tol=1e-7)
end
