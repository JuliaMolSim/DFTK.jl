include("silicon_runners.jl")

@testset "Silicon LDA (small, Float64)" begin
    run_silicon_lda(Float64, Ecut=5, test_tol=0.03, n_ignored=0, grid_size=15)
end
@testset "Silicon LDA (small, Float32)" begin
    run_silicon_lda(Float32, Ecut=5, test_tol=0.03, n_ignored=0, grid_size=15, scf_tol=1e-4)
end
