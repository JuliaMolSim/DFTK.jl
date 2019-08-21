include("silicon_runners.jl")

@testset "Silicon LDA (small, Float64)" begin
    run_silicon_lda(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17, scf_tol=1e-4)
end
@testset "Silicon LDA (small, Float32)" begin
    run_silicon_lda(Float32, Ecut=7, test_tol=0.03, n_ignored=1, grid_size=19, scf_tol=1e-4)
end
