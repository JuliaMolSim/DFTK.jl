include("silicon_runners.jl")

@testset "Silicon LDA (small, Float64)" begin
    run_silicon_lda(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17, scf_tol=1e-5)
end

@warn "Float32 test disabled"
#@testset "Silicon LDA (small, Float32)" begin
#    run_silicon_lda(Float32, Ecut=7, test_tol=0.03, n_ignored=1, grid_size=19, scf_tol=1e-4,
#                    lobpcg_tol=5e-5, n_noconv_check=1)
#end
