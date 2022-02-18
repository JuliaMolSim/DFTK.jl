using DFTK
using ForwardDiff
using Test
include("testcases.jl")

@testset "Force derivatives using finite differences" begin
    function compute_force(ε::T) where T
        Si  = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8 + ε * [1., 0, 0]]
        atoms = [Si => pos]
        # TODO symmetries = true gives issues for now ... debug after refactoring done.
        model = model_DFT(Matrix{T}(silicon.lattice), atoms, [:lda_x, :lda_c_pw], symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged=is_converged)
        compute_forces_cart(scfres)[1]
    end

    derivative_fd = let ε = 1e-5
        (compute_force(ε) - compute_force(0.0)) / ε
    end
    derivative = ForwardDiff.derivative(compute_force, 0.0)
    @test maximum(maximum, derivative - derivative_fd) < 1e-4
end
