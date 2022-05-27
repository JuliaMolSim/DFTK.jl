using DFTK
using ForwardDiff
using Test
include("testcases.jl")


@testset "Force derivatives using ForwardDiff" begin
    function compute_force(ε1, ε2; metal=false)
        T = promote_type(typeof(ε1), typeof(ε2))
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8 + ε1 * [1., 0, 0] + ε2 * [0, 1., 0]]
        if metal
            # Silicon reduced HF is metallic
            model = model_DFT(Matrix{T}(silicon.lattice), silicon.atoms, pos, [];
                              temperature=1e-3)
        else
            model = model_LDA(Matrix{T}(silicon.lattice), silicon.atoms, pos)
        end
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged,
                                       response=ResponseOptions(verbose=true))
        compute_forces_cart(scfres)
    end

    F = compute_force(0.0, 0.0)
    derivative_ε1_fd = let ε1 = 1e-5
        (compute_force(ε1, 0.0) - F) / ε1
    end
    derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0), 0.0)
    @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4

    derivative_ε2_fd = let ε2 = 1e-5
        (compute_force(0.0, ε2) - F) / ε2
    end
    derivative_ε2 = ForwardDiff.derivative(ε2 -> compute_force(0.0, ε2), 0.0)
    @test norm(derivative_ε2 - derivative_ε2_fd) < 1e-4

    @testset "Multiple partials" begin
        grad = ForwardDiff.gradient(v -> compute_force(v...)[1][1], [0.0, 0.0])
        @test abs(grad[1] - derivative_ε1[1][1]) < 1e-4
        @test abs(grad[2] - derivative_ε2[1][1]) < 1e-4

        jac = ForwardDiff.jacobian(v -> compute_force(v...)[1], [0.0, 0.0])
        @test norm(grad - jac[1, :]) < 1e-10
    end

    @testset "Derivative for metals" begin
        F = compute_force(0.0, 0.0; metal=true)
        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; metal=true) - F) / ε1
        end
        derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0; metal=true), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end
end


@testset "Force pseudo-sensitivity using ForwardDiff" begin
    # TODO Maybe later change to a band-energy test once eigenvalue derivatives work.
    function compute_force(ε::T) where {T}
        psp = load_psp("hgh/lda/si-q4")
        rloc = convert(T, psp.rloc)

        pspmod = PspHgh(psp.Zion, rloc,
                        psp.cloc, psp.rp .+ [0, ε], psp.h;
                        psp.identifier, psp.description)
        elem  = ElementPsp(silicon.atnum, psp=pspmod)
        atoms = [elem, elem]
        positions = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8]
        model = model_LDA(Matrix{T}(silicon.lattice), atoms, positions)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged,
                                       response=ResponseOptions(verbose=true))
        compute_forces_cart(scfres)
    end

    derivative_fd = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_ε = ForwardDiff.derivative(compute_force, 0.0)
    @test norm(derivative_fd - derivative_ε) < 1e-4
end
