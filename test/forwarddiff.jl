using DFTK
using ForwardDiff
using Test
using ComponentArrays

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

@testset "scfres PSP sensitivity using ForwardDiff" begin
    function compute_band_energies(ε::T) where {T}
        psp  = load_psp("hgh/lda/al-q3")
        rloc = convert(T, psp.rloc)

        pspmod = PspHgh(psp.Zion, rloc,
                        psp.cloc, psp.rp .+ [0, ε], psp.h;
                        psp.identifier, psp.description)
        atoms = fill(ElementPsp(aluminium.atnum, psp=pspmod), length(aluminium.positions))
        model = model_LDA(Matrix{T}(aluminium.lattice), atoms, aluminium.positions,
                          temperature=1e-2, smearing=Smearing.Gaussian())
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged, mixing=KerkerMixing(),
                                       bandspol=FixedBands(; n_bands_converge=10),
                                       damping=0.6, response=ResponseOptions(verbose=true))

        ComponentArray(
           eigenvalues=hcat([ev[1:10] for ev in scfres.eigenvalues]...),
           ρ=scfres.ρ,
           energies=collect(values(scfres.energies)),
           εF=scfres.εF,
           occupation=vcat(scfres.occupation...),
        )
    end

    derivative_ε = let ε = 1e-4
        (compute_band_energies(ε) - compute_band_energies(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_band_energies, 0.0)
    @test norm(derivative_fd - derivative_ε) < 5e-4
end

@testset "Functional force sensitivity using ForwardDiff" begin
    using DftFunctionals

    function compute_force(ε1::T) where {T}
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8]
        pbec = DftFunctional(:gga_c_pbe)
        pbex = DftFunctional(:gga_x_pbe)
        pbex = change_parameters(pbex, parameters(pbex) + ComponentArray(κ=0, μ=ε1))

        model = model_DFT(Matrix{T}(silicon.lattice), silicon.atoms, pos, [pbex, pbec])
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis;
                                       is_converged, response=ResponseOptions(verbose=true))
        compute_forces_cart(scfres)
    end

    derivative_ε = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_force, 0.0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end

@testset "Derivative of complex function" begin
    using SpecialFunctions, FiniteDifferences

    tol = 1e-10
    α = randn(ComplexF64)
    erfcα = x -> erfc(α * x)

    x0  = randn()
    fd1 = ForwardDiff.derivative(erfcα , x0)
    fd2 = FiniteDifferences.central_fdm(5, 1)(erfcα, x0)
    @test norm(fd1 - fd2) < tol
end
