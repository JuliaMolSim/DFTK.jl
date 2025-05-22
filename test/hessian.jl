@testmodule Hessian begin
using DFTK
using DFTK: select_occupied_orbitals, compute_projected_gradient

function setup_quantities(testcase)
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      functionals=[:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=3, kgrid=(3, 3, 3), fft_size=[9, 9, 9])
    scfres = self_consistent_field(basis; tol=10)

    ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

    ρ = compute_density(basis, ψ, occupation)
    rhs = compute_projected_gradient(basis, ψ, occupation)
    ϕ = rhs + ψ

    (; scfres, basis, ψ, occupation, ρ, rhs, ϕ)
end
end


@testitem "self-adjointness of solve_ΩplusK" #=
    =#    tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK: solve_ΩplusK
    using LinearAlgebra
    (; basis, ψ, occupation, rhs, ϕ) = Hessian.setup_quantities(TestCases.silicon)

    @test isapprox(
        real(dot(ϕ, solve_ΩplusK(basis, ψ, rhs, occupation).δψ)),
        real(dot(solve_ΩplusK(basis, ψ, ϕ, occupation).δψ, rhs)),
        atol=1e-7
    )
end

@testitem "self-adjointness of apply_Ω" #=
    =#    tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK
    using DFTK: apply_Ω
    using LinearAlgebra
    (; basis, ψ, occupation, ρ, rhs, ϕ) = Hessian.setup_quantities(TestCases.silicon)

    H = energy_hamiltonian(basis, ψ, occupation; ρ).ham
    # Rayleigh-coefficients
    Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ, H * ψ)]

    # Ω is complex-linear and so self-adjoint as a complex operator.
    @test isapprox(
        dot(ϕ, apply_Ω(rhs, ψ, H, Λ)),
        dot(apply_Ω(ϕ, ψ, H, Λ), rhs),
        atol=1e-7
    )
end

@testitem "self-adjointness of apply_K" #=
    =#    tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK: apply_K
    using LinearAlgebra
    (; basis, ψ, occupation, ρ, rhs, ϕ) = Hessian.setup_quantities(TestCases.silicon)

    # K involves conjugates and is only a real-linear operator,
    # hence we test using the real dot product.
    @test isapprox(
        real(dot(ϕ, apply_K(basis, rhs, ψ, ρ, occupation))),
        real(dot(apply_K(basis, ϕ, ψ, ρ, occupation), rhs)),
        atol=1e-7
    )
end

@testitem "ΩplusK_split, 0K" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: compute_projected_gradient
    using DFTK: select_occupied_orbitals, solve_ΩplusK, solve_ΩplusK_split
    using LinearAlgebra
    silicon = TestCases.silicon

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                      functionals=[:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=3, kgrid=(3, 3, 3), fft_size=[9, 9, 9])
    scfres = self_consistent_field(basis; tol=10)

    rhs = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
    ϕ = rhs + scfres.ψ

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
                        real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
                        atol=1e-7)
    end

    @testset "solve_ΩplusK_split agrees with solve_ΩplusK" begin
        scfres = self_consistent_field(basis; tol=1e-10)
        δψ1 = solve_ΩplusK_split(scfres, rhs).δψ
        δψ1 = select_occupied_orbitals(basis, δψ1, scfres.occupation).ψ
        (; ψ, occupation) = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)
        rhs_trunc = select_occupied_orbitals(basis, rhs, occupation).ψ
        δψ2 = solve_ΩplusK(basis, ψ, rhs_trunc, occupation).δψ
        @test norm(δψ1 - δψ2) < 1e-7
    end
end

@testitem "ΩplusK_split, temperature" tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK
    using DFTK: compute_projected_gradient, solve_ΩplusK_split
    using LinearAlgebra
    magnesium = TestCases.magnesium

    model = model_DFT(magnesium.lattice, magnesium.atoms, magnesium.positions;
                      functionals=[:lda_xc_teter93], magnesium.temperature)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=(3, 3, 3), fft_size=[9, 9, 9])
    nbandsalg = AdaptiveBands(basis.model; occupation_threshold=1e-10)
    scfres = self_consistent_field(basis; tol=1e-12, nbandsalg)

    ψ = scfres.ψ
    rhs = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
    ϕ = rhs + ψ

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
                        real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
                        atol=1e-7)
    end
end

@testitem "Test solve_ΩplusK_split achieves promised accuracies" begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    function test_solve_ΩplusK(scfres, δVext)
        # Compute a reference solution
        δHψ = DFTK.multiply_ψ_by_blochwave(scfres.basis, scfres.ψ, δVext)
        bandtolalg = DFTK.BandtolGuaranteed(scfres)
        ref = DFTK.solve_ΩplusK_split(scfres, -δHψ; tol=1e-12, verbose=false, bandtolalg)

        # Test agreement of non-interacting response
        δρ0 = apply_χ0(scfres, δVext, tol=1e-14).δρ
        @test maximum(abs, δρ0 - ref.δρ0) < 5e-11
        @test norm(δρ0 - ref.δρ0)         < 5e-11

        # Check residual is small
        ε = DFTK.DielectricAdjoint(scfres; bandtolalg=DFTK.BandtolGuaranteed(scfres))
        εδρ = reshape(DFTK.inexact_mul(ε, ref.δρ; tol=1e-14).Ax, size(δρ0))
        @test norm(δρ0 - εδρ)         < 1e-12
        @test maximum(abs, δρ0 - εδρ) < 1e-12

        # TODO δψ agreement not tested
        @assert length(scfres.basis.kpoints) == 1

        for tol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            res = DFTK.solve_ΩplusK_split(scfres, -δHψ; tol, verbose=false)
            @test norm(res.δρ - ref.δρ) < tol

            # This is probably a bit flaky ... orbitals could rotate
            @test maximum(abs, res.δψ[1] - ref.δψ[1]) < tol
        end

        for s in (1, 10, 100, 10^5)
            tol = 1e-8
            res = DFTK.solve_ΩplusK_split(scfres, -δHψ; tol, s, verbose=false)
            @test norm(res.δρ - ref.δρ) < tol
            @test maximum(abs, res.δψ[1] - ref.δψ[1]) < tol
        end
    end

    @testset "Polarisability in Helium" begin
        # Helium at the centre of the box
        a = 10.0
        lattice = a * I(3)  # cube of ``a`` bohrs
        pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
        atoms     = [ElementPsp(:He, pseudopotentials)]
        positions = [[1/2, 1/2, 1/2]]

        model  = model_DFT(lattice, atoms, positions;
                           functionals=PBE(), symmetries=false)
        basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(1, 1, 1))
        scfres = self_consistent_field(basis; tol=1e-12)

        # δVext is the potential from a uniform field interacting with the dielectric dipole
        # of the density. Note, that this expression only makes sense for a molecule
        δVext = [-(r[1] - a/2) for r in r_vectors_cart(basis)]
        δVext = cat(δVext; dims=4)

        test_solve_ΩplusK(scfres, δVext)
    end

    @testset "Aluminium atomic displacements" begin


        # TODO displace some atoms and solve response


    end
end
