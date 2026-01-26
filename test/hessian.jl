@testmodule Hessian begin
using DFTK
using DFTK: select_occupied_orbitals, compute_projected_gradient
using AtomsBuilder
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using Test


function weighted_kdot(basis::PlaneWaveBasis, ϕ, ψ)
    DFTK.weighted_ksum(basis, [dot(ϕk, ψk) for (ϕk, ψk) in zip(ϕ, ψ)])
end

function setup_quantities(testcase)
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      functionals=[:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=3, kgrid=(3, 3, 3))
    scfres = self_consistent_field(basis; tol=10)

    ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

    ρ = compute_density(basis, ψ, occupation)
    rhs = compute_projected_gradient(basis, ψ, occupation)
    ϕ = DFTK.random_orbitals(basis, size(ψ[1], 2))

    (; scfres, basis, ψ, occupation, ρ, rhs, ϕ)
end

function test_solve_ΩplusK(scfres, δVext)
    # Compute a reference solution
    δHψ = DFTK.multiply_ψ_by_blochwave(scfres.basis, scfres.ψ, δVext)
    ref = DFTK.solve_ΩplusK_split(scfres, δHψ; verbose=true, s=1.0, tol=1e-12,
                                  bandtolalg=1e-6 * DFTK.BandtolGuaranteed(scfres))
    δρ0 = apply_χ0(scfres, δVext, tol=1e-13).δρ

    @testset "Agreement of non-interacting response" begin
        @test maximum(abs, δρ0 - ref.δρ0) < 1e-11
    end
    @testset "Residual is small" begin
        ε_adj = DFTK.DielectricAdjoint(scfres; bandtolalg=DFTK.BandtolGuaranteed(scfres))
        ε_adj_δρ = reshape(DFTK.mul_approximate(ε_adj, ref.δρ; tol=1e-13).Ax, size(δρ0))
        @test maximum(abs, δρ0 - ε_adj_δρ) < 1e-11
    end

    @testset "Adaptive algorithm yields desired tolerances" begin
        for tol in (1e-3, 1e-6, 1e-8, 1e-10)
            res = DFTK.solve_ΩplusK_split(scfres, δHψ; tol, verbose=false)
            @test maximum(abs, res.δρ - ref.δρ) < tol

            for ik in 1:length(scfres.basis.kpoints)
                @test maximum(abs, res.δψ[ik] - ref.δψ[ik]) < 8tol
            end
        end
    end

    @testset "Try very large value for s" begin
        tol = 1e-8
        res = DFTK.solve_ΩplusK_split(scfres, δHψ; tol, s=10^5, verbose=false)
        @test maximum(abs, res.δρ - ref.δρ) < tol

        for ik in 1:length(scfres.basis.kpoints)
            @test maximum(abs, res.δψ[ik] - ref.δψ[ik]) < 8tol
        end
    end  # testset
end  # function

function test_solve_ΩplusK_aluminium_displace(; Ecut=15, repeat=2)
    pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    system = bulk(:Al; cubic=true) * (repeat, 1, 1)
    model  = model_DFT(system; pseudopotentials, functionals=PBE(),
                       symmetries=false, smearing=Smearing.Gaussian(), temperature=1e-3)
    basis  = PlaneWaveBasis(model; Ecut, kgrid=(1, repeat, repeat))
    scfres = self_consistent_field(basis;
                                   tol=1e-13, mixing=KerkerMixing(), callback=identity)

    # Random displacement
    function shuttering_potential(ε::T) where {T}
        R = [[-0.38208735137492345, 0.6771285038851875, 0.8216198265778598],
             [0.47584302275188817, 0.16146729711581953, -0.8362498421291198],
             [-0.7875578631279081, 0.38498434793801706, 0.24066509135980163],
             [-0.1944040678498673, -0.0929778516688402, -0.4728393892106624],
             [0.5120807312623552, -0.856940256477597, 0.24358596168773672],
             [0.18433862295200498, 0.1864489996856027, -0.8793003414514147],
             [-0.9293646854065234, 0.29193708439091415, -0.02071246486656042],
             [-0.8866516343128119, -0.5739781361336571, 0.1948169665929571]]
        R = R[1:(4repeat)]
        @assert length(R) == length(model.positions)
        newpositions = model.positions + ε * R
        modelV = Model(Matrix{T}(model.lattice), model.atoms, newpositions;
                       terms=[DFTK.AtomicLocal(), DFTK.AtomicNonlocal()], symmetries=false)
        basisV = PlaneWaveBasis(modelV; Ecut=basis.Ecut, kgrid=basis.kgrid)
        DFTK.total_local_potential(Hamiltonian(basisV))
    end
    δV = ForwardDiff.derivative(shuttering_potential, 0.0)
    (; scfres, δV)
end

end


@testitem "self-adjointness of solve_ΩplusK" #=
    =#    tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK: solve_ΩplusK
    using LinearAlgebra
    (; basis, ψ, occupation, rhs, ϕ) = Hessian.setup_quantities(TestCases.silicon)

    @test isapprox(
        real(Hessian.weighted_kdot(basis, ϕ, solve_ΩplusK(basis, ψ, rhs, occupation).δψ)),
        real(Hessian.weighted_kdot(basis, solve_ΩplusK(basis, ψ, ϕ, occupation).δψ, rhs)),
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
        Hessian.weighted_kdot(basis, ϕ, apply_Ω(rhs, ψ, H, Λ)),
        Hessian.weighted_kdot(basis, apply_Ω(ϕ, ψ, H, Λ), rhs),
        atol=1e-14
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
        real(Hessian.weighted_kdot(basis, ϕ, apply_K(basis, rhs, ψ, ρ, occupation))),
        real(Hessian.weighted_kdot(basis, apply_K(basis, ϕ, ψ, ρ, occupation), rhs)),
        atol=1e-14
    )
end

@testitem "ΩplusK_split, 0K" tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
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
    ϕ = DFTK.random_orbitals(basis, size(scfres.ψ[1], 2))

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(
            real(Hessian.weighted_kdot(basis, ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
            real(Hessian.weighted_kdot(basis, solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
            atol=1e-7
        )
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
    ϕ = DFTK.random_orbitals(basis, size(ψ[1], 2))

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(
            real(Hessian.weighted_kdot(basis, ϕ, solve_ΩplusK_split(scfres, -rhs).δψ)),
            real(Hessian.weighted_kdot(basis, solve_ΩplusK_split(scfres, -ϕ).δψ, rhs)),
            atol=1e-7
        )
    end
end

@testitem "solve_ΩplusK_split achieves accuracy" setup=[Hessian] begin
    @testset "Aluminium atomic displacements (cheap)" begin
        (; scfres, δV) = Hessian.test_solve_ΩplusK_aluminium_displace(; Ecut=8, repeat=2)
        Hessian.test_solve_ΩplusK(scfres, δV)
    end

    # TODO Some other response test would be good
end

@testitem "solve_ΩplusK_split achieves accuracy (slow)" tags=[:slow] setup=[Hessian] begin
    @testset "Aluminium atomic displacements" begin
        (; scfres, δV) = Hessian.test_solve_ΩplusK_aluminium_displace(; Ecut=15, repeat=2)
        Hessian.test_solve_ΩplusK(scfres, δV)
    end
end
