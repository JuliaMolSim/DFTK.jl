@testsetup module Hessian
using DFTK
using DFTK: select_occupied_orbitals, compute_projected_gradient

function setup_quantities(testcase)
    Ecut = 3
    fft_size = [9, 9, 9]
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions, [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, testcase.kcoords, testcase.kweights; fft_size)
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

@testitem "ΩplusK_split, 0K" tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK
    using DFTK: compute_projected_gradient
    using DFTK: select_occupied_orbitals, solve_ΩplusK, solve_ΩplusK_split
    using LinearAlgebra

    (; scfres, basis, ψ, occupation, ρ) = Hessian.setup_quantities(TestCases.silicon)
    rhs_ref = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
    ϕ = rhs_ref + scfres.ψ

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs_ref).δψ)),
                        real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs_ref)),
                        atol=1e-7)
    end

    @testset "solve_ΩplusK_split <=> solve_ΩplusK" begin
        scfres = self_consistent_field(basis; tol=1e-10)
        δψ1 = solve_ΩplusK_split(scfres, rhs_ref).δψ
        δψ1, _ = select_occupied_orbitals(basis, δψ1, scfres.occupation)
        ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)
        rhs, _ = select_occupied_orbitals(basis, rhs_ref, occupation)
        δψ2 = solve_ΩplusK(basis, ψ, rhs, occupation).δψ
        @test norm(δψ1 - δψ2) < 1e-7
    end
end

@testitem "ΩplusK_split, temperature" tags=[:dont_test_mpi] setup=[Hessian, TestCases] begin
    using DFTK
    using DFTK: compute_projected_gradient, solve_ΩplusK_split
    using LinearAlgebra

    (; scfres, basis, ψ, occupation, ρ) = Hessian.setup_quantities(TestCases.silicon)
    rhs = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
    ϕ = rhs + scfres.ψ

    @testset "self-adjointness of solve_ΩplusK_split" begin
        @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
                        real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
                        atol=1e-7)
    end
end
