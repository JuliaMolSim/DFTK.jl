@testitem "Adaptive depth Anderson" setup=[TestCases] begin
using DFTK
using LinearAlgebra
(; silicon, aluminium) = TestCases.all_testcases

function test_addiis(testcase; temperature=0, Ecut=10, kgrid=[3, 3, 3], n_bands=8)
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      functionals=LDA(), temperature)
    basis = PlaneWaveBasis(model; kgrid, Ecut)
    tol   = 1e-10
    ρ = guess_density(basis)
    ψ = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]

    solver = scf_anderson_solver(; errorfactor=Inf, maxcond=Inf, m=100)
    scfres_simple = self_consistent_field(basis; ρ, ψ, tol, mixing=SimpleMixing(), solver)

    solver = scf_anderson_solver(; errorfactor=Inf, maxcond=1e6, m=100)
    scfres_rdiis = self_consistent_field(basis; ρ, ψ, tol, mixing=SimpleMixing(), solver)

    solver = scf_anderson_solver(; errorfactor=1e5, maxcond=Inf, m=100)
    scfres_addiis = self_consistent_field(basis; ρ, ψ, tol, mixing=SimpleMixing(), solver)

    @test norm(scfres_addiis.ρ - scfres_rdiis.ρ) * sqrt(basis.dvol) < 10tol
    @test norm(scfres_simple.ρ - scfres_rdiis.ρ) * sqrt(basis.dvol) < 10tol
end

@testset "Aluminium, temp" begin
    test_addiis(aluminium; temperature=0.01)
end
end
