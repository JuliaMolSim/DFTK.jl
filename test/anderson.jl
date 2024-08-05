@testitem "Adaptive depth Anderson" setup=[TestCases] begin
using DFTK
using LinearAlgebra
(; silicon, aluminium) = TestCases.all_testcases

function test_addiis(testcase; temperature=0, Ecut=10, kgrid=[3, 3, 3])
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions; temperature)
    basis = PlaneWaveBasis(model; kgrid, Ecut)
    tol   = 1e-6

    solver = scf_anderson_solver(; errorfactor=Inf, maxcond=1e6, m=100)
    scfres_rdiis = self_consistent_field(basis; tol, mixing=SimpleMixing(), solver)

    solver = scf_anderson_solver(; errorfactor=1e4, maxcond=Inf, m=100)
    scfres_addiis = self_consistent_field(basis; tol, mixing=SimpleMixing(), solver)

    @test norm(scfres_addiis.ρ - scfres_rdiis.ρ) * sqrt(basis.dvol) < 10tol
    @test scfres_addiis.n_iter ≤ scfres_rdiis.n_iter
end

@testset "Silicon, no temp" begin
    test_addiis(silicon; temperature=0)
end
@testset "Aluminium, temp" begin
    test_addiis(aluminium; temperature=0.03)
end
end
