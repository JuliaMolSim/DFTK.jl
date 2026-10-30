@testitem "Adaptive depth Anderson" setup=[TestCases] begin
using DFTK
using LinearAlgebra
(; aluminium) = TestCases.all_testcases

function test_addiis(testcase; temperature=0, Ecut=10, kgrid=[3, 3, 3], n_bands=8)
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      functionals=LDA(), temperature)
    basis = PlaneWaveBasis(model; kgrid, Ecut)
    ρ = guess_density(basis)
    ψ = DFTK.random_orbitals(basis, n_bands)
    scfargs = (; ρ, ψ, tol=1e-10, mixing=SimpleMixing())

    solver = ScfAndersonDensitySolver(; errorfactor=Inf, maxcond=Inf, m=100)
    scfres_simple = self_consistent_field(basis; solver, seed=0x67a2ba06e7acfacb, scfargs...)

    solver = ScfAndersonDensitySolver(; errorfactor=Inf, maxcond=1e6, m=100)
    scfres_rdiis  = self_consistent_field(basis; solver, seed=0x67a2ba06e7acfacb, scfargs...)

    solver = ScfAndersonDensitySolver(; errorfactor=1e5, maxcond=Inf, m=100)
    scfres_addiis = self_consistent_field(basis; solver, seed=0x67a2ba06e7acfacb, scfargs...)

    @test scfres_simple.converged
    @test scfres_rdiis.converged
    @test scfres_addiis.converged
    @test norm(scfres_addiis.ρ -  scfres_rdiis.ρ) * sqrt(basis.dvol) < 5e-9
    @test norm(scfres_simple.ρ -  scfres_rdiis.ρ) * sqrt(basis.dvol) < 5e-9
    @test norm(scfres_addiis.ρ - scfres_simple.ρ) * sqrt(basis.dvol) < 5e-9
end

@testset "Aluminium, temp" begin
    test_addiis(aluminium; temperature=0.01)
end
end
