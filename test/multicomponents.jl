# Note: The consistency tests at temperature may have an error of the order of magnitude
# of `default_occupation_threshold`; use `nbandsalg` if small tolerance.
@testsetup module Multicomponents
using Test
using DFTK
using LinearAlgebra

function test_consistency(setup; Ecut=5, kgrid=(4,4,4), tol=1e-5)
    res1 = setup(Ecut, kgrid, tol, 1)
    res2 = setup(Ecut, kgrid, tol, 2)
    @test norm(res1 - res2) < tol
end
end

@testitem "Energy & Density: LOBPCG" setup=[TestCases, Multicomponents] begin
    using DFTK
    test_consistency = Multicomponents.test_consistency
    testcase = TestCases.silicon
    testcase = merge(testcase, (; temperature=0.03))
    eigensolver = lobpcg_hyper

    test_consistency() do Ecut, kgrid, tol, n_components
        model = model_PBE(testcase.lattice, testcase.atoms, testcase.positions;
                          n_components=1, testcase.temperature)
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        nbandsalg = AdaptiveBands(model; occupation_threshold=tol/10)
        scfres = self_consistent_field(basis; eigensolver, tol, nbandsalg)
        [scfres.energies.total, scfres.ρ]
    end
end

@testitem "Energy & Density: diag_full" setup=[TestCases, Multicomponents] begin
    using DFTK
    test_consistency = Multicomponents.test_consistency
    testcase = TestCases.silicon
    testcase = merge(testcase, (; temperature=0.03))
    eigensolver = diag_full

    test_consistency() do Ecut, kgrid, tol, n_components
        model = model_PBE(testcase.lattice, testcase.atoms, testcase.positions;
                          n_components=1, testcase.temperature)
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        nbandsalg = AdaptiveBands(model; occupation_threshold=tol/10)
        scfres = self_consistent_field(basis; eigensolver, tol, nbandsalg)
        [scfres.energies.total, scfres.ρ]
    end
end

@testitem "Forces" setup=[TestCases, Multicomponents] begin
    using DFTK
    test_consistency = Multicomponents.test_consistency
    testcase = TestCases.silicon
    testcase = merge(testcase, (; temperature=0.03))

    test_consistency() do Ecut, kgrid, tol, n_components
        positions = [(ones(3)) / 4, -ones(3) / 8]
        model = model_PBE(testcase.lattice, testcase.atoms, positions; n_components,
                          testcase.temperature)
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        nbandsalg = AdaptiveBands(model; occupation_threshold=tol/100)
        scfres = self_consistent_field(basis; tol=tol/10, nbandsalg)
        compute_forces(scfres)
    end
end

@testitem "δρ, temperature" setup=[TestCases, Multicomponents] begin
    using DFTK
    using LinearAlgebra
    test_consistency = Multicomponents.test_consistency
    testcase = TestCases.aluminium_primitive
    testcase = merge(testcase, (; temperature=0.03))

    Ecut = 10
    kgrid = (1, 1, 1)

    test_consistency(; Ecut, kgrid) do Ecut, kgrid, tol, n_components
        model = model_PBE(testcase.lattice, testcase.atoms, testcase.positions;
                          n_components, testcase.temperature)
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        nbandsalg = AdaptiveBands(model; occupation_threshold=tol/10)
        scfres = self_consistent_field(basis; tol, nbandsalg)
        δV = guess_density(basis)
        apply_χ0(scfres, δV)
    end
end
