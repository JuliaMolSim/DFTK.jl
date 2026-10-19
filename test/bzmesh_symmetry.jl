@testitem "Symmetrization and not symmetrization yields the same density and energy" #=
    =#    setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    testcase = TestCases.silicon

    cases = (MonkhorstPack([2, 2, 2], kshift=[1/2,   0,   0]),
             MonkhorstPack([2, 2, 2], kshift=[1/2, 1/2,   0]),
             MonkhorstPack([2, 2, 2], kshift=[  0,   0,   0]),
             MonkhorstPack([3, 2, 3], kshift=[  0,   0,   0]),
             MonkhorstPack([3, 2, 3], kshift=[  0, 1/2, 1/2]))
    for kgrid in cases
        model_nosym = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                                functionals=LDA(), symmetries=false)
        basis = PlaneWaveBasis(model_nosym; Ecut=5, kgrid)
        DFTK.check_group(basis.symmetries)

        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ1 = scfres.ρ
        E1 = scfres.energies.total

        model_sym = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                              functionals=LDA())
        basis = PlaneWaveBasis(model_sym; Ecut=5, kgrid) 
        DFTK.check_group(basis.symmetries)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ2 = scfres.ρ
        E2 = scfres.energies.total

        @test abs(E1 - E2) < 1e-10
        @test norm(ρ1 - ρ2) .* sqrt(basis.dvol) < 1e-8
    end
end

@testitem "SCF on a generically shifted k-grid" setup=[TestCases] begin
    # Tests that generically shifted SCF runs and that the results are not too far
    # off a case without a generic shift
    using DFTK
    using LinearAlgebra
    testcase = TestCases.silicon

    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      functionals=LDA())

    kgrid_size = [4, 5, 6]
    basis_noshift  = PlaneWaveBasis(model; Ecut=5, kgrid=MonkhorstPack(kgrid_size)
    scfres_noshift = self_consistent_field(basis_noshift; tol=1e-10)
    @test length(basis_noshift.symmetries) > 1

    basis_shift  = PlaneWaveBasis(model; Ecut=5,
                                  kgrid=MonkhorstPack(kgrid_size; kshift=[0.0, 0.1, 0.05])
    scfres_shift = self_consistent_field(basis_shift; tol=1e-10)
    @test length(basis_shift.symmetries) == 1  # Symmetries disabled

    @test abs(scfres_shift.energies.total - scfres_noshift.energies.total) < 1e-3
    @test norm(scfres_shift.ρ - scfres_noshift.ρ) < 1e-4
end
