@testitem "Symmetrization and not symmetrization yields the same density and energy" #=
    =#    setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    testcase = TestCases.silicon

    args = ((; kgrid=[2, 2, 2], kshift=[1/2, 0, 0]),
            (; kgrid=[2, 2, 2], kshift=[1/2, 1/2, 0]),
            (; kgrid=[2, 2, 2], kshift=[0, 0, 0]),
            (; kgrid=[3, 2, 3], kshift=[0, 0, 0]),
            (; kgrid=[3, 2, 3], kshift=[0, 1/2, 1/2]))
    for case in args
        model_nosym = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                                symmetries=false)
        basis = PlaneWaveBasis(model_nosym; Ecut=5, case...)
        DFTK.check_group(basis.symmetries)

        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ1 = scfres.ρ
        E1 = scfres.energies.total

        model_sym = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
        basis = PlaneWaveBasis(model_sym; Ecut=5, case...)
        DFTK.check_group(basis.symmetries)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ2 = scfres.ρ
        E2 = scfres.energies.total

        @test abs(E1 - E2) < 1e-10
        @test norm(ρ1 - ρ2) .* sqrt(basis.dvol) < 1e-8
    end
end
