# Quick and dirty testset that runs the Base.show code of the most important
# data structures of DFTK. Point is not to test that the correct thing is printed,
# rather to ensure that the code does not randomly stop working.
@testitem "Test printing" setup=[TestCases] begin
    using DFTK
    magnesium = TestCases.magnesium

    function test_basis_printing(; modelargs=(; temperature=1e-3),
                                   basisargs=(; Ecut=5, kgrid=[1, 3, 2], kshift=[0, 0, 0]))
        model = model_LDA(magnesium.lattice, magnesium.atoms, magnesium.positions;
                          disable_electrostatics_check=true, modelargs...)
        basis = PlaneWaveBasis(model; basisargs...)

        println(model)
        show(stdout, "text/plain", model)

        println(basis)
        show(stdout, "text/plain", basis)

        println(basis.kpoints[1])
        show(stdout, "text/plain", basis.kpoints[1])

        basis
    end

    function test_scfres_printing(; kwargs...)
        basis = test_basis_printing(; kwargs...)
        scfres = self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=6),
                                       tol=1e-3)

        println(scfres.energies)
        show(stdout, "text/plain", scfres.energies)
    end

    test_scfres_printing()
    test_basis_printing(; modelargs=(; εF=0.5))
end

@testitem "versioninfo" begin
    using DFTK

    versioninfo = sprint(DFTK.versioninfo)
    @test occursin("Julia Version", versioninfo)
    @test occursin("DFTK Version", versioninfo)
    @test occursin("BLAS", versioninfo)
end
