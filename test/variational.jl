@testsetup module Variational
using DFTK

function get_scf_energies(testcase, supersampling, functionals)
    Ecut=3
    scf_tol=1e-12  # Tolerance in total enengy
    kgrid = ExplicitKpoints([[.2, .3, .4]])

    # force symmetries to false because the symmetrization is weird at low ecuts
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions, functionals;
                      symmetries=false)
    fft_size = compute_fft_size(model, Ecut, kgrid; supersampling,
                                ensure_smallprimes=false, algorithm=:precise)
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)
    scfres = self_consistent_field(basis; tol=scf_tol)
    values(scfres.energies)
end
end


@testitem "Energy is exact for supersampling>2 without XC" #=
    =#    setup=[Variational, TestCases] begin
    using LinearAlgebra: norm
    testcase = TestCases.silicon

    energies = [Variational.get_scf_energies(testcase, supersampling, [])
                for supersampling in (1, 2, 3)]

    @test abs(sum(energies[1]) - sum(energies[2])) > 1e-10

    # supersampling == 2 is exact and going beyond has no effect on the total energy
    @test abs(sum(energies[2]) - sum(energies[3])) < 1e-10

    # Individual terms are not variational and only converged to the amount
    # the density and the orbitals are converged, which is about sqrt(energy)
    @test norm(energies[2] .- energies[3]) < 1e-5
end

@testitem "Energy is not exact for supersampling>2 with XC" #=
    =#    setup=[Variational, TestCases] begin
    testcase = TestCases.silicon

    energies = [Variational.get_scf_energies(testcase, supersampling, [:lda_x, :lda_c_vwn])
                for supersampling in (1, 2, 3)]

    @test abs(sum(energies[1]) - sum(energies[2])) > 1e-10

    # supersampling == 2 is not exact in total energy for XC
    @test abs(sum(energies[2]) - sum(energies[3])) > 1e-10
end
