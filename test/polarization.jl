# TODO: make a GPU version as well
@testitem "Berry phase polarization of ZnO" tags=[:dont_test_mpi] begin
    using DFTK
    using PseudoPotentialData

    a = 6.1905
    c = 9.9886
    lattice = [a    -a/2    0.0;
               0.0  a*√3/2  0.0;
               0.0  0.0     c]
    pseudofamily = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Zn = ElementPsp(:Zn, pseudofamily)
    O = ElementPsp(:O, pseudofamily)
    atoms = [Zn, Zn, O, O]
    positions = [[1/3, 2/3, 0],
                [2/3, 1/3, 1/2],
                [1/3, 2/3, 0.3792],
                [2/3, 1/3, 1/2 + 0.3792]]

    model = model_DFT(lattice, atoms, positions; functionals=PBE())
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=[2, 2, 2])
    scfres = self_consistent_field(basis; tol=1e-6)

    scfres_unfolded = DFTK.unfold_bz(scfres; skip_hamiltonian=true)

    # TODO: change call to use direction 3 only!
    P_red_3 = DFTK.electronic_polarization(scfres_unfolded, DFTK.BerryPhasePolarization())[3]

    # Reference electronic phase computed with QE at the same very coarse settings
    qe_electronic_phase = 0.92020

    @test P_red_3 ≈ qe_electronic_phase rtol=1e-2
end
