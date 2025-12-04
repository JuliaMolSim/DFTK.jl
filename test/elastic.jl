@testitem "AD-DFPT elastic constants on silicon" setup=[TestCases] begin
    using DFTK
    using PseudoPotentialData
    using ForwardDiff
    using LinearAlgebra
    silicon = TestCases.silicon

    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    a0_pbe = 10.33
    lattice = a0_pbe / 2 * [[0 1 1];
                            [1 0 1];
                            [1 1 0]]
    atoms = fill(ElementPsp(silicon.atnum, load_psp(silicon.psp_gth)), 2)
    model0 = model_DFT(lattice, atoms, silicon.positions; functionals=PBE())

    Ecut = 18
    kgrid = (4, 4, 4)
    tol = 1e-8
    basis = PlaneWaveBasis(model0; Ecut, kgrid)

    scfres = self_consistent_field(basis; tol)
    (; voigt_stress, C) = DFTK.elastic_tensor(scfres)

    @test size(voigt_stress) == (6,)
    @test size(C) == (6, 6)
    # @test norm(C - Cref) < 3e-6  # ~0.1 GPa

    # Compare against finite difference of the stress
    h = 1e-5
    strain_pattern = [1, 0, 0, 1, 0, 0]
    symmetries_strain = DFTK.symmetries_from_strain(model0, 0.01 * strain_pattern)
    stress_fn(η) = DFTK.stress_from_strain(model0, η; symmetries=symmetries_strain,
                                           Ecut, kgrid, tol)
    dstress_fd = (stress_fn(h * strain_pattern) - stress_fn(-h * strain_pattern)) / 2h

    @test C * strain_pattern ≈ dstress_fd   atol=1e-6
end