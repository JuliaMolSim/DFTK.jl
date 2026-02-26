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

    # Using the kinetic cutoff blowup suppresses the basis size discontinuities
    # with respect to strain. This allows us to verify numerical consistency,
    # while keeping the test cheap at low Ecut.
    model0 = model_DFT(lattice, atoms, silicon.positions; functionals=PBE(),
                        kinetic_blowup=BlowupCHV())
    Ecut = 5
    kgrid = (2, 2, 2)
    tol = 1e-8
    basis = PlaneWaveBasis(model0; Ecut, kgrid)

    scfres = self_consistent_field(basis; tol)
    (; voigt_stress, C) = elastic_tensor(scfres)

    @test size(voigt_stress) == (6,)
    @test size(C) == (6, 6)

    # AD-DFPT numbers from this same test (only useful for regression testing)
    C11 = 0.010145339323700887
    C12 = 0.003622281868389229
    C44 = 0.005064094595621912
    Cref = [C11 C12 C12 0   0   0;
             C12 C11 C12 0   0   0;
             C12 C12 C11 0   0   0;
             0   0   0   C44 0   0;
             0   0   0   0   C44 0;
             0   0   0   0   0   C44]
    @test isapprox(C, Cref; atol=tol)

    # TODO could also compare more converged values against ABINIT DFPT

    # Compare against finite difference (which needs much tighter SCF tol) of the stress
    h = 1e-5
    strain_pattern = [1, 0, 0, 1, 0, 0]
    strained_lattice = DFTK.voigt_strain_to_full(0.01 * strain_pattern) * model0.lattice
    stress_fn(η) = DFTK._stress_from_strain(basis, η; tol=1e-10)
    dstress_fd = (stress_fn(h * strain_pattern) - stress_fn(-h * strain_pattern)) / 2h

    @test C * strain_pattern ≈ dstress_fd   atol=h
end