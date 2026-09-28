@testitem "energy_forces_ewald Lithium hydride" begin
    using DFTK
    using PseudoPotentialData
    using LinearAlgebra: Diagonal

    lattice = 16 * Diagonal(ones(3))
    H  = ElementCoulomb(1)
    Li = ElementPsp(:Li, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth"))
    atoms = [H, Li]
    positions = [
        [1/2, 1/2, 0.5953697526034847],
        [1/2, 1/2, 0.40463024613039883],
    ]

    ref = -0.02196861  # TODO source?
    γ_E = DFTK.energy_forces_ewald(lattice, charge_ionic.(atoms), positions).energy
    @test abs(γ_E - ref) < 1e-8
end

@testitem "energy_forces_ewald silicon" begin
    using DFTK
    using PseudoPotentialData

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]

    ref = -8.39789357839024  # from ABINIT
    γ_E = DFTK.energy_forces_ewald(lattice, charge_ionic.(atoms), positions).energy
    @test abs(γ_E - ref) < 1e-10
end

@testitem "energy_psp_correction silicon" begin
    using DFTK
    using PseudoPotentialData

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]
    model = Model(lattice, atoms, positions; terms=[PspCorrection()])

    ref = -0.294622067023269  # from ABINIT
    e_corr = DFTK.energy_psp_correction(model)
    @test abs(e_corr - ref) < 1e-10
end
