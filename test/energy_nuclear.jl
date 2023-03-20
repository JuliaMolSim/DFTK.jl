using Test
using DFTK
using LinearAlgebra: Diagonal


@testset "energy_ewald Lithium hydride" begin
    lattice = 16 * Diagonal(ones(3))
    H  = ElementCoulomb(1)
    Li = ElementPsp(3, psp=PseudoPotentialIO.load_psp("hgh_lda_hgh", "li-q1.hgh"))
    atoms = [H, Li]
    positions = [
        [1/2, 1/2, 0.5953697526034847],
        [1/2, 1/2, 0.40463024613039883],
    ]

    ref = -0.02196861  # TODO source?
    γ_E = DFTK.energy_ewald(Model(lattice, atoms, positions; terms=[Ewald()]))
    @test abs(γ_E - ref) < 1e-8
end

@testset "energy_ewald silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(14, psp=PseudoPotentialIO.load_psp("hgh_lda_hgh", "si-q4.hgh"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]

    ref = -8.39789357839024  # from ABINIT
    γ_E = DFTK.energy_ewald(Model(lattice, atoms, positions; terms=[Ewald()]))
    @test abs(γ_E - ref) < 1e-10
end

@testset "energy_psp_correction silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(14, psp=PseudoPotentialIO.load_psp("hgh_lda_hgh", "si-q4.hgh"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]
    model = Model(lattice, atoms, positions; terms=[PspCorrection()])

    ref = -0.294622067023269  # from ABINIT
    e_corr = DFTK.energy_psp_correction(model)
    @test abs(e_corr - ref) < 1e-10
end
