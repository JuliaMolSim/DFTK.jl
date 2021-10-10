using Test
using DFTK
using LinearAlgebra: Diagonal


@testset "energy_ewald Lithium hydride" begin
    lattice = 16 * Diagonal(ones(3))
    hydrogen = ElementCoulomb(1)
    lithium = ElementPsp(3, psp=load_psp("hgh/lda/Li-q1"))
    atoms = [
        hydrogen => [[1/2, 1/2, 0.5953697526034847]],
        lithium => [[1/2, 1/2, 0.40463024613039883]],
    ]

    ref = -0.02196861  # TODO source?
    γ_E = DFTK.energy_ewald(Model(lattice; atoms=atoms, terms=[Ewald()]))
    @test abs(γ_E - ref) < 1e-8
end

@testset "energy_ewald silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    silicon = ElementPsp(14, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [silicon => [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]], ]

    ref = -8.39789357839024  # from ABINIT
    γ_E = DFTK.energy_ewald(Model(lattice; atoms=atoms, terms=[Ewald()]))
    @test abs(γ_E - ref) < 1e-10
end

@testset "energy_psp_correction silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    silicon = ElementPsp(14, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [silicon => [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]], ]

    ref = -0.294622067023269  # from ABINIT
    e_corr = DFTK.energy_psp_correction(lattice, atoms)
    @test abs(e_corr - ref) < 1e-10
end
