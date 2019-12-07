using Test
using DFTK: Species, load_psp
using DFTK: energy_nuclear_ewald, energy_nuclear_psp_correction
using LinearAlgebra: Diagonal


@testset "energy_nuclear_ewald Lithium hydride" begin
    lattice = 16 * Diagonal(ones(3))
    hydrogen = Species(1)
    lithium = Species(3, psp=load_psp("hgh/lda/Li-q1"))
    composition = [
        hydrogen => [[1/2, 1/2, 0.5953697526034847]],
        lithium => [[1/2, 1/2, 0.40463024613039883]],
    ]

    ref = -0.02196861  # TODO source?
    γ_E = energy_nuclear_ewald(lattice, composition...)
    @test abs(γ_E - ref) < 1e-8
end

@testset "energy_nuclear_ewald silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    silicon = Species(14, psp=load_psp("hgh/lda/Si-q4"))
    composition = [silicon => [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]], ]

    ref = -8.39789357839024  # from ABINIT
    γ_E = energy_nuclear_ewald(lattice, composition...)
    @test abs(γ_E - ref) < 1e-10
end

@testset "energy_nuclear_psp_correction silicon" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    silicon = Species(14, psp=load_psp("hgh/lda/Si-q4"))
    composition = [silicon => [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]], ]

    ref = -0.294622067023269  # from ABINIT
    e_corr = energy_nuclear_psp_correction(lattice, composition...)
    @test abs(e_corr - ref) < 1e-10
end
