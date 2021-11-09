using DFTK
using LinearAlgebra
using Test

@testset "spglib" begin
    a = 10.3
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    Ge = ElementPsp(:Ge, psp=load_psp("hgh/lda/Ge-q4"))

    # silicon
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 227
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # silicon with Cartesian x coordinates flipped
    lattice = a / 2 * [[0 -1 -1.]; [1 0 1.]; [1 1 0.]]
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 227
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # silicon with different lattice vectors
    lattice = a / 2 * [[0 1 1.]; [-1 0 1.]; [-1 1 0.]]
    atoms = [Si => [[-1, 1, 1]/8, -[-1, 1, 1]/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 227
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # Zincblende structure
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms = [Si => [ones(3)/8], Ge => [-ones(3)/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 216
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # Zincblende structure with Cartesian x coordinates flipped
    lattice = a / 2 * [[0 -1 -1.]; [1 0 1.]; [1 1 0.]]
    atoms = [Si => [ones(3)/8], Ge => [-ones(3)/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 216
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # Zincblende structure with different lattice vectors
    lattice = a / 2 * [[0 1 1.]; [-1 0 1.]; [-1 1 0.]]
    atoms = [Si => [[-1, 1, 1]/8], Ge => [-[-1, 1, 1]/8]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 216
    @test DFTK.get_spglib_lattice(model) ≈ a * I(3)

    # Hexagonal close packed lattice
    lattice = a * [[1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms = [Si => [[0, 0, 0], [1/3, 2/3, 1/2]]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 194
    @test DFTK.get_spglib_lattice(model) ≈ lattice

    # Hexagonal close packed lattice with x coordinates flipped
    lattice = a * [[-1. 1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms = [Si => [[0, 0, 0], [1/3, 2/3, 1/2]]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 194
    @test !(DFTK.get_spglib_lattice(model) ≈ lattice)

    # Hexagonal close packed lattice with different lattice vectors
    lattice = a * [[-1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms = [Si => [[0, 0, 0], [-1/3, 2/3, 1/2]]]
    model = model_LDA(lattice, atoms)
    @test DFTK.spglib_spacegroup_number(model) == 194
    @test DFTK.get_spglib_lattice(model) ≈ a * [[1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
end