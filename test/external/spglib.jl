@testitem "spglib" begin
    using DFTK
    using DFTK: spglib_dataset, spglib_standardize_cell
    using LinearAlgebra

    a = 10.3
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
    Ge = ElementPsp(:Ge; psp=load_psp("hgh/lda/Ge-q4"))

    # silicon
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 227
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # silicon with Cartesian x coordinates flipped
    lattice = a / 2 * [[0 -1 -1.]; [1 0 1.]; [1 1 0.]]
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 227
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # silicon with different lattice vectors
    lattice = a / 2 * [[0 1 1.]; [-1 0 1.]; [-1 1 0.]]
    atoms     = [Si, Si]
    positions = [[-1, 1, 1]/8, -[-1, 1, 1]/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 227
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # Zincblende structure
    lattice = a / 2 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms     = [Si, Ge]
    positions = [ones(3)/8, -ones(3)/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 216
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # Zincblende structure with Cartesian x coordinates flipped
    lattice = a / 2 * [[0 -1 -1.]; [1 0 1.]; [1 1 0.]]
    atoms     = [Si, Ge]
    positions = [ones(3)/8, -ones(3)/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 216
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # Zincblende structure with different lattice vectors
    lattice = a / 2 * [[0 1 1.]; [-1 0 1.]; [-1 1 0.]]
    atoms     = [Si, Ge]
    positions = [[-1, 1, 1]/8, -[-1, 1, 1]/8]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 216
    @test spglib_standardize_cell(model).lattice ≈ a * I(3)

    # Hexagonal close packed lattice
    lattice = a * [[1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms     = [Si, Si]
    positions = [[0, 0, 0], [1/3, 2/3, 1/2]]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 194
    @test spglib_standardize_cell(model).lattice ≈ lattice

    # Hexagonal close packed lattice with x coordinates flipped
    lattice = a * [[-1. 1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms     = [Si, Si]
    positions =  [[0, 0, 0], [1/3, 2/3, 1/2]]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 194
    @test !(spglib_standardize_cell(model).lattice ≈ lattice)

    # Hexagonal close packed lattice with different lattice vectors
    lattice = a * [[-1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]]
    atoms     = [Si, Si]
    positions =  [[0, 0, 0], [-1/3, 2/3, 1/2]]
    model = model_LDA(lattice, atoms, positions)
    @test spglib_dataset(atomic_system(model)).spacegroup_number == 194
    @test (  spglib_standardize_cell(model).lattice
           ≈ a * [[1. -1/2 0.]; [0. sqrt(3)/2 0.]; [0. 0. sqrt(8/3)]])
end
