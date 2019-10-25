using Test
using LinearAlgebra
using DFTK: determine_grid_size, model_free_electron

include("testcases.jl")

@testset "Test determine_grid_size on Silicon" begin
    @test determine_grid_size(silicon.lattice, 3, supersampling=2) == [15, 15, 15]
    @test determine_grid_size(silicon.lattice, 4, supersampling=2) == [15, 15, 15]
    @test determine_grid_size(silicon.lattice, 5, supersampling=2) == [18, 18, 18]
    @test determine_grid_size(silicon.lattice, 15, supersampling=2) == [27, 27, 27]
    @test determine_grid_size(silicon.lattice, 25, supersampling=2) == [36, 36, 36]
    @test determine_grid_size(silicon.lattice, 30, supersampling=2) == [40, 40, 40]

    # Test the model interface as well
    model = model_free_electron(silicon.lattice, silicon.n_electrons)
    @test determine_grid_size(model, 30) == [40, 40, 40]
    @test determine_grid_size(model, 30, supersampling=1.8) == [36, 36, 36]
end

@testset "Test determine_grid_size on skewed lattice" begin
    lattice = Diagonal([1, 1e-12, 1e-12])
    @test determine_grid_size(lattice, 15, supersampling=2) == [5, 1, 1]
    @test determine_grid_size(lattice, 300, supersampling=2) == [18, 1, 1]
end
