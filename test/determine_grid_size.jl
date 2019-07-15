using Test
using DFTK: determine_grid_size

include("silicon_testcases.jl")

@testset "Test determine_grid_size" begin
    @test determine_grid_size(lattice, 3, kpoints=kpoints, supersampling=2) == 13
    @test determine_grid_size(lattice, 4, kpoints=kpoints, supersampling=2) == 13
    @test determine_grid_size(lattice, 5, kpoints=kpoints, supersampling=2) == 15
    @test determine_grid_size(lattice, 15, kpoints=kpoints, supersampling=2) == 27
    @test determine_grid_size(lattice, 25, kpoints=kpoints, supersampling=2) == 33
    @test determine_grid_size(lattice, 30, kpoints=kpoints, supersampling=2) == 37


    @test determine_grid_size(lattice, 30) == 35
    @test determine_grid_size(lattice, 30, kpoints=kpoints, supersampling=1.8) == 33
end
