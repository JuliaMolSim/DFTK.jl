using DFTK: bzmesh_uniform, bzmesh_ir_wedge, Species, Vec3, Mat3
using LinearAlgebra
using PyCall
using Test
include("testcases.jl")


@testset "bzmesh_uniform agrees with spglib" begin
    spglib = pyimport_conda("spglib", "spglib")

    function test_against_spglib(kgrid_size)
        kgrid_size = Vec3(kgrid_size)
        identity = [reshape(Mat3{Int}(I), 1, 3, 3)]
        _, grid = spglib.get_stabilized_reciprocal_mesh(kgrid_size, identity)
        kcoords_spglib = [Vec3{Int}(grid[ik, :]) .// kgrid_size for ik in 1:size(grid, 1)]
        sort!(kcoords_spglib)

        kcoords, _ = bzmesh_uniform(kgrid_size)
        sort!(kcoords)

        @test kcoords == kcoords_spglib
    end

    test_against_spglib([ 2,  3,  2])
    test_against_spglib([ 3,  3,  3])
    test_against_spglib([ 2,  3,  4])
    test_against_spglib([ 9, 11, 13])
end

@testset "bzmesh_ir_wedge is correct reduction" begin
    function test_reduction(system, kgrid_size)
        red_kcoords, _ = bzmesh_uniform(kgrid_size)

        irred_kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, system.lattice,
                                                 [Species(system.atnum) => system.positions])

        # Try to reproduce all kcoords from irred_kcoords
        all_kcoords = Vector{Vec3{Rational{Int}}}()
        for (ik, k) in enumerate(irred_kcoords)
            append!(all_kcoords, [S * k for (S, τ) in ksymops[ik]])
        end

        # Normalise the obtained k-Points and test for equality
        red_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in red_kcoords])
        all_kcoords = sort([mod.(k .* kgrid_size, kgrid_size) for k in all_kcoords])
        @test all_kcoords == red_kcoords
    end

    test_reduction(silicon, [ 2,  3,  2])
    test_reduction(silicon, [ 3,  3,  3])
    test_reduction(silicon, [ 2,  3,  4])
    test_reduction(silicon, [ 9, 11, 13])

    test_reduction(manganese, [ 2,  3,  2])
    test_reduction(manganese, [ 3,  3,  3])
    test_reduction(manganese, [ 2,  3,  4])
    test_reduction(manganese, [ 9, 11, 13])
end
