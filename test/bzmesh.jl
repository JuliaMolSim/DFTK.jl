using DFTK: bzmesh_uniform, bzmesh_ir_wedge, Element, Vec3, Mat3
using DFTK: pymatgen_structure, load_lattice
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
    function test_reduction(system, kgrid_size; supercell=[1, 1, 1])
        lattice = system.lattice
        atoms = [Element(system.atnum) => system.positions]
        if supercell != [1, 1, 1]  # Make a supercell
            pystruct = pymatgen_structure(lattice, atoms)
            pystruct.make_supercell(supercell)
            lattice = load_lattice(pystruct)
            atoms = [Element(system.atnum) => [s.frac_coords for s in pystruct.sites]]
        end

        red_kcoords, _ = bzmesh_uniform(kgrid_size)
        irred_kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, lattice, atoms)

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

    test_reduction(silicon, [ 1,  4,  4], supercell=[2, 1, 1])
    test_reduction(silicon, [ 1,  16,  16], supercell=[4, 1, 1])

    test_reduction(magnesium, [ 2,  3,  2])
    test_reduction(magnesium, [ 3,  3,  3])
    test_reduction(magnesium, [ 2,  3,  4])
    test_reduction(magnesium, [ 9, 11, 13])
end
